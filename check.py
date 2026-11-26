"""Record a reference DNeo rollout, replay its rejected branches, and check the next draft consumes them.

Arms per rejecting round k (memory consumed by the draft call of round k+1):
  runtime  = `DNeoRuntime.memory` captured live (incremental verify forward with KV cache)
  indep    = causal forward of committed[:fork] + branch
  masked   = one masked prefill of the whole committed sequence + every branch
The next draft runs with the same prefix and anchor; only the memory source changes.
"""

from __future__ import annotations

import torch

from replay import (
    branches_from_trace,
    draft_hidden,
    independent_branch_features,
    masked_prefill_features,
    target_features,
)


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def per_layer(a, b, layer_ids):
    return {str(layer): rel_l2(a[:, :, layer], b[:, :, layer]) for layer in layer_ids}


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max())


def record_rollout(runtime, prompt_ids, **generate_kwargs):
    """Run `generate(trace=True)` and log every draft call: prefix length, anchor, live memory, draft hidden."""
    calls = []
    original = runtime.draft_hidden

    def logged(anchor):
        hidden = original(anchor)
        calls.append(
            dict(
                prefix_length=runtime.prefix_length,
                anchor=int(anchor.reshape(-1)[0]),
                memory=None if runtime.memory is None else runtime.memory.clone(),
                memory_positions=None if runtime.memory_positions is None else runtime.memory_positions.clone(),
                hidden=hidden.clone(),
            )
        )
        return hidden

    runtime.draft_hidden = logged
    try:
        result = runtime.generate(prompt_ids, trace=True, **generate_kwargs)
    finally:
        del runtime.draft_hidden
    # The reference runtime only appends to raw_features, so the final tensor holds every earlier prefix.
    return result, calls, runtime.raw_features


def _perturb_valid_row(memory, donor):
    row = memory.shape[1] // 2
    perturbed = memory.clone()
    perturbed[:, row] = donor
    return perturbed


def _pad(memory, positions, extra, fill):
    pad = memory.new_full((memory.shape[0], extra, *memory.shape[2:]), fill)
    valid = torch.cat(
        (
            torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device),
            torch.zeros(memory.shape[0], extra, dtype=torch.bool, device=memory.device),
        ),
        1,
    )
    pad_positions = positions[:, -1:] + 1 + torch.arange(extra, device=positions.device)[None]
    return torch.cat((memory, pad), 1), torch.cat((positions, pad_positions), 1), valid


def evaluate(target, model, sequence, trace, calls=None, final_prefix=None, pad=3):
    """Per-round measurements. `calls`/`final_prefix` come from `record_rollout`; without them the prefix is
    taken from an independent forward and runtime arms are skipped. Aggregate feature errors span every
    captured target layer (what the reference runtime stores); `feature_layers` splits them by the
    checkpoint's selected layers."""
    device = target.model.embed_tokens.weight.device
    layer_ids = list(model.config.target_layer_ids)
    branches = branches_from_trace(sequence, trace)
    rejecting = [b for b in branches if b is not None]
    _, masked_all = masked_prefill_features(target, sequence, rejecting)
    masked_by_round = {b.round: f for b, f in zip(rejecting, masked_all)}
    if calls is not None and len(calls) != len(trace):
        raise ValueError(f"{len(calls)} draft calls for {len(trace)} verify rounds")
    rounds, stale = [], None
    for index, record in enumerate(trace):
        fork = record["anchor_index"] + 1 + record["accepted"]
        if fork >= len(sequence) or (calls is not None and index + 1 >= len(calls)):
            continue
        anchor = int(sequence[fork])
        branch = branches[index]
        call = None if calls is None else calls[index + 1]
        out = dict(round=index, fork=fork, accepted=record["accepted"], gamma=len(record["proposed"]))
        out["after_rejection"] = index > 0 and branches[index - 1] is not None
        if call is not None:
            out["runtime_prefix_matches"] = call["prefix_length"] == fork and call["anchor"] == anchor
            prefix = final_prefix[:, :fork]
        else:
            prefix = target_features(target, torch.tensor(sequence[:fork], device=device)[None])
        if branch is None:
            out["kind"] = "all_accepted"
            base = draft_hidden(model, prefix, anchor)
            if call is not None:
                out["runtime_memory_cleared"] = call["memory"] is None
                out["runtime_hidden_equal_no_memory"] = bool(torch.equal(base, call["hidden"]))
            if stale is not None:
                memory, positions = stale
                out["stale_memory_effect"] = rel_l2(draft_hidden(model, prefix, anchor, memory, positions), base)
            rounds.append(out)
            continue
        out["kind"] = "rejected"
        out["memory_rows"] = len(branch.token_ids)
        positions = torch.tensor(list(branch.positions), device=device)[None]
        indep = independent_branch_features(target, sequence, branch)
        masked = masked_by_round[index]
        out["feature_masked_vs_indep"] = rel_l2(masked, indep)
        out["feature_masked_vs_indep_max_abs"] = max_abs(masked, indep)
        layers = {"masked_vs_indep": per_layer(masked, indep, layer_ids)}
        h_indep = draft_hidden(model, prefix, anchor, indep, positions)
        h_masked = draft_hidden(model, prefix, anchor, masked, positions)
        out["hidden_masked_vs_indep"] = rel_l2(h_masked, h_indep)
        out["hidden_masked_vs_indep_max_abs"] = max_abs(h_masked, h_indep)
        out["memory_effect"] = rel_l2(draft_hidden(model, prefix, anchor), h_masked)
        donor = prefix[:, fork // 2]
        out["valid_row_perturbation_effect"] = rel_l2(
            draft_hidden(model, prefix, anchor, _perturb_valid_row(masked, donor), positions), h_masked
        )
        padded_a, padded_positions, valid = _pad(masked, positions, pad, 0.0)
        padded_b, _, _ = _pad(masked, positions, pad, 1e4)
        h_pad_a = draft_hidden(model, prefix, anchor, padded_a, padded_positions, valid)
        h_pad_b = draft_hidden(model, prefix, anchor, padded_b, padded_positions, valid)
        out["masked_rows_ignored"] = bool(torch.equal(h_pad_a, h_pad_b))
        out["padded_vs_unpadded"] = rel_l2(h_pad_a, h_masked)
        if call is not None:
            runtime_memory = call["memory"]
            out["runtime_positions_match"] = (
                call["memory_positions"] is not None and call["memory_positions"].tolist() == positions.tolist()
            )
            out["feature_runtime_vs_indep"] = rel_l2(runtime_memory, indep)
            out["feature_masked_vs_runtime"] = rel_l2(masked, runtime_memory)
            layers["runtime_vs_indep"] = per_layer(runtime_memory, indep, layer_ids)
            layers["masked_vs_runtime"] = per_layer(masked, runtime_memory, layer_ids)
            h_runtime = draft_hidden(model, prefix, anchor, runtime_memory, positions)
            out["runtime_hidden_reproduced"] = bool(torch.equal(h_runtime, call["hidden"]))
            out["hidden_runtime_vs_indep"] = rel_l2(h_runtime, h_indep)
            out["hidden_masked_vs_runtime"] = rel_l2(h_masked, call["hidden"])
        out["feature_layers"] = layers
        stale = (masked, positions)
        rounds.append(out)
    return rounds


def _limit(tol, r, key):
    return tol(r, key) if callable(tol) else tol


def verdict(rounds, feature_tol, hidden_tol, runtime_tol=None, effect_ratio=10.0):
    """Pre-registered pass rule (PREREG.md). A tolerance is a number or `f(round, key)`."""
    failures = []
    rejected = [r for r in rounds if r["kind"] == "rejected"]
    accepted = [r for r in rounds if r["kind"] == "all_accepted"]
    if not rejected:
        failures.append("no rejecting round consumed by a later draft")
    for r in rejected:
        tag = f"round {r['round']}"
        limits = [("feature_masked_vs_indep", feature_tol), ("hidden_masked_vs_indep", hidden_tol)]
        if runtime_tol is not None and "feature_masked_vs_runtime" in r:
            limits += [("feature_masked_vs_runtime", runtime_tol), ("hidden_masked_vs_runtime", runtime_tol)]
        for key, tol in limits:
            if r[key] > _limit(tol, r, key):
                failures.append(f"{tag}: {key} {r[key]:.3e} > {_limit(tol, r, key):.3e}")
        floor = effect_ratio * r["hidden_masked_vs_indep"]
        if not r["valid_row_perturbation_effect"] > floor:
            failures.append(f"{tag}: valid-row perturbation {r['valid_row_perturbation_effect']:.3e} <= {floor:.3e}")
        if not r["masked_rows_ignored"]:
            failures.append(f"{tag}: changing masked rows changed the draft")
        for key in ("runtime_prefix_matches", "runtime_positions_match", "runtime_hidden_reproduced"):
            if key in r and not r[key]:
                failures.append(f"{tag}: {key} is false")
    for r in accepted:
        tag = f"round {r['round']}"
        for key in ("runtime_prefix_matches", "runtime_memory_cleared", "runtime_hidden_equal_no_memory"):
            if key in r and not r[key]:
                failures.append(f"{tag}: {key} is false")
        if "stale_memory_effect" in r and not r["stale_memory_effect"] > 0:
            failures.append(f"{tag}: stale memory has no effect, so the clear check has no power")
    return not failures, failures
