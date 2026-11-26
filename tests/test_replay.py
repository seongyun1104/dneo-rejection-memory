import pytest
import torch

import check
import replay
from check import evaluate, record_rollout, rel_l2, verdict
from replay import Branch, branches_from_trace, independent_branch_features, masked_prefill_features

FP32_FEATURE_TOL = FP32_HIDDEN_TOL = 1e-5
FP32_RUNTIME_TOL = 1e-5


def test_branches_follow_the_runtime_acceptance_boundary():
    sequence = [9, 9, 9, 10, 11, 30, 40]
    trace = [
        dict(anchor_index=2, proposed=[10, 11, 12, 13], accepted=2, memory_positions=[5, 6]),
        dict(anchor_index=5, proposed=[40], accepted=1, memory_positions=[]),
    ]
    first, second = branches_from_trace(sequence, trace)
    assert first == Branch(0, 5, (12, 13)) and list(first.positions) == [5, 6]
    assert second is None
    with pytest.raises(ValueError):
        branches_from_trace([9, 9, 9, 10, 99], trace[:1])
    with pytest.raises(ValueError):
        branches_from_trace(sequence, [dict(trace[0], memory_positions=[4, 5])])


def test_masked_prefill_layout():
    ids, positions, mask = replay.masked_prefill_inputs(
        [1, 2, 3, 4], [Branch(0, 2, (7, 8)), Branch(1, 4, (9,))], torch.float32
    )
    assert ids.tolist() == [[1, 2, 3, 4, 7, 8, 9]]
    assert positions.tolist() == [[0, 1, 2, 3, 2, 3, 4]]
    allowed = (mask[0, 0] == 0).int().tolist()
    assert allowed == [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1],
    ]


def test_hook_capture_equals_reference_features(pair):
    from dneo.runtime.feature_layout import NormalizedFeatureCapture

    target, _ = pair
    ids = torch.randint(0, 64, (1, 12))
    capture = NormalizedFeatureCapture(target.model, (0, 3))
    try:
        with torch.no_grad():
            target.model(input_ids=ids)
        hooked = capture.stack()
    finally:
        capture.close()
    assert torch.equal(hooked, replay.target_features(target, ids)[:, :, [0, 3]])


def test_masked_prefill_equals_independent_forwards(pair):
    target, _ = pair
    torch.manual_seed(1)
    sequence = torch.randint(0, 64, (40,)).tolist()
    branches = [
        Branch(0, 7, tuple(torch.randint(0, 64, (3,)).tolist())),
        Branch(1, 7, tuple(torch.randint(0, 64, (2,)).tolist())),
        Branch(2, 25, tuple(torch.randint(0, 64, (5,)).tolist())),
        Branch(3, 40, tuple(torch.randint(0, 64, (1,)).tolist())),
    ]
    committed, features = masked_prefill_features(target, sequence, branches)
    assert rel_l2(committed, replay.target_features(target, torch.tensor(sequence)[None])) <= FP32_FEATURE_TOL
    for branch, masked in zip(branches, features):
        assert rel_l2(masked, independent_branch_features(target, sequence, branch)) <= FP32_FEATURE_TOL


def _rollout(target, model, seed=0, max_new_tokens=48, gamma=3):
    from dneo.runtime.dneo import DNeoRuntime

    runtime = DNeoRuntime(target, model, variant="reference", context_capacity=256, draft_kernel="reference")
    prompt = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(seed))
    result, calls, prefix = record_rollout(runtime, prompt, max_new_tokens=max_new_tokens, seed=seed, gamma=gamma)
    runtime.close()
    return prompt[0].tolist() + result["token_ids"], result["trace"], calls, prefix


def _covering_rollout(target, model):
    """First seed whose rollout rejects at 0 and at >0 accepted and fully accepts right after a rejection.
    Searched rather than fixed: random init and sampling differ across torch/transformers versions."""
    for seed in range(40):
        sequence, trace, calls, prefix = _rollout(target, model, seed=seed)
        counts = {(r["accepted"] == len(r["proposed"]), r["accepted"] > 0) for r in trace}
        after = any(
            a["accepted"] < len(a["proposed"]) and b["accepted"] == len(b["proposed"])
            for a, b in zip(trace, trace[1:-1])
        )
        if {(False, False), (False, True)} <= counts and after:
            return sequence, trace, calls, prefix
    raise AssertionError("no seed in 0..39 covers both rejection kinds and accept-after-reject")


def test_recorded_rollout_replays_into_the_next_draft(pair):
    target, model = pair
    sequence, trace, calls, prefix = _covering_rollout(target, model)
    rounds = evaluate(target, model, sequence, trace, calls, prefix)
    passed, failures = verdict(rounds, FP32_FEATURE_TOL, FP32_HIDDEN_TOL, FP32_RUNTIME_TOL)
    assert passed, failures
    rejected = [r for r in rounds if r["kind"] == "rejected"]
    assert 0 in {r["accepted"] for r in rejected} and max(r["accepted"] for r in rejected) > 0
    assert all(r["memory_effect"] > 1e-3 for r in rejected)
    after_rejection = [r for r in rounds if r["kind"] == "all_accepted" and r["after_rejection"]]
    assert after_rejection and all(r["stale_memory_effect"] > 1e-3 for r in after_rejection)


def test_standalone_replay_without_runtime(pair):
    target, model = pair
    sequence, trace, _, _ = _covering_rollout(target, model)
    passed, failures = verdict(evaluate(target, model, sequence, trace), FP32_FEATURE_TOL, FP32_HIDDEN_TOL)
    assert passed, failures


def _shift_positions(sequence, branches, dtype, original=replay.masked_prefill_inputs):
    ids, positions, mask = original(sequence, branches, dtype)
    positions[:, len(sequence) :] -= 1
    return ids, positions, mask


def _leak_future(sequence, branches, dtype, original=replay.masked_prefill_inputs):
    ids, positions, mask = original(sequence, branches, dtype)
    mask[..., len(sequence) :, : len(sequence)] = 0
    return ids, positions, mask


def _isolate_branches(sequence, branches, dtype, original=replay.masked_prefill_inputs):
    ids, positions, mask = original(sequence, branches, dtype)
    mask[..., len(sequence) :, : len(sequence)] = torch.finfo(dtype).min
    return ids, positions, mask


def _raw_features(backbone, hidden_states, **kwargs):
    return torch.stack(hidden_states[: len(backbone.layers)], dim=2)


def _draft_ignores_memory(model, prefix, anchor, memory=None, positions=None, valid=None, original=replay.draft_hidden):
    return original(model, prefix, anchor)


def _draft_drops_valid_mask(
    model, prefix, anchor, memory=None, positions=None, valid=None, original=replay.draft_hidden
):
    return original(model, prefix, anchor, memory, positions)


@pytest.mark.parametrize(
    "module, name, broken",
    [
        (replay, "masked_prefill_inputs", _shift_positions),
        (replay, "masked_prefill_inputs", _leak_future),
        (replay, "masked_prefill_inputs", _isolate_branches),
        (replay, "target_attention_input_features", _raw_features),
        (check, "draft_hidden", _draft_ignores_memory),
        (check, "draft_hidden", _draft_drops_valid_mask),
    ],
    ids=[
        "positions_off_by_one",
        "branch_sees_later_committed",
        "branch_without_prefix",
        "raw_not_normalized",
        "draft_ignores_memory",
        "draft_drops_valid_mask",
    ],
)
def test_broken_replays_are_detected(pair, monkeypatch, module, name, broken):
    target, model = pair
    sequence, trace, calls, prefix = _covering_rollout(target, model)
    monkeypatch.setattr(module, name, broken)
    passed, failures = verdict(
        evaluate(target, model, sequence, trace, calls, prefix), FP32_FEATURE_TOL, FP32_HIDDEN_TOL, FP32_RUNTIME_TOL
    )
    assert not passed
