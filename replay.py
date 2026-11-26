"""Rebuild DNeo rejection memory from recorded rollouts with one masked target prefill.

Reference: J-sandler/DNeo 8449a3ce89 (model files identical to HF J-sandler/Qwen3-4B-DNeo 8451a2e8).
Features follow `HFTarget.forward` of the reference runtime: `target_attention_input_features`, i.e. every
target layer's `input_layernorm` output. The memory update follows `DNeoRuntime.generate` (dneo.py L860):
after a verify round at anchor position P with `count` accepted of `gamma`, the rejected drafts
`proposed[count:]` keep their original positions `P+1+count .. P+gamma`; a fully accepted round clears memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dneo.reference import target_attention_input_features


@dataclass(frozen=True)
class Branch:
    round: int
    fork: int
    token_ids: tuple[int, ...]

    @property
    def positions(self) -> range:
        return range(self.fork, self.fork + len(self.token_ids))


def branches_from_trace(sequence, trace):
    """One Branch per verify round that rejected something; `None` for fully accepted rounds."""
    branches = []
    for index, record in enumerate(trace):
        anchor, count, proposed = record["anchor_index"], record["accepted"], record["proposed"]
        fork = anchor + 1 + count
        committed = list(sequence[anchor + 1 : anchor + 1 + count])
        if committed != list(proposed[: len(committed)]):
            raise ValueError(f"round {index}: accepted drafts are not in the committed sequence")
        rejected = tuple(proposed[count:])
        if list(record["memory_positions"]) != list(range(fork, fork + len(rejected))):
            raise ValueError(f"round {index}: recorded memory positions disagree with the acceptance boundary")
        branches.append(Branch(index, fork, rejected) if rejected else None)
    return branches


def masked_prefill_inputs(sequence, branches, dtype):
    """Committed tokens followed by every branch; each branch sees only committed[:fork] and itself."""
    committed = len(sequence)
    ids, positions, owners = list(sequence), list(range(committed)), [(-1, 0)] * committed
    for slot, branch in enumerate(branches):
        if branch.fork > committed:
            raise ValueError(f"branch of round {branch.round} forks after the committed sequence")
        ids += branch.token_ids
        positions += branch.positions
        owners += [(slot, branch.fork)] * len(branch.token_ids)
    total = len(ids)
    position = torch.tensor(positions)
    owner = torch.tensor([o for o, _ in owners])
    fork = torch.tensor([f for _, f in owners])
    index = torch.arange(total)
    key_committed = index[None] < committed
    same_branch = (owner[:, None] == owner[None]) & (owner[:, None] >= 0)
    causal_in_row = index[None] <= index[:, None]
    allowed = torch.where(
        (owner >= 0)[:, None],
        (key_committed & (index[None] < fork[:, None])) | (same_branch & causal_in_row),
        key_committed & causal_in_row,
    )
    mask = torch.zeros(total, total, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)
    return torch.tensor(ids)[None], position[None], mask[None, None]


@torch.no_grad()
def target_features(target, input_ids, position_ids=None, attention_mask=None):
    """[batch, tokens, num_target_layers, hidden] normalized attention inputs, as the reference HFTarget."""
    backbone = target.model
    output = backbone(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return target_attention_input_features(backbone, output.hidden_states)


def masked_prefill_features(target, sequence, branches):
    """(committed features [1, len(sequence), L, H], per-branch features [1, len(branch), L, H])."""
    device = target.model.embed_tokens.weight.device
    dtype = target.model.embed_tokens.weight.dtype
    ids, positions, mask = masked_prefill_inputs(sequence, branches, dtype)
    features = target_features(target, ids.to(device), positions.to(device), mask.to(device))
    committed = features[:, : len(sequence)]
    out, start = [], len(sequence)
    for branch in branches:
        out.append(features[:, start : start + len(branch.token_ids)])
        start += len(branch.token_ids)
    return committed, out


@dataclass(frozen=True, eq=False)
class RejectionMemory:
    round: int
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    features: torch.Tensor


def replay_rollout(target, sequence, trace):
    """Rejection memory of every rejecting round, in trace order; fully accepted rounds have none.

    `features` is [1, len(positions), num_target_layers, hidden], the tensor `DNeoRuntime.memory` holds after
    that round and the draft of round + 1 consumes.
    """
    branches = [b for b in branches_from_trace(sequence, trace) if b is not None]
    if not branches:
        return []
    _, features = masked_prefill_features(target, sequence, branches)
    return [RejectionMemory(b.round, tuple(b.positions), b.token_ids, f) for b, f in zip(branches, features)]


def independent_branch_features(target, sequence, branch):
    """Plain causal forward of committed[:fork] + branch; returns only the branch rows."""
    device = target.model.embed_tokens.weight.device
    ids = torch.tensor(list(sequence[: branch.fork]) + list(branch.token_ids), device=device)[None]
    return target_features(target, ids)[:, branch.fork :]


@torch.no_grad()
def draft_hidden(model, prefix_features, anchor_token, memory=None, memory_positions=None, memory_valid=None):
    """Next-round draft hidden through the reference Conditioner + DraftBackbone (DNeoDraftModel.forward).

    Same call as `DNeoRuntime.draft_hidden` (reference variant): block = [anchor, mask * (b-1)] at position
    len(prefix); memory routed by the shared MoFA router and read as attention K/V. Returns [batch, b-1, H].
    """
    batch, length = prefix_features.shape[:2]
    device = prefix_features.device
    ids = torch.full((batch, 1, model.block_size), model.mask_token_id, device=device, dtype=torch.long)
    ids[:, :, 0] = anchor_token
    embeddings = model.embed_tokens(ids)
    anchors = torch.full((batch, 1), length, device=device)
    if memory is None:
        output = model(embeddings, prefix_features, anchors)
    else:
        if memory_valid is None:
            memory_valid = torch.ones(memory.shape[:2], device=device, dtype=torch.bool)
        output = model(
            embeddings, prefix_features, anchors, memory[:, None], memory_valid[:, None], memory_positions[:, None]
        )
    return output[:, 0, 1:]
