import json

import torch
from test_replay import _covering_rollout

from check import rel_l2
from replay import Branch, independent_branch_features, replay_rollout

TOL = 1e-5


def test_replay_rollout_keeps_rounds_positions_and_layers(pair):
    target, model = pair
    sequence, trace, calls, _ = _covering_rollout(target, model)
    sequence, trace = json.loads(json.dumps([sequence, trace]))
    memories = replay_rollout(target, sequence, trace)
    rejecting = [i for i, r in enumerate(trace) if r["accepted"] < len(r["proposed"])]
    assert [m.round for m in memories] == rejecting
    hidden = target.config.hidden_size
    for m in memories:
        record = trace[m.round]
        assert m.positions == tuple(record["memory_positions"])
        assert m.token_ids == tuple(record["proposed"][record["accepted"] :])
        assert m.features.shape == (1, len(m.positions), len(target.model.layers), hidden)
        branch = Branch(m.round, m.positions[0], m.token_ids)
        assert rel_l2(m.features, independent_branch_features(target, sequence, branch)) <= TOL
        if m.round + 1 < len(calls):
            live = calls[m.round + 1]
            assert live["memory_positions"].tolist() == [list(m.positions)]
            assert rel_l2(m.features, live["memory"]) <= TOL


def test_fully_accepted_rollout_has_no_memory(pair):
    target, _ = pair
    sequence = torch.randint(0, 64, (12,)).tolist()
    trace = [dict(anchor_index=5, proposed=sequence[6:9], accepted=3, memory_positions=[])]
    assert replay_rollout(target, sequence, trace) == []
