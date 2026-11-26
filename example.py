"""Small CPU example of the replay contract on tiny random models (no checkpoint, no GPU).

Records one reference DNeo rollout, writes it in the `report.rollouts.json` format, reads it back, replays it with
`replay_rollout`, and prints one line per rejecting round next to the runtime's live memory for that round.

usage: DNEO_SRC=<DNeo checkout>/src python example.py [rollout.json]
"""

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests"), os.environ["DNEO_SRC"]]
torch.cuda.synchronize = lambda *a, **k: None

from conftest import tiny_pair  # noqa: E402

from check import record_rollout, rel_l2  # noqa: E402
from dneo.runtime.dneo import DNeoRuntime  # noqa: E402
from replay import Branch, independent_branch_features, replay_rollout  # noqa: E402


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "example_out/rollout.json")
    target, model = tiny_pair()
    runtime = DNeoRuntime(target, model, variant="reference", context_capacity=256, draft_kernel="reference")
    prompt = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(0))
    result, calls, _ = record_rollout(runtime, prompt, max_new_tokens=48, seed=0, gamma=3)
    runtime.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dict(sequence=prompt[0].tolist() + result["token_ids"], trace=result["trace"])]))

    rollout = json.loads(path.read_text())[0]
    sequence, trace = rollout["sequence"], rollout["trace"]
    memories = replay_rollout(target, sequence, trace)
    print(f"{path}: {len(sequence)} tokens, {len(trace)} verify rounds, {len(memories)} rejecting")
    for m in memories:
        record = trace[m.round]
        indep = independent_branch_features(target, sequence, Branch(m.round, m.positions[0], m.token_ids))
        live = calls[m.round + 1]["memory"] if m.round + 1 < len(calls) else None
        print(
            f"round {m.round:2d}: accepted {record['accepted']}/{len(record['proposed'])}"
            f"  positions {m.positions[0]}..{m.positions[-1]}  features {tuple(m.features.shape)}"
            f"  vs independent {rel_l2(m.features, indep):.1e}"
            f"  vs runtime {'-' if live is None else f'{rel_l2(m.features, live):.1e}'}"
        )


if __name__ == "__main__":
    main()
