# DNeo rejection memory replay

Rebuilds DNeo's rejection memory from a recorded rollout with one masked target prefill, and checks that the
reference DNeo drafter consumes it like the live runtime memory.

Reference implementation: [J-sandler/DNeo](https://github.com/J-sandler/DNeo) `8449a3ce891f`. Its model files
are byte-identical to HF `J-sandler/Qwen3-4B-DNeo` `8451a2e8446c`. The target is `Qwen/Qwen3-4B` `1cfa9a720891`.

## Contract

**Input: a recorded rollout and its rejection boundaries.**

- `sequence`: the committed token ids (prompt + generated), as returned by `DNeoRuntime.generate`.
- `trace`: the `generate(trace=True)` records, one per verify round. Only these fields are read:
  - `anchor_index` (P): the position of the round's anchor token.
  - `proposed`: the γ draft tokens.
  - `accepted` (count): the number of accepted drafts.
  - `memory_positions`: the runtime's memory positions after the round.

  The rejection boundary is `fork = P + 1 + count`. `proposed[:count]` must equal `sequence[P+1 : fork]`, and
  `memory_positions` must equal `fork .. P+γ`. Otherwise `ValueError` is raised, so a trace that disagrees with
  its sequence is never replayed. `results/h100_20261125/report.rollouts.json` uses this format
  (`[{"sequence": [...], "trace": [...]}]`).

**Output: per-layer `input_layernorm` features at the original positions, one entry per rejecting round.**

`replay.replay_rollout(target, sequence, trace)` returns `list[RejectionMemory]`:

| field | value |
|---|---|
| `round` | Index of the verify round in `trace`. Entries follow trace order, and fully accepted rounds get no entry because the runtime clears memory after them. |
| `positions` | Original absolute positions `fork .. P+γ`, equal to the trace's `memory_positions`. |
| `token_ids` | The rejected drafts `proposed[count:]`. |
| `features` | `[1, n, L, H]` in the target's dtype. See the axes below. |

`features` axes, in order:

| axis | size | meaning |
|---|---|---|
| 0 | 1 | batch |
| 1 | `n = len(positions)` | rejected token; row `i` is at position `positions[i]` (ascending) |
| 2 | `L` = number of target decoder layers (36 for Qwen3-4B) | index `l` = target decoder layer `l`, from 0 to `L-1` in order |
| 3 | `H` = target hidden size (2560 for Qwen3-4B) | output of that layer's `input_layernorm` (normalized attention input, not the raw residual stream) |

`L` is the full layer stack, not the selected layers. The reference runtime stores the full stack in
`DNeoRuntime.memory`, built by `target_attention_input_features` in `reference.py`. The drafter's shared MoFA
router mixes over all `L` layers and masks out the layers outside `config.target_layer_ids`
(`modeling_dneo.py`, `shared_mofa_candidate_mask`). So the checkpoint's selected IDs
`[0, 17, 21, 24, 28, 31, 35]` are indices on axis 2. To keep only those, use `features[:, :, target_layer_ids]`.
The `static-prefix` runtime variant captures only the selected layers. This replay does not cover that variant.

Round correspondence: the memory of round `k` is what `DNeoRuntime.memory` holds after round `k`, and the draft
call of round `k + 1` consumes it. Pass `features` and `positions` to `replay.draft_hidden` to run that draft.

How the replay works: the committed sequence is followed by every rejected branch in one forward. A 4D additive
mask lets each branch token see only `sequence[:fork]` and the earlier tokens of its own branch. Branch tokens
keep their original position ids.

## What was verified where

| commit | content | verified by |
|---|---|---|
| `f060166` | `replay.py`, `check.py`, `run_rental.py`, `summarize.py`, `tests/`, `ruff.toml` | H100 measurement (below). These files are byte-identical to the ones that ran on the box |
| the commit adding this README | adds `replay_rollout`/`RejectionMemory`, `tests/test_contract.py`, `example.py`, `results/`, this README | CPU only: `pytest -q tests` gives 14 passed, and `example.py` and `run_rental.py --smoke` were run from a clean clone. Not re-run on H100 |

`replay_rollout` only wraps `branches_from_trace` and `masked_prefill_features`, which the H100 run measured.
The wrapper itself was not run on a real checkpoint.

## Files

| file | role |
|---|---|
| `replay.py` | `branches_from_trace`, `masked_prefill_inputs` / `masked_prefill_features` (masked prefill), `replay_rollout` (the contract above), `independent_branch_features` (the plain causal forward of `sequence[:fork] + branch`, used as reference), and `draft_hidden` (the reference `DNeoDraftModel.forward`, the same call as `DNeoRuntime.draft_hidden`) |
| `check.py` | `record_rollout` (wraps the runtime's `draft_hidden` to log the prefix length, anchor, live memory and output hidden of every draft call), `evaluate` (per-round measurements) and `verdict` (the pre-registered pass rule) |
| `tests/` | Tiny random Qwen3 target (4 layers) and tiny DNeo drafter (2 layers, shared MoFA router), CPU fp32 |
| `example.py` | Small CPU example of the contract |
| `run_rental.py` | Real-checkpoint run on a GPU. `--smoke` checks the plumbing with the tiny models on CPU |
| `summarize.py` | Turns `report.json` into `rounds.csv`, `feature_layers.csv` and `SUMMARY.md` |
| `results/h100_20261125/` | H100 run outputs: `report.json`, `report.rollouts.json`, `rounds.csv`, `feature_layers.csv`, `pip_freeze.txt` |

## Small example (CPU, tiny random models)

```bash
git clone https://github.com/J-sandler/DNeo && git -C DNeo checkout 8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7
export DNEO_SRC=$PWD/DNeo/src
pip install torch "transformers>=5.9" pytest   # tested: torch 2.6.0 + transformers 5.9.0 (CPU), torch 2.13.0 + transformers 5.12.1 (H100 box)

python -m pytest -q tests          # 14 passed
python example.py                  # writes example_out/rollout.json, replays it
python run_rental.py --smoke --dneo-src "$DNEO_SRC" --out example_out/smoke/report.json --max-new-tokens 48
python summarize.py example_out/smoke/report.json example_out/smoke
```

`example.py` output (macOS, CPU, torch 2.6.0):

```
example_out/rollout.json: 58 tokens, 13 verify rounds, 2 rejecting
round  5: accepted 2/3  positions 33..33  features (1, 1, 4, 32)  vs independent 9.6e-08  vs runtime 1.1e-07
round  7: accepted 0/3  positions 38..40  features (1, 3, 4, 32)  vs independent 8.8e-08  vs runtime 8.4e-08
```

The tests check the following:

- Branch boundaries and the mask layout against literal values.
- Hook capture equals `target_attention_input_features`.
- Masked prefill equals independent forwards, including two branches from the same fork.
- The full prereg rule on a rollout that has rejections at 0 and at >0 accepted, plus a fully accepted round right
  after a rejection.
- The `replay_rollout` contract (rounds, positions, token ids, shape, and equality with the runtime's live memory).
- Six deliberately broken replays are each detected: position off by one, a branch that sees later committed
  tokens, a branch without its prefix, raw instead of normalized features, a drafter that ignores memory, and a
  dropped valid mask.

## H100 result (2026-11-25)

Hardware and software: NVIDIA H100 80GB HBM3 (SM90), torch 2.13.0+cu130, CUDA 13.0, transformers 5.12.1.
All 29 model files were hash-checked against the pinned revisions. The code was commit `f060166` of this branch.

`report.rollouts.json` holds only two things. The inputs are the 4 fixed prompts in `run_rental.py`, wrapped in
the Qwen3 chat template. The outputs are 256 tokens per prompt that the model generated (thinking text on
arithmetic, Fibonacci, unit conversion and TCP/UDP). No other data is in it.

Setup: 4 fixed prompts (`run_rental.py`) with the thinking chat template, `max_new_tokens` 256, seed 42,
temperature 1, γ = 15, and TF32 off. Phase bf16 records `DNeoRuntime(variant="reference")` rollouts with live
memory logging and replays them. Phase fp32 replays the same tokens with the target and drafter upcast. For each
rejecting round, the prefix (taken from the runtime) and the anchor are fixed, and only the memory source
changes: **runtime** = live memory, **indep** = independent causal forward, **masked** = masked prefill.

Pre-registered criteria (fixed before the run):

| # | criterion | result |
|---|---|---|
| 1 | fp32 feature, masked vs indep, rel L2 ≤ 1e-4 | pass, 203/203 rounds, max 1.18e-5 (median 1.69e-6) |
| 2 | fp32 next-draft hidden, masked vs indep, rel L2 ≤ 1e-4 | pass, 203/203, max 3.00e-6 |
| 3a | replacing one valid memory row changes the hidden by > 10 × #2 | pass, min 6.37e-3 |
| 3b | changing 3 padded (masked) rows from 0 to 1e4 leaves the hidden bit-identical | pass, fp32 203/203, bf16 202/202 |
| 3c | after a fully accepted round, memory is cleared and unused | pass, 5/5 rounds: the runtime memory is `None` and the hidden is bit-identical to the no-memory call |
| R | the replay's draft call reproduces the runtime call | pass, 202/202 bit-identical, with prefix length, anchor and positions matching in 202/202 |
| N | bf16 (supplementary): masked vs indep and masked vs runtime ≤ 2 × (runtime vs indep) + 1e-3 | pass, 4/4 prompts |

A fully accepted round right after a rejection occurred 3 times, at (prompt, round) = (0, 1), (1, 7) and (2, 5).
In each case the runtime cleared memory. Forcing the previous memory back in changes the hidden by
0.126 / 0.267 / 0.147, so the check has power.

bf16 runtime, 202 rejecting rounds (rel L2 min / median / max):

| pair | features (all 36 layers) | next-draft hidden |
|---|---|---|
| masked vs indep | 9.45e-3 / 1.12e-2 / 3.79e-2 | 1.09e-2 / 1.38e-2 / 2.74e-2 |
| runtime vs indep | 9.00e-3 / 1.12e-2 / 4.41e-2 | 1.03e-2 / 1.37e-2 / 2.82e-2 |
| masked vs runtime | 9.35e-3 / 1.11e-2 / 3.62e-2 | 1.10e-2 / 1.37e-2 / 2.76e-2 |

In bf16 all three sources differ from each other by about 1e-2. Masked prefill is no farther from the live
memory than the independent forward is. In fp32 the masked vs indep gap drops to about 1e-6, so the bf16 gap
is precision noise (same math, different reduction order). Per selected layer, the fp32 masked vs indep error
peaks at 1.2e-5 at layer 35. Layer 0 is exactly 0 for every pair, and the error grows with depth. The memory
effect (hidden with vs without memory) has a median of 0.36 and a range of 0.069–0.78.

Reproduce on a GPU with at least 40 GB, using local snapshots at the pinned revisions:

```bash
hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir models/qwen3_4b
hf download J-sandler/Qwen3-4B-DNeo --revision 8451a2e8446c0bfa09be47623ff044d896540ec4 --local-dir models/dneo
pip install torch==2.13.0 transformers==5.12.1 safetensors==0.8.0 pytest
python run_rental.py --dneo-src "$DNEO_SRC" --target-path models/qwen3_4b --draft-path models/dneo \
    --out out/result/report.json
python summarize.py out/result/report.json out/summary
```

Without `--target-path` and `--draft-path`, the script downloads the same revisions from the Hub. The H100
instance ran for about 19 minutes, including downloads and one restart of the package install.

## Scope and limits

- This covers one GPU type, 4 prompts × 256 tokens and one seed. Like the reference runtime, generation
  continues past EOS up to 256 tokens.
- The fp32 phase has one more row (203) than bf16 (202). It replays without runtime calls, so it also
  evaluates the last rejecting round.
- The masked prefill builds a dense `[T, T]` mask over committed and branch tokens. Memory grows quadratically
  with rollout length. It was run at ~280 committed tokens plus branches.
- The claim is only that masked prefill reconstructs the rejection memory, and the reference DNeo drafter reads
  it as it reads the live memory. It makes no claim about acceptance rate, training effect, performance, or
  integration with an inference engine.
