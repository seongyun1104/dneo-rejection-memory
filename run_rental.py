"""Real-checkpoint run of the masked-prefill rejection-memory replay (GPU rental only; README.md PREREG).

Phase bf16: reference DNeoRuntime rollouts with live memory logging, replay, runtime + noise-relative checks.
Phase fp32: the same recorded tokens replayed with target and drafter upcast, fixed tolerances.
Writes <out> (report, rewritten after every prompt and phase), <out>.rollouts.json (tokens + traces) and
<out minus suffix>.DONE when both phases finished.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

DRAFT = "J-sandler/Qwen3-4B-DNeo"
DRAFT_REVISION = "8451a2e8446c0bfa09be47623ff044d896540ec4"
TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPTS = [
    "Explain why the sum of two odd numbers is always even.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train travels 120 km in 1.5 hours. What is its average speed in m/s?",
    "List three differences between TCP and UDP.",
]
FP32_TOL = 1e-4
BF16_FLOOR = 1e-3


def bf16_tol(r, key):
    reference = {"feature": r.get("feature_runtime_vs_indep"), "hidden": r.get("hidden_runtime_vs_indep")}
    return 2 * reference[key.split("_")[0]] + BF16_FLOOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dneo-src", required=True)
    parser.add_argument("--target-path", help="local snapshot of Qwen/Qwen3-4B at TARGET_REVISION")
    parser.add_argument("--draft-path", help="local snapshot of the drafter at DRAFT_REVISION")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="tiny random models on CPU; plumbing check only")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here / "tests"), args.dneo_src]
    if args.smoke:
        from conftest import tiny_pair

        torch.cuda.synchronize = lambda *a, **k: None
        target, model = tiny_pair()
        prompts = [torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(i)) for i in range(2)]
        report = dict(smoke=True, torch=torch.__version__, max_new_tokens=args.max_new_tokens, seed=args.seed)
        return run(target, model, prompts, report, args)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dneo.loading import load_draft

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    model = load_draft(args.draft_path or DRAFT, device, revision=None if args.draft_path else DRAFT_REVISION)
    assert model.config.target_revision == TARGET_REVISION, model.config.target_revision
    target_source = args.target_path or model.config.target_model_name_or_path
    target = (
        AutoModelForCausalLM.from_pretrained(
            target_source,
            revision=None if args.target_path else TARGET_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    model.bind_target(target)
    tokenizer = AutoTokenizer.from_pretrained(target_source, revision=None if args.target_path else TARGET_REVISION)
    report = dict(
        draft=DRAFT,
        draft_revision=DRAFT_REVISION,
        target_revision=TARGET_REVISION,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=__import__("transformers").__version__,
        gpu=torch.cuda.get_device_name(),
        prompt_texts=PROMPTS,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            enable_thinking=True,
        ).to(device)
        for prompt in PROMPTS
    ]
    return run(target, model, prompts, report, args)


def run(target, model, prompts, report, args):
    from check import evaluate, record_rollout, verdict
    from dneo.runtime.dneo import DNeoRuntime

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["selected_layers"] = list(model.config.target_layer_ids)
    report["prompts"], recorded = [], []

    def save():
        out.write_text(json.dumps(report, indent=1))
        out.with_suffix(".rollouts.json").write_text(json.dumps([dict(sequence=s, trace=t) for s, t in recorded]))

    runtime = DNeoRuntime(target, model, variant="reference", draft_kernel="reference", context_capacity=1024)
    for index, ids in enumerate(prompts):
        start = time.time()
        result, calls, prefix = record_rollout(runtime, ids, max_new_tokens=args.max_new_tokens, seed=args.seed)
        sequence = ids[0].tolist() + result["token_ids"]
        rounds = evaluate(target, model, sequence, result["trace"], calls, prefix)
        passed, failures = verdict(rounds, bf16_tol, bf16_tol, bf16_tol, effect_ratio=0.0)
        report["prompts"].append(
            dict(
                index=index,
                prompt_tokens=ids.shape[1],
                generated=len(result["token_ids"]),
                rounds=rounds,
                bf16_passed=passed,
                bf16_failures=failures,
                bf16_seconds=time.time() - start,
            )
        )
        recorded.append((sequence, result["trace"]))
        save()
        del calls, prefix
        torch.cuda.empty_cache()
    runtime.close()

    target.float()
    model.float()
    model.bind_target(target)
    for entry, (sequence, trace) in zip(report["prompts"], recorded):
        start = time.time()
        rounds = evaluate(target, model, sequence, trace)
        passed, failures = verdict(rounds, FP32_TOL, FP32_TOL)
        entry.update(fp32_rounds=rounds, fp32_passed=passed, fp32_failures=failures, fp32_seconds=time.time() - start)
        save()
        torch.cuda.empty_cache()
    report["fp32_passed"] = all(e["fp32_passed"] for e in report["prompts"])
    report["bf16_passed"] = all(e["bf16_passed"] for e in report["prompts"])
    save()
    print(json.dumps({k: report[k] for k in ("fp32_passed", "bf16_passed")}))
    out.with_suffix(".DONE").write_text("done\n")


if __name__ == "__main__":
    main()
