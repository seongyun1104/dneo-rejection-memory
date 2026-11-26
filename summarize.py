"""report.json -> rounds.csv, feature_layers.csv, SUMMARY.md (three memory sources compared per round).

usage: python summarize.py <report.json> <out_dir>
"""

import csv
import json
import sys
from pathlib import Path

ROUND_COLUMNS = [
    "phase", "prompt", "round", "kind", "after_rejection", "fork", "accepted", "gamma", "memory_rows",
    "feature_masked_vs_indep", "feature_runtime_vs_indep", "feature_masked_vs_runtime", "feature_masked_vs_indep_max_abs",
    "hidden_masked_vs_indep", "hidden_runtime_vs_indep", "hidden_masked_vs_runtime", "hidden_masked_vs_indep_max_abs",
    "runtime_prefix_matches", "runtime_positions_match", "runtime_hidden_reproduced",
    "memory_effect", "valid_row_perturbation_effect", "masked_rows_ignored", "padded_vs_unpadded",
    "runtime_memory_cleared", "runtime_hidden_equal_no_memory", "stale_memory_effect",
]  # fmt: skip
PAIRS = ("masked_vs_indep", "runtime_vs_indep", "masked_vs_runtime")


def phases(report):
    for entry in report["prompts"]:
        yield "bf16", entry, entry.get("rounds", [])
        if "fp32_rounds" in entry:
            yield "fp32", entry, entry["fp32_rounds"]


def fmt(value):
    return "—" if value is None else f"{value:.2e}" if isinstance(value, float) else str(value)


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    layers = report.get("selected_layers", [])
    with (
        open(out / "rounds.csv", "w", newline="") as rounds_file,
        open(out / "feature_layers.csv", "w", newline="") as f,
    ):
        rounds_csv = csv.DictWriter(rounds_file, ROUND_COLUMNS, extrasaction="ignore")
        rounds_csv.writeheader()
        layer_csv = csv.writer(f)
        layer_csv.writerow(["phase", "prompt", "round", "pair", "layer", "rel_l2"])
        for phase, entry, rounds in phases(report):
            for r in rounds:
                rounds_csv.writerow(dict(r, phase=phase, prompt=entry["index"]))
                for pair, values in r.get("feature_layers", {}).items():
                    for layer, value in values.items():
                        layer_csv.writerow([phase, entry["index"], r["round"], pair, layer, value])

    lines = ["# DNeo rejection memory — 실제 체크포인트 결과 요약", ""]
    lines += [
        f"- {k}: `{report.get(k)}`"
        for k in ("draft_revision", "target_revision", "torch", "cuda", "transformers", "gpu", "max_new_tokens", "seed")
    ]
    lines.append(
        f"- **주 판정(fp32)**: `{report.get('fp32_passed')}` · bf16(런타임·보조): `{report.get('bf16_passed')}`"
    )
    observed = [
        (entry["index"], r["round"])
        for phase, entry, rounds in phases(report)
        if phase == "bf16"
        for r in rounds
        if r["kind"] == "all_accepted" and r.get("after_rejection")
    ]
    lines += [
        "- **거절 직후 전부 수락(실제 rollout, bf16 런타임)**: "
        + (
            f"관측 {len(observed)}회 (prompt, round) = {observed}"
            if observed
            else "**미관측** — 이 동작(memory 비움·이전 memory 미사용)은 **tiny 모델에서만 검증**했다"
        ),
        "",
    ]
    for phase, entry, rounds in phases(report):
        rejected = [r for r in rounds if r["kind"] == "rejected"]
        accepted = [r for r in rounds if r["kind"] == "all_accepted"]
        after = [r for r in accepted if r.get("after_rejection")]
        lines.append(
            f"## prompt {entry['index']} · {phase} — 생성 {entry['generated']} 토큰, 거절 라운드 {len(rejected)}, "
            f"전부 수락 {len(accepted)}(거절 직후 {len(after)}) · 판정 `{entry.get(phase + '_passed')}`"
        )
        failures = entry.get(phase + "_failures") or []
        if failures:
            lines += [f"- 실패: {x}" for x in failures[:20]]
        lines += [
            "",
            "| round | 수락/γ | rows | feat m↔i | feat r↔i | feat m↔r | hid m↔i | hid r↔i | hid m↔r | "
            "런타임 재현 | 유효행 교체 | 마스킹행 무시 |",
            "|" + "---|" * 12,
        ]
        for r in rejected:
            lines.append(
                "| "
                + " | ".join(
                    fmt(x)
                    for x in (
                        r["round"],
                        f"{r['accepted']}/{r['gamma']}",
                        r["memory_rows"],
                        r["feature_masked_vs_indep"],
                        r.get("feature_runtime_vs_indep"),
                        r.get("feature_masked_vs_runtime"),
                        r["hidden_masked_vs_indep"],
                        r.get("hidden_runtime_vs_indep"),
                        r.get("hidden_masked_vs_runtime"),
                        r.get("runtime_hidden_reproduced"),
                        r["valid_row_perturbation_effect"],
                        r["masked_rows_ignored"],
                    )
                )
                + " |"
            )
        if rejected and layers:
            lines += [
                "",
                "선택 레이어별 feature rel L2 (라운드 최댓값):",
                "",
                "| pair | " + " | ".join(f"L{x}" for x in layers) + " |",
                "|" + "---|" * (len(layers) + 1),
            ]
            for pair in PAIRS:
                values = [r["feature_layers"].get(pair) for r in rejected if pair in r.get("feature_layers", {})]
                if values:
                    lines.append(
                        f"| {pair} | " + " | ".join(fmt(max(v[str(x)] for v in values)) for x in layers) + " |"
                    )
        if after:
            lines += ["", "거절 직후 전부 수락 라운드:", ""]
            lines += [
                f"- round {r['round']}: memory 비움 `{r.get('runtime_memory_cleared')}`, 런타임 hidden = memory 없는 호출 "
                f"`{r.get('runtime_hidden_equal_no_memory')}`, 직전 memory 주입 효과 {fmt(r.get('stale_memory_effect'))}"
                for r in after
            ]
        lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((out / "SUMMARY.md").read_text()[:1500])


if __name__ == "__main__":
    main()
