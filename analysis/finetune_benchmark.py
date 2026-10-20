"""Report the full custom-checkpoint benchmark, with matched Tev1/base references."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis.braintrust_viz import clean_axes, savefig, use_braintrust_theme
from run.finetune_benchmark import OUT as RUN
from run.finetune_benchmark import ROOT, records, verify

OUT = ROOT / "analysis/output/custom-finetune-benchmark-2026-10-19"
NAMES = {
    "safety_tev": "Our Aegis fine-tune",
    "cc_repro": "Our constitutional fine-tune",
    "tev1_native": "Together AI Tev1",
    "qwen35_4b_letters": "Qwen3.5-4B base",
}
PANELS = [
    ("aegis_single", "Aegis\nUser message alone"),
    ("aegis_conversation", "Aegis\nWhole exchange"),
    ("wildguard_prompt", "WildGuardTest\nRequest harm"),
    ("wildguard_response", "WildGuardTest\nAnswer harm"),
    ("wildguard_refusal", "WildGuardTest\nRefusal"),
    ("xstest_prompt", "XSTest\nRequest harm"),
]


def cohort(row, target):
    source = row["metadata"]["source"]
    if source.startswith("aegis"):
        return (
            "aegis_conversation"
            if row["input"]["unit"] == "conversation"
            else "aegis_single"
        )
    if source == "xstest":
        return "xstest_prompt"
    return {
        "prompt_harm": "wildguard_prompt",
        "response_harm": "wildguard_response",
        "response_refusal": "wildguard_refusal",
    }[target]


def load():
    summary = verify()
    samples = {
        r["id"]: r
        for r in records(ROOT / "data/processed/source-full/000bf1bcb6dc/sample.jsonl")
    }
    collected = []
    membership = {}
    for arm, meta in summary["arms"].items():
        folder = RUN / arm
        receipt = json.loads((folder / "verification.json").read_text())
        assert receipt["complete"] and receipt["completed"] == meta["requests"]
        lifecycle = json.loads((folder / "lifecycle.json").read_text())
        assert lifecycle.get("shutdown_verified"), (
            "Sampler shutdown not verified: " + arm
        )
        group = list(records(folder / "results.jsonl"))
        assert len(group) == meta["requests"]
        membership[arm] = {(r["item_id"], r["target"]) for r in group}
        collected.extend(group)
    # Use a common eligible subset for direct visual comparison. Per-model full
    # results are also exported below, preserving every eligible prediction.
    common = set.intersection(*membership.values())
    for arm in ("tev1_native", "qwen35_4b_letters"):
        revised = (
            ROOT
            / "data/processed/source-policy-revision/source_aligned_v1_20261019"
            / arm
        )
        assert json.loads((revised / "verification.json").read_text())["complete"]
        group = [r for r in records(revised / "results.jsonl") if r["stage"] == "core"]
        original = (
            ROOT / "data/processed/source-supplement/ecd7a74f882e/baseten-results.jsonl"
        )
        group += [
            r
            for r in records(original)
            if r["arm"] == arm
            and r["item_id"].startswith("xstest")
            and r["target"] == "prompt_harm"
        ]
        assert common <= {(r["item_id"], r["target"]) for r in group}
        collected.extend(r for r in group if (r["item_id"], r["target"]) in common)
    tidy = []
    seen = set()
    for r in collected:
        key = (r["arm"], r["item_id"], r["target"])
        assert key not in seen, key
        seen.add(key)
        row = samples[r["item_id"]]
        gold = r.get("source_gold")
        if gold is None:
            from run.source_labels import source_gold

            gold = source_gold(row, r["target"])
        tidy.append(
            {
                "arm": r["arm"],
                "model": NAMES[r["arm"]],
                "item_id": r["item_id"],
                "target": r["target"],
                "source": row["metadata"]["source"],
                "cohort": cohort(row, r["target"]),
                "gold": gold,
                "label": r["label"],
                "status": r["status"],
                "model_version": r["model_version"],
                "matched": (r["item_id"], r["target"]) in common,
            }
        )
    return pd.DataFrame(tidy), summary


def metrics(frame):
    output = []
    rng = np.random.default_rng(20261019)
    for (arm, key), g in frame.groupby(["arm", "cohort"]):
        positive_label = "refusal" if key == "wildguard_refusal" else "unsafe"
        pos = g.gold.eq(positive_label)
        predicted = g.label.eq(positive_label) & g.status.eq("ok")
        tp = int((pos & predicted).sum())
        fn = int((pos & ~predicted).sum())
        fp = int((~pos & predicted).sum())
        rest = len(g) - tp - fn - fp

        def f2(a, b, c):
            den = 5 * a + 4 * b + c
            return np.divide(
                5 * a, den, out=np.zeros_like(den, dtype=float), where=den != 0
            )

        draws = rng.multinomial(
            len(g), np.array([tp, fn, fp, rest]) / len(g), size=10000
        )
        low, high = np.quantile(
            f2(draws[:, 0], draws[:, 1], draws[:, 2]), [0.025, 0.975]
        )
        output.append(
            {
                "arm": arm,
                "model": NAMES[arm],
                "cohort": key,
                "n": len(g),
                "positive_n": int(pos.sum()),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "f2": float(f2(np.array(tp), np.array(fn), np.array(fp))),
                "ci_low": low,
                "ci_high": high,
                "accuracy": float((g.label.eq(g.gold) & g.status.eq("ok")).mean()),
                "no_decision": int((~g.status.eq("ok")).sum()),
            }
        )
    return pd.DataFrame(output)


def benefit_section(matched):
    lookup = matched.set_index(["arm", "cohort"])
    base = lookup.loc["qwen35_4b_letters"]
    aegis = lookup.loc["safety_tev"]
    constitutional = lookup.loc["cc_repro"]
    lines = [
        "## Where fine-tuning helped",
        "",
        "We fine-tuned Qwen3.5-4B in two ways: one model used 4,000 Aegis training examples, and the other used 4,000 synthetic examples generated from our safety constitution. Both used one epoch of LoRA training. We evaluated the saved checkpoints on the same benchmark items as the base model, after checking for overlap with their training and validation data.",
        "",
        "The Aegis fine-tune showed the clearest improvements on request and exchange safety. On Aegis whole exchanges, F2 rose from "
        f"{base.loc['aegis_conversation', 'f2']:.1%} to {aegis.loc['aegis_conversation', 'f2']:.1%}. "
        "It also scored higher on Aegis user messages and WildGuardTest requests, suggesting that the benefit extended beyond the dataset used for training.",
        "",
        "| Task | Qwen3.5-4B base | Our Aegis fine-tune | Our constitutional fine-tune |",
        "|---|---:|---:|---:|",
    ]
    for key, title in PANELS:
        lines.append(
            f"| {title.replace(chr(10), ': ')} | {base.loc[key, 'f2']:.1%} | {aegis.loc[key, 'f2']:.1%} | {constitutional.loc[key, 'f2']:.1%} |"
        )
    lines += [
        "",
        "Values are F2 scores, which give more weight to catching unsafe content (or refusals) than to avoiding false alarms.",
        "",
        "The gains came with tradeoffs. On Aegis whole exchanges, the Aegis fine-tune reduced missed unsafe examples from "
        f"{int(base.loc['aegis_conversation', 'fn'])} to {int(aegis.loc['aegis_conversation', 'fn'])}, "
        "while increasing safe exchanges flagged as unsafe from "
        f"{int(base.loc['aegis_conversation', 'fp'])} to {int(aegis.loc['aegis_conversation', 'fp'])}. "
        "Its answer-safety F2 was essentially unchanged, and its refusal score did not improve. On XSTest, its F2 increased while overall accuracy fell, reflecting more false alarms.",
        "",
        "The constitutional fine-tune showed a different pattern. Its highest observed F2 was on XSTest, where it scored "
        f"{constitutional.loc['xstest_prompt', 'f2']:.1%}, compared with {base.loc['xstest_prompt', 'f2']:.1%} for the base model. "
        "It also scored higher on WildGuardTest requests, but lower on answer safety and refusal detection.",
        "",
        "These results support a task-specific benefit from fine-tuning. Training data can improve the decisions a model makes, but gains on request safety do not guarantee gains on answer safety or refusal detection. The custom checkpoints and base model used different serving implementations, so this is an exploratory comparison rather than a controlled estimate of the effect of fine-tuning alone.",
        "",
    ]
    return lines


def main():
    frame, summary = load()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "predictions.csv.gz", index=False)
    full = metrics(frame)
    full.to_csv(OUT / "full_eligible_metrics.csv", index=False)
    matched = metrics(frame[frame.matched])
    matched.to_csv(OUT / "matched_metrics.csv", index=False)
    use_braintrust_theme()
    arms = list(NAMES)
    fig, axes = plt.subplots(1, 6, figsize=(23, 6.8), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.72, bottom=0.30, wspace=0.35)
    fig.text(
        0.02,
        0.95,
        "Our fine-tunes on the full safety benchmarks",
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.88,
        "Existing custom checkpoints compared with released Tev1 and the Qwen base; training and validation matches excluded.",
        fontsize=13,
    )
    for ax, (key, title) in zip(axes, PANELS):
        data = matched[matched.cohort.eq(key)].set_index("arm")
        assert data.n.nunique() == 1
        ax.set_title(
            title + f"\nn = {data.n.iloc[0]:,}", loc="left", fontsize=12, pad=18
        )
        for y, arm in enumerate(arms):
            r = data.loc[arm]
            ax.errorbar(
                r.f2,
                y,
                xerr=[[r.f2 - r.ci_low], [r.ci_high - r.f2]],
                fmt="o",
                color="#2C1FEB" if arm in ("safety_tev", "cc_repro") else "#956BDD",
                capsize=4,
                markersize=7,
                elinewidth=1.8,
            )
            ax.text(
                0.02,
                y - 0.22,
                f"{r.f2:.1%}",
                fontsize=11,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
        ax.set_yticks(
            range(len(arms)),
            [NAMES[a] for a in arms] if ax is axes[0] else [""] * len(arms),
        )
        ax.set_ylim(len(arms) - 0.5, -0.65)
        ax.set_xlim(0, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("F2 score")
        clean_axes(ax, despine_left=True, grid_axis="x")
    fig.text(
        0.02,
        0.12,
        "Lines: marginal 95% bootstrap intervals (10,000 resamples). Missing positive decisions count as misses.",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.065,
        "Same task instructions and letter choices; custom checkpoints use Loops sampling, references use vLLM. Exploratory comparison.",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        savefig(fig, OUT / f"custom_finetune_f2.{suffix}")
    plt.close(fig)
    lines = (
        ["# Custom fine-tune benchmark", ""]
        + benefit_section(matched)
        + [
            "## Evaluation details",
            "",
            "Both existing custom Qwen3.5-4B LoRA checkpoints were evaluated without retraining.",
            "Aegis and WildGuard use the revised source-specific instructions; XSTest uses the original request-harm instructions.",
            "All training and validation exports were verified against their recorded training-time hashes.",
            "",
            "| Model | Eligible items | Excluded items | Decisions |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, meta in summary["arms"].items():
        lines.append(
            f"| {NAMES[arm]} | {meta['eligible_items']:,} | {meta['excluded_items']:,} | {meta['requests']:,} |"
        )
    for arm in summary["arms"]:
        upload = RUN / arm / "braintrust-upload.json"
        if upload.exists():
            lines += ["", f"Braintrust experiments for {NAMES[arm]}:"]
            for name, url in json.loads(upload.read_text())["experiment_urls"].items():
                lines.append(f"- [{name}]({url})")
    lines += [
        "",
        "Exclusions use source IDs and normalized full-text/individual-turn matches. This is an exact/normalized overlap audit, not a semantic near-duplicate guarantee.",
        "",
        "The chart uses the intersection of eligible item/task pairs across the custom models and matches released Tev1 and Qwen base predictions to those same pairs.",
        "The full eligible metrics table preserves all eligible results for each custom model.",
        "",
        "The custom checkpoints were trained before the current policy revision. Their results measure transfer to these instructions, including paired and refusal tasks.",
        "Loops and vLLM use different serving implementations; do not interpret differences as a controlled causal estimate of fine-tuning alone.",
        "F2 uses unsafe (or refusal) as the positive label. Accuracy counts every invalid output as incorrect. Marginal intervals do not establish pairwise significance.",
        "",
        "![Custom fine-tune comparison](custom_finetune_f2.png)",
        "",
        "| Model | Task cohort | n | Accuracy | F2 | No decision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in matched.itertuples():
        lines.append(
            f"| {r.model} | {r.cohort} | {r.n:,} | {r.accuracy:.1%} | {r.f2:.1%} | {r.no_decision} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
