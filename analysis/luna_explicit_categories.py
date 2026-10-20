"""Compare the archived run, a fresh repeat, and explicit category instructions."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.braintrust_viz import (
    header,
    make_figure,
    sequential_cmap,
    use_braintrust_theme,
    wilson_interval,
)
from analysis.source_report import heatmap_ink, save
from run.luna_explicit_categories import OUT as RUN
from run.luna_explicit_categories import ROOT, records
from run.offline_review import digest

OUT = ROOT / "analysis/output/luna-explicit-categories-2026-10-14"
VARIANTS = ["archived_original", "original_policy_repeat", "explicit_categories"]
NAMES = ["Archived original", "Fresh original", "Explicit rules"]
SHORT = {
    "causing_material_harm_by_disseminating_misinformation": "Harmful misinformation",
    "copyright_violations": "Copyright",
    "cyberattack": "Cyberattacks",
    "defamation_encouraging_unethical_or_unsafe_actions": "Defamation / unsafe actions",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "Disinformation campaigns",
    "fraud_assisting_illegal_activities": "Fraud / illegal activities",
    "mental_health_over-reliance_crisis": "Mental health crisis / dependence",
    "others": "Other harms",
    "private_information_individual": "Private personal information",
    "sensitive_information_organization_government": "Sensitive organizational information",
    "sexual_content": "Sexual content",
    "social_stereotypes_and_unfair_discrimination": "Stereotypes / discrimination",
    "toxic_language_hate_speech": "Toxic language / hate speech",
    "violence_and_physical_harm": "Violence / physical harm",
}


def main():
    jobs = {r["job_id"]: r for r in records(RUN / "manifest.jsonl")}
    results = list(records(RUN / "results.jsonl"))
    assert len(results) == len(jobs) == 3362
    assert len({r["job_id"] for r in results}) == len(results)
    cases = []
    for r in results:
        job = jobs[r["job_id"]]
        assert r["raw"]["request"] == job["request"]
        assert (
            digest(r["raw"]["request"]) == r["request_sha256"] == job["request_sha256"]
        )
        cases.append(
            {
                k: r[k]
                for k in (
                    "item_id",
                    "variant",
                    "category",
                    "source_gold",
                    "original_label",
                    "label",
                    "status",
                )
            }
        )
    df = pd.DataFrame(cases)
    assert not df.duplicated(["item_id", "variant"]).any()
    baseline = df[df.variant == "original_policy_repeat"].copy()
    baseline["variant"], baseline["label"], baseline["status"] = (
        "archived_original",
        baseline.original_label,
        "ok",
    )
    df = pd.concat([df, baseline], ignore_index=True)
    df["error"] = df.status.ne("ok") | df.label.ne(df.source_gold)
    metrics = []
    for (variant, gold), g in df.groupby(["variant", "source_gold"]):
        k, n = int(g.error.sum()), len(g)
        lo, hi = wilson_interval(k, n)
        metrics.append(
            {
                "variant": variant,
                "reference_label": gold,
                "errors": k,
                "n": n,
                "error_rate": k / n,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    categories = []
    for (variant, category), g in df[df.source_gold == "unsafe"].groupby(
        ["variant", "category"]
    ):
        k, n = int(g.error.sum()), len(g)
        lo, hi = wilson_interval(k, n)
        categories.append(
            {
                "variant": variant,
                "category": category,
                "missed": k,
                "n": n,
                "miss_rate": k / n,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    categories = pd.DataFrame(categories)
    wide = df.pivot(index="item_id", columns="variant", values="label")
    items = baseline.set_index("item_id")[["source_gold", "category"]]
    wide = items.join(wide, validate="one_to_one")
    wide["corrected_by_explicit"] = wide.original_policy_repeat.ne(
        wide.source_gold
    ) & wide.explicit_categories.eq(wide.source_gold)
    wide["introduced_by_explicit"] = wide.original_policy_repeat.eq(
        wide.source_gold
    ) & wide.explicit_categories.ne(wide.source_gold)
    transitions = []
    for gold, g in wide.groupby("source_gold"):
        transitions.append(
            {
                "reference_label": gold,
                "n": len(g),
                "corrected": int(g.corrected_by_explicit.sum()),
                "introduced": int(g.introduced_by_explicit.sum()),
                "unchanged": int(
                    g.original_policy_repeat.eq(g.explicit_categories).sum()
                ),
            }
        )
    tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
    summary = {
        "examples": len(wide),
        "new_calls": len(results),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "model_versions": dict(Counter(r["model_version"] for r in results)),
        "original_repeat_changed_labels": int(
            wide.archived_original.ne(wide.original_policy_repeat).sum()
        ),
        "input_tokens": tokens,
        "estimated_input_cost_usd": tokens * 0.10 / 1e6,
        "pricing_source": "https://developers.openai.com/api/docs/guides/decisions",
        "paired_transitions": transitions,
        "metrics": metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "decisions.csv", index=False)
    wide.to_csv(OUT / "paired_cases.csv")
    categories.to_csv(OUT / "category_metrics.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "overall_metrics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    use_braintrust_theme()
    order = (
        categories[categories.variant == "archived_original"]
        .sort_values("miss_rate", ascending=False)
        .category.tolist()
    )
    fig, ax = make_figure(width=13, height=13, left=0.36, right=0.98, top=0.83)
    matrix = np.array(
        [
            [
                categories[(categories.category == c) & (categories.variant == v)]
                .iloc[0]
                .miss_rate
                for v in VARIANTS
            ]
            for c in order
        ]
    )
    cmap = sequential_cmap()
    ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i, c in enumerate(order):
        for j, variant in enumerate(VARIANTS):
            r = categories[
                (categories.category == c) & (categories.variant == variant)
            ].iloc[0]
            ax.text(
                j,
                i,
                f"{r.miss_rate:.1%} · {int(r.missed)}/{int(r.n)}\n95% CI {r.ci_low:.0%}–{r.ci_high:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color=heatmap_ink(r.miss_rate, cmap, 0, 1),
            )
    ax.set_xticks(range(3), NAMES, fontsize=12)
    ax.set_yticks(range(len(order)), [SHORT[c] for c in order], fontsize=11)
    ax.tick_params(length=0)
    ax.grid(False)
    header(
        ax,
        "Explicit rules reduce Luna’s unsafe misses",
        "Decisions Luna · identical WildGuard inputs and choice order",
        "Miss rate against human prompt labels. Each cell: missed / unsafe n; 95% Wilson interval.",
        title_size=20,
        subtitle_size=13,
        key_size=10,
    )
    save(fig, OUT, "category_comparison")
    plt.close(fig)
    lines = [
        "# Decisions Luna: explicit category rules",
        "",
        "Explicit rules reduced disagreement with WildGuard’s unsafe-prompt labels, while increasing false alarms on safe prompts. This supports policy wording as a substantial contributor to the earlier miss rates; it does not explain every remaining disagreement.",
        "",
        "All 1,681 original WildGuardTest pairs were sent unchanged. The target is user-request safety; assistant safety and refusal remain separate.",
        "",
        "| Reference group | Archived original | Fresh original | Explicit rules |",
        "| --- | ---: | ---: | ---: |",
    ]
    m = pd.DataFrame(metrics).set_index(["variant", "reference_label"])
    for gold, label in [
        ("unsafe", "Unsafe prompts missed"),
        ("safe", "Safe prompts flagged"),
    ]:
        cells = []
        for variant in VARIANTS:
            r = m.loc[(variant, gold)]
            cells.append(f"{int(r.errors)}/{int(r.n)} ({r.error_rate:.1%})")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"The fresh original policy changed {summary['original_repeat_changed_labels']} labels relative to the archived run.",
        "",
        "| Reference group | Errors corrected by explicit rules | Errors introduced |",
        "| --- | ---: | ---: |",
    ]
    for t in transitions:
        lines.append(
            f"| {t['reference_label']} | {t['corrected']} | {t['introduced']} |"
        )
    lines += [
        "",
        "## Categories that prompted this check",
        "",
        "| Category | Archived original misses | Fresh original misses | Explicit-rule misses |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in [
        "copyright_violations",
        "social_stereotypes_and_unfair_discrimination",
    ]:
        values = []
        for variant in VARIANTS:
            r = categories[
                (categories.category == category) & (categories.variant == variant)
            ].iloc[0]
            values.append(f"{int(r.missed)}/{int(r.n)} ({r.miss_rate:.1%})")
        lines.append("| " + SHORT[category] + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Category comparison](category_comparison.png)",
        "",
        "The same complete category rules were appended to every explicit-policy request. Source category, reference label, and original prediction were never sent to the model. Model, input bytes, safe/unsafe descriptions, question name, and per-item choice order were preserved. Separate requests were interleaved in deterministic shuffled item order with four workers.",
        "",
        "These are investigator-written rules, not an official reproduction of the WildGuard annotation policy. They both clarify and expand the original policy. Results are a diagnostic on previously inspected benchmark examples, not held-out performance or proof of an API defect. The fresh original-policy repeat provides a contemporary comparison; it is one repeat per example.",
        "",
        f"Completed {len(results):,} new calls. Statuses: {json.dumps(summary['status_counts'])}. Recorded input tokens: {tokens:,}. Estimated input charge: ${summary['estimated_input_cost_usd']:.3f}, using [OpenAI’s published Decisions rate](https://developers.openai.com/api/docs/guides/decisions); this is not an invoice.",
        "",
        "[Exact added rules](../../../data/processed/luna-explicit-categories/2026-10-14/explicit_rules.md) · [Raw results](../../../data/processed/luna-explicit-categories/2026-10-14/results.jsonl) · [Paired cases](paired_cases.csv) · [Category counts](category_metrics.csv)",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
