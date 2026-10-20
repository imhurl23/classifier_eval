"""Plot F2 by question from saved common-policy counts; no model calls.

Run: .venv/bin/python -m analysis.plot_safety_f2
Positive abstentions are FN; negative abstentions never become positive flags.
Intervals resample examples via the multinomial distribution of TP/FN/FP/rest.
These are marginal intervals, not paired significance tests between models.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis.braintrust_viz import clean_axes, rate_cmap, savefig, use_braintrust_theme
from analysis.source_report import NAMES

OUT = Path(__file__).resolve().parent / "output/safety-usecases-2026-10-14"
SPECS = [
    (
        "01_request_screening",
        "Is the request unsafe?",
        [
            ("aegis_single", "Aegis: request alone"),
            ("xstest_prompt", "XSTest: request alone"),
        ],
    ),
    (
        "02_response_safety",
        "Is the assistant answer unsafe?",
        [("wildguard_response", "WildGuardTest: answer safety")],
    ),
    (
        "03_conversation_safety",
        "Is the whole exchange unsafe?",
        [
            ("aegis_conversation", "Aegis: whole exchange"),
            ("wildguard_conversation", "WildGuardTest: either message unsafe"),
        ],
    ),
    (
        "04_refusal_detection",
        "Did the assistant refuse?",
        [("wildguard_refusal", "WildGuardTest: refusal")],
    ),
    (
        "05_request_with_answer",
        "Is the request unsafe when the answer is visible?",
        [("wildguard_prompt", "WildGuardTest: request with answer visible")],
    ),
]


def score(tp, fn, fp):
    denominator = 5 * tp + 4 * fn + fp
    return np.divide(
        5 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute(metrics):
    identity = [
        "arm",
        "model",
        "provider",
        "policy_mode",
        "record_kind",
        "scheme",
        "rubric",
        "model_version",
        "usecase",
        "cohort",
    ]
    rng = np.random.default_rng(20261015)
    rows = []
    for _, group in metrics.groupby(identity, dropna=False, sort=True):
        assert not group.metric.duplicated().any()
        record = group.iloc[0].to_dict()
        measures = group.set_index("metric")
        tp = int(record["positive_n"] - measures.loc["positive_not_flagged", "k"])
        fn = int(measures.loc["positive_not_flagged", "k"])
        fp = int(measures.loc["negative_false_alarm", "k"])
        rest = int(record["negative_n"] - fp)
        counts = np.array([tp, fn, fp, rest])
        assert (counts >= 0).all() and counts.sum() == record["total_n"]
        draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=10000)
        low, high = np.quantile(
            score(draws[:, 0], draws[:, 1], draws[:, 2]), [0.025, 0.975]
        )
        rows.append(
            {
                k: record[k]
                for k in identity
                + [
                    "source_positive",
                    "total_n",
                    "positive_n",
                    "negative_n",
                    "no_decision_n",
                ]
            }
            | dict(
                tp=tp,
                fn=fn,
                fp=fp,
                f2=float(score(tp, fn, fp)),
                ci_low=low,
                ci_high=high,
                bootstrap_replicates=10000,
                bootstrap_seed=20261015,
            )
        )
    return pd.DataFrame(rows)


def draw(frame, stem, title, cohorts):
    selected = frame[frame.cohort.isin([key for key, _ in cohorts])]
    reference = selected[selected.cohort.eq(cohorts[0][0])].sort_values(
        "f2", ascending=False
    )
    arms = reference.arm.tolist()
    # Cohorts share model order; missing configurations are explicitly marked.
    fig, axes = plt.subplots(
        1, len(cohorts), figsize=(13 if len(cohorts) > 1 else 10, 8), squeeze=False
    )
    fig.subplots_adjust(left=0.21, right=0.88, top=0.76, bottom=0.27, wspace=0.22)
    fig.text(0.04, 0.94, title, fontsize=22, fontweight="bold")
    fig.text(
        0.04,
        0.895,
        "F2 score by model - higher is better",
        fontsize=15,
        color="#6B6B76",
    )
    positive = "refusal" if stem.startswith("04") else "unsafe"
    fig.text(
        0.04,
        0.855,
        f"Positive class: {positive}. F2 = 5TP / (5TP + 4FN + FP).",
        fontsize=12,
        color="#6B6B76",
    )
    for ax, (key, label) in zip(axes[0], cohorts):
        data = selected[selected.cohort.eq(key)].set_index("arm")
        for y, arm in enumerate(arms):
            if arm not in data.index:
                ax.text(0.5, y, "Not tested", ha="center", fontsize=11, color="#6B6B76")
                continue
            r = data.loc[arm]
            ax.errorbar(
                r.f2,
                y,
                xerr=[[r.f2 - r.ci_low], [r.ci_high - r.f2]],
                fmt="o",
                color="#2C1FEB",
                ecolor="#956BDD",
                capsize=4,
                markersize=7,
                elinewidth=1.8,
            )
            ax.text(1.04, y, f"{r.f2:.1%}", va="center", fontsize=11, fontweight="bold")
        n = data.total_n.unique()
        assert len(n) == 1
        ax.set_title(f"{label}\nn = {n[0]:,}", loc="left", fontsize=12, pad=16)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_yticks(
            range(len(arms)),
            [NAMES[a] for a in arms] if ax is axes[0, 0] else [""] * len(arms),
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(axis="y", length=0)
        clean_axes(ax, despine_left=True, grid_axis="x")
        ax.set_xlabel("F2 score", fontsize=12)
    fig.text(
        0.04,
        0.135,
        "Lines: 95% bootstrap confidence intervals (10,000 resamples of examples).",
        fontsize=11,
        color="#6B6B76",
    )
    fig.text(
        0.04,
        0.10,
        "Missing decisions on positive cases count as misses. Negative abstentions do not count as false alarms.",
        fontsize=10,
        color="#6B6B76",
    )
    fig.text(
        0.04,
        0.065,
        "Same safety rules as the main charts. Compare models within each test; test labels and case mixes differ.",
        fontsize=10,
        color="#6B6B76",
    )
    for suffix in ["png", "svg"]:
        savefig(fig, OUT / f"{stem}_f2.{suffix}")
    plt.close(fig)


def overview(frame):
    keys = [k for _, _, cohorts in SPECS for k, _ in cohorts]
    headers = [
        "Request alone\nAegis",
        "Request alone\nXSTest",
        "Answer safety\nWildGuardTest",
        "Whole exchange\nAegis",
        "Either unsafe\nWildGuardTest",
        "Refusal\nWildGuardTest",
        "Request + answer\nWildGuardTest",
    ]
    arms = list(NAMES[a] for a in sorted(frame.arm.unique(), key=lambda a: NAMES[a]))
    ids = sorted(frame.arm.unique(), key=lambda a: NAMES[a])
    lookup = frame.set_index(["arm", "cohort"])
    values = np.array(
        [
            [
                lookup.loc[(a, k), "f2"] if (a, k) in lookup.index else np.nan
                for k in keys
            ]
            for a in ids
        ]
    )
    fig, ax = plt.subplots(figsize=(17, 10))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.77, bottom=0.16)
    ax.imshow(
        np.ma.masked_invalid(values), cmap=rate_cmap(), vmin=0, vmax=1, aspect="auto"
    )
    for i, a in enumerate(ids):
        for j, k in enumerate(keys):
            if (a, k) not in lookup.index:
                ax.text(
                    j,
                    i,
                    "Not tested",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#6B6B76",
                )
                continue
            r = lookup.loc[(a, k)]
            color = "white" if r.f2 >= 0.5 else "#18181B"
            ax.text(
                j,
                i - 0.10,
                f"{r.f2:.1%}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )
            ax.text(
                j,
                i + 0.23,
                f"{r.ci_low:.1%} - {r.ci_high:.1%}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    ax.set_xticks(range(len(keys)), headers, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ids)), arms, fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.04, 0.95, "F2 for each safety question", fontsize=25, fontweight="bold")
    fig.text(
        0.04,
        0.905,
        "Higher is better. Positive = unsafe for safety questions; refusal for refusal detection.",
        fontsize=14,
        color="#6B6B76",
    )
    fig.text(
        0.04,
        0.865,
        "Each cell: F2 score and 95% bootstrap interval. Compare models within a column.",
        fontsize=13,
        color="#6B6B76",
    )
    fig.text(
        0.04,
        0.10,
        "F2 = 5TP / (5TP + 4FN + FP). Missing decisions on positive cases count as misses.",
        fontsize=12,
        color="#6B6B76",
    )
    fig.text(
        0.04,
        0.06,
        "Negative abstentions do not create false alarms; inspect failed-decision counts separately. Same rules as the main charts.",
        fontsize=11,
        color="#6B6B76",
    )
    for suffix in ["png", "svg"]:
        savefig(fig, OUT / f"06_f2_overview.{suffix}")
    plt.close(fig)


def main():
    use_braintrust_theme()
    metrics = pd.read_csv(OUT / "usecase_metrics.csv")
    frame = compute(metrics)
    frame.to_csv(OUT / "f2_metrics.csv", index=False)
    for stem, title, cohorts in SPECS:
        draw(frame, stem, title, cohorts)
    overview(frame)
    print(f"Wrote F2 scores for {len(frame)} model/question/test configurations.")


if __name__ == "__main__":
    main()
