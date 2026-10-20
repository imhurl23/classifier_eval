"""Draft visualization of the existing consistency export; no inference calls."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis.braintrust_viz import (
    clean_axes,
    savefig,
    use_braintrust_theme,
    wilson_interval,
)
from analysis.source_report import NAMES

OUT = Path(__file__).resolve().parent / "output/failures-2026-10-12/superseded-or-view"
MODES = ["joint safe despite unsafe turn", "joint unsafe despite safe turns"]
COLORS = ["#956BDD", "#2C1FEB"]


def main():
    use_braintrust_theme()
    data = pd.read_csv(OUT / "cross_turn_consistency_cases.csv")
    # Denominators retain the original report's model-specific valid triples.
    data = data[data.valid_triple]
    rows = []
    for arm, group in data.groupby("arm"):
        n = len(group)
        counts = [int(group.consistency_mode.eq(mode).sum()) for mode in MODES]
        rows.append({"arm": arm, "n": n, "counts": counts, "total": sum(counts) / n})
    rows.sort(key=lambda row: row["total"], reverse=True)
    fig = plt.figure(figsize=(17, 11), facecolor="white")
    fig.text(
        0.025,
        0.97,
        "Do the three safety answers agree?",
        fontsize=27,
        weight="bold",
        va="top",
    )
    fig.text(
        0.025,
        0.90,
        "Study rule: the whole exchange is unsafe if either message is unsafe.",
        fontsize=16,
    )
    fig.text(
        0.025,
        0.86,
        "Three separate questions about the same conversation; this checks agreement between answers.",
        fontsize=12,
        color="#6B6B76",
    )
    cards = [
        (
            "1. A message is unsafe, but the exchange is safe",
            "Example judgments: user UNSAFE · assistant SAFE · exchange SAFE",
        ),
        (
            "2. Both messages are safe, but the exchange is unsafe",
            "Example judgments: user SAFE · assistant SAFE · exchange UNSAFE",
        ),
    ]
    for x, (title, example), color in zip([0.025, 0.515], cards, COLORS):
        fig.text(x, 0.807, title, fontsize=14, weight="bold", color=color)
        fig.text(x, 0.777, example, fontsize=11)
    fig.text(
        0.025,
        0.738,
        "Illustrations above are answer combinations, not source-label judgments.",
        fontsize=10,
        color="#6B6B76",
    )
    axes = [
        fig.add_axes([0.22, 0.145, 0.265, 0.535]),
        fig.add_axes([0.565, 0.145, 0.265, 0.535]),
    ]
    for j, ax in enumerate(axes):
        for i, row in enumerate(rows):
            k, n = row["counts"][j], row["n"]
            value = k / n
            low, high = wilson_interval(k, n)
            ax.barh(i, value, height=0.57, color=COLORS[j])
            ax.errorbar(
                value,
                i,
                xerr=[[value - low], [high - value]],
                fmt="none",
                color="#18181B",
                capsize=3,
                linewidth=1.3,
                zorder=4,
            )
            ax.text(
                1.025,
                i,
                f"{value:.1%}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=12,
                weight="bold",
            )
        ax.set_ylim(len(rows) - 0.45, -0.6)
        ax.set_xlim(0, 0.125)
        ax.set_xticks([0, 0.04, 0.08, 0.12])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([NAMES[row["arm"]] for row in rows] if j == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.set_xlabel("Share of valid conversations", fontsize=12, labelpad=12)
        clean_axes(ax, despine_left=True, grid_axis="x")
        ax.set_title(
            [
                "Exchange safe despite an unsafe message",
                "Exchange unsafe despite two safe messages",
            ][j],
            fontsize=12,
            loc="left",
            pad=16,
            color=COLORS[j],
            weight="bold",
        )
    totals = fig.add_axes([0.905, 0.145, 0.09, 0.535])
    totals.set_xlim(0, 1)
    totals.set_ylim(len(rows) - 0.45, -0.6)
    totals.axis("off")
    totals.set_title("Either pattern", fontsize=12, loc="left", pad=16, weight="bold")
    for i, row in enumerate(rows):
        totals.text(
            0, i - 0.12, f"{row['total']:.1%}", fontsize=12, weight="bold", va="center"
        )
        totals.text(
            0,
            i + 0.17,
            f"{sum(row['counts']):,} / {row['n']:,}",
            fontsize=9,
            va="center",
            color="#6B6B76",
        )
    fig.text(
        0.025,
        0.073,
        "WildGuardTest · common-policy interfaces · valid safe/unsafe answers to all three questions only.",
        fontsize=11,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.048,
        "Whiskers: 95% Wilson intervals. Each rate uses the same valid-conversation denominator within a model.",
        fontsize=11,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.023,
        "Agreement does not imply accuracy. Interfaces that compute the exchange label from message labels are excluded.",
        fontsize=11,
        color="#6B6B76",
    )
    savefig(fig, OUT / "06_consistency_explained_draft.png")
    savefig(fig, OUT / "06_consistency_explained_draft.svg")
    plt.close(fig)
    pd.DataFrame(
        [
            {
                "arm": r["arm"],
                "valid_conversations": r["n"],
                "exchange_safe_despite_unsafe_message": r["counts"][0],
                "exchange_unsafe_despite_safe_messages": r["counts"][1],
                "either_pattern_rate": r["total"],
            }
            for r in rows
        ]
    ).to_csv(OUT / "06_consistency_explained_draft.csv", index=False)


if __name__ == "__main__":
    main()
