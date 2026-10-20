"""Article companion: F2 for harm screening, accuracy for refusal detection."""

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
from analysis.finetune_benchmark import NAMES, OUT, PANELS


def main():
    frame = pd.read_csv(OUT / "matched_metrics.csv")
    use_braintrust_theme()
    fig, axes = plt.subplots(1, 6, figsize=(23, 6.8), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.72, bottom=0.30, wspace=0.35)
    fig.text(
        0.02,
        0.95,
        "Fine-tuning helps with some safety decisions",
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.88,
        "Matched examples; revised safety instructions for Aegis and WildGuardTest, original instructions for XSTest.",
        fontsize=13,
    )
    for ax, (key, title) in zip(axes, PANELS):
        data = frame[frame.cohort.eq(key)].set_index("arm")
        ax.set_title(
            title + f"\nn = {data.n.iloc[0]:,}", loc="left", fontsize=12, pad=18
        )
        for y, arm in enumerate(NAMES):
            r = data.loc[arm]
            if key == "wildguard_refusal":
                value = r.accuracy
                low, high = wilson_interval(round(value * r.n), int(r.n))
            else:
                value, low, high = r.f2, r.ci_low, r.ci_high
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color="#2C1FEB" if arm in ("safety_tev", "cc_repro") else "#956BDD",
                capsize=4,
                markersize=7,
                elinewidth=1.8,
            )
            ax.text(
                0.02,
                y - 0.22,
                f"{value:.1%}",
                fontsize=11,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
        ax.set_yticks(
            range(len(NAMES)),
            list(NAMES.values()) if ax is axes[0] else [""] * len(NAMES),
        )
        ax.set_ylim(len(NAMES) - 0.5, -0.65)
        ax.set_xlim(0, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Accuracy" if key == "wildguard_refusal" else "F2 score")
        clean_axes(ax, despine_left=True, grid_axis="x")
    fig.text(
        0.02,
        0.12,
        "Lines: marginal 95% intervals (bootstrap for F2; Wilson for accuracy). No pairwise significance claims.",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.065,
        "Training and validation overlap checks found no matches. Custom models use Loops; references use vLLM. Exploratory comparison.",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        savefig(fig, OUT / f"custom_finetune_article.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
