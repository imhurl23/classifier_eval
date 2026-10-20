"""Article views excluding the derived WildGuard either-message-unsafe task.

Reads archived metrics without rerunning models. Uses accuracy for refusal and
F2 for harm screening. Recalculates Holm across the original 345 comparison
slots, replacing refusal F2 tests with paired accuracy tests.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from itertools import combinations

import matplotlib.pyplot as plt
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests

from analysis.braintrust_viz import clean_axes, savefig, use_braintrust_theme
from analysis.plot_safety_usecases import COHORTS, select
from analysis.plot_safety_usecases import SOURCE as JUDGMENTS
from analysis.source_report import NAMES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/output/safety-usecases-2026-10-14"
OUT = ROOT / "analysis/output/article-human-labels"
EXCLUDED = "wildguard_conversation"


def prepare():
    scores = pd.read_csv(SOURCE / "f2_metrics.csv").rename(columns={"f2": "score"})
    scores["metric"] = "F2"
    accuracy = pd.read_csv(SOURCE / "usecase_metrics.csv")
    accuracy = accuracy[
        accuracy.cohort.eq("wildguard_refusal") & accuracy.metric.eq("accuracy")
    ].set_index("arm")
    for i, row in scores[scores.cohort.eq("wildguard_refusal")].iterrows():
        a = accuracy.loc[row.arm]
        assert row.total_n == a.n
        scores.loc[i, ["score", "ci_low", "ci_high", "metric"]] = [
            a.rate,
            a.ci_low,
            a.ci_high,
            "Accuracy",
        ]
        scores.loc[i, ["bootstrap_replicates", "bootstrap_seed"]] = np.nan
    scores["interval_method"] = np.where(
        scores.metric.eq("Accuracy"), "Wilson 95%", "10,000-resample bootstrap 95%"
    )

    pairwise = pd.read_csv(SOURCE / "f2_pairwise_significance.csv").rename(
        columns={"f2_a": "score_a", "f2_b": "score_b"}
    )
    pairwise["metric"] = "F2"
    pairwise["test"] = "Paired delta method"
    judgments = pd.read_csv(JUDGMENTS / "judgments.csv.gz", low_memory=False)
    refusal = select(judgments, COHORTS["wildguard_refusal"])
    refusal = refusal[
        refusal.policy_mode.eq("common_policy")
        & refusal.record_kind.eq("physical")
        & refusal.source_gold.notna()
    ]
    outcomes, keys = {}, {}
    for arm, rows in refusal.groupby("arm"):
        rows = rows.sort_values(["source_dataset", "item_id"])
        assert not rows.duplicated(["source_dataset", "item_id"]).any()
        keys[arm] = list(zip(rows.source_dataset, rows.item_id))
        outcomes[arm] = (
            rows.status.eq("ok") & rows.label.eq(rows.source_gold)
        ).to_numpy(dtype=float)
        assert np.isclose(outcomes[arm].mean(), accuracy.loc[arm, "rate"])
    bootstrap_checks = []
    rng = np.random.default_rng(20261019)
    for a, b in combinations(sorted(outcomes), 2):
        assert keys[a] == keys[b]
        oa, ob = outcomes[a], outcomes[b]
        delta = oa - ob
        n = len(delta)
        se = delta.std(ddof=1) / np.sqrt(n)
        awins, bwins = int((delta > 0).sum()), int((delta < 0).sum())
        p = binomtest(awins, awins + bwins, 0.5).pvalue if awins + bwins else 1.0
        match = (
            pairwise.cohort.eq("wildguard_refusal")
            & pairwise.arm_a.eq(a)
            & pairwise.arm_b.eq(b)
        )
        assert match.sum() == 1
        pairwise.loc[
            match,
            [
                "score_a",
                "score_b",
                "difference",
                "se_paired",
                "p_unadjusted",
                "ci_low",
                "ci_high",
                "metric",
                "test",
            ],
        ] = [
            oa.mean(),
            ob.mean(),
            delta.mean(),
            se,
            p,
            delta.mean() - norm.ppf(0.975) * se,
            delta.mean() + norm.ppf(0.975) * se,
            "Accuracy",
            "Exact McNemar",
        ]
        # A paired multinomial bootstrap checks the SE using the same examples.
        counts = np.array([bwins, n - awins - bwins, awins])
        draws = rng.multinomial(n, counts / n, size=10000)
        boot = (draws[:, 2] - draws[:, 0]) / n
        ratio = boot.std(ddof=1) / se if se else 1.0
        assert 0.9 < ratio < 1.1
        bootstrap_checks.append(
            dict(
                arm_a=a,
                arm_b=b,
                n=n,
                a_only_correct=awins,
                b_only_correct=bwins,
                bootstrap_se_ratio=ratio,
            )
        )
    assert len(pairwise) == 345
    pairwise["p_holm_all_345"] = multipletests(pairwise.p_unadjusted, method="holm")[1]
    pairwise["significant_holm_05"] = pairwise.p_holm_all_345.lt(0.05)
    z = norm.ppf(1 - 0.05 / (2 * len(pairwise)))
    pairwise["simultaneous_ci_low"] = pairwise.difference - z * pairwise.se_paired
    pairwise["simultaneous_ci_high"] = pairwise.difference + z * pairwise.se_paired
    pairwise.to_csv(OUT / "score_pairwise_all_345.csv", index=False)
    pd.DataFrame(bootstrap_checks).to_csv(
        OUT / "refusal_accuracy_paired_checks.csv", index=False
    )

    questions = (
        pd.read_csv(SOURCE / "clef_overall_evidence.csv").set_index("cohort").question
    )
    records = []
    for cohort in questions.index:
        ranking = scores[scores.cohort.eq(cohort)].sort_values(
            ["score", "arm"], ascending=[False, True]
        )
        clef = ranking[ranking.arm.eq("clef_choice")].iloc[0]
        peer = ranking[ranking.arm.ne("clef_choice")].iloc[0]
        comparisons = pairwise[
            pairwise.cohort.eq(cohort)
            & (pairwise.arm_a.eq("clef_choice") | pairwise.arm_b.eq("clef_choice"))
        ]
        differences = np.where(
            comparisons.arm_a.eq("clef_choice"),
            comparisons.difference,
            -comparisons.difference,
        )
        sig = comparisons.p_holm_all_345.lt(0.05)
        top = comparisons[
            comparisons.arm_a.eq(peer.arm) | comparisons.arm_b.eq(peer.arm)
        ].iloc[0]
        sign = 1 if top.arm_a == "clef_choice" else -1
        ci = sorted([100 * sign * top.ci_low, 100 * sign * top.ci_high])
        records.append(
            dict(
                cohort=cohort,
                question=questions[cohort],
                metric=clef.metric,
                clef_score=clef.score,
                clef_rank=ranking.arm.tolist().index("clef_choice") + 1,
                models_tested=len(ranking),
                best_alternative_arm=peer.arm,
                best_alternative=NAMES[peer.arm],
                alternative_score=peer.score,
                gap_to_best_alternative_pp=100 * (clef.score - peer.score),
                paired_ci_low_pp=ci[0],
                paired_ci_high_pp=ci[1],
                p_adjusted_vs_best=top.p_holm_all_345,
                significant_wins=int((sig & (differences > 0)).sum()),
                significant_losses=int((sig & (differences < 0)).sum()),
                unresolved=int((~sig).sum()),
            )
        )
    return (
        pd.DataFrame(records).query("cohort != @EXCLUDED").copy(),
        scores.query("cohort != @EXCLUDED").copy(),
        pairwise.query("cohort != @EXCLUDED").copy(),
    )


def refusal_chart(scores):
    from matplotlib.ticker import PercentFormatter

    data = scores[scores.cohort.eq("wildguard_refusal")].sort_values(
        ["score", "arm"], ascending=[False, True]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.23, right=0.88, top=0.78, bottom=0.22)
    y = np.arange(len(data))
    ax.errorbar(
        data.score,
        y,
        xerr=[data.score - data.ci_low, data.ci_high - data.score],
        fmt="o",
        color="#2C1FEB",
        ecolor="#956BDD",
        capsize=4,
        markersize=7,
    )
    for pos, row in enumerate(data.itertuples()):
        ax.text(
            1.03,
            pos,
            f"{row.score:.1%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontweight="bold",
        )
    ax.set_yticks(y, [NAMES[a] for a in data.arm])
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.set_xlim(0, 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Accuracy (higher is better)")
    clean_axes(ax, despine_left=True, grid_axis="x")
    fig.text(0.05, 0.94, "Did the assistant refuse?", fontsize=23, fontweight="bold")
    fig.text(
        0.05,
        0.89,
        "WildGuardTest: refusal detection, ranked by accuracy",
        fontsize=14,
        color="#6B6B76",
    )
    r = data.iloc[0]
    fig.text(
        0.05,
        0.845,
        f"{r.total_n:,} examples: {r.positive_n:,} refusals / {r.negative_n:,} non-refusals",
        fontsize=12,
        color="#6B6B76",
    )
    fig.text(
        0.05,
        0.065,
        "Lines: 95% Wilson confidence intervals. Missing or unreadable decisions count as incorrect.",
        fontsize=11,
        color="#6B6B76",
    )
    for suffix in ("png", "svg"):
        savefig(fig, OUT / f"04_refusal_detection_accuracy.{suffix}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_braintrust_theme()
    evidence, scores, pairwise = prepare()
    summary = dict(
        task_test_combinations=len(evidence),
        configurations=len(scores),
        comparisons=len(pairwise),
        significant_comparisons=int(pairwise.p_holm_all_345.lt(0.05).sum()),
        clef_observed_leads=int(evidence.clef_rank.eq(1).sum()),
        clef_significant_wins=int(evidence.significant_wins.sum()),
        clef_significant_losses=int(evidence.significant_losses.sum()),
        clef_unresolved=int(evidence.unresolved.sum()),
        original_holm_family=345,
        excluded_cohort=EXCLUDED,
        refusal_metric="Accuracy",
        screening_metric="F2",
    )
    assert summary["comparisons"] == 290
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    evidence.to_csv(OUT / "clef_evidence.csv", index=False)
    scores.to_csv(OUT / "score_metrics.csv", index=False)
    pairwise.to_csv(OUT / "score_pairwise_significance.csv", index=False)
    refusal_chart(scores)

    # These are descriptive counts of a fixed comparison set, not estimates of
    # a population win rate; additional uncertainty bars would be misleading.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    y = np.arange(len(evidence))
    offset = np.zeros(len(evidence))
    for column, color, label in [
        ("significant_wins", "#2C1FEB", "Clef significantly higher"),
        ("unresolved", "#F3F3F5", "Unresolved"),
        ("significant_losses", "#A15B96", "Clef significantly lower"),
    ]:
        values = evidence[column].to_numpy()
        ax.barh(
            y,
            values,
            left=offset,
            height=0.62,
            color=color,
            edgecolor="#B8B8C0" if column == "unresolved" else "white",
            linewidth=0.6,
            label=label,
        )
        for pos, (width, start) in enumerate(zip(values, offset)):
            if width:
                ax.text(
                    start + width / 2,
                    pos,
                    str(width),
                    va="center",
                    ha="center",
                    fontsize=12,
                    color="#18181B" if column == "unresolved" else "white",
                    fontweight="bold",
                )
        offset += values
    for pos, r in enumerate(evidence.itertuples()):
        ax.text(
            offset[pos] + 0.25,
            pos,
            f"{r.metric} rank {r.clef_rank}/{r.models_tested}",
            va="center",
            fontsize=11,
        )
    labels = evidence.question + "\n(" + evidence.metric + ")"
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_ylim(len(y) - 0.6, -0.6)
    ax.set_xlim(0, 13)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Number of other models in this question/test")
    ax.set_title(
        "Clef across six human-labeled task/test combinations",
        loc="left",
        pad=54,
        fontsize=21,
        fontweight="bold",
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3, frameon=False, fontsize=10
    )
    clean_axes(ax, despine_left=True, grid_axis="x")
    fig.tight_layout(pad=2.5)
    savefig(fig, OUT / "01_clef_evidence.png")
    plt.close(fig)

    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.subplots_adjust(
        left=0.15, right=0.93, top=0.84, bottom=0.12, wspace=0.67, hspace=1.05
    )
    titles = {
        "aegis_single": "Request screening · Aegis",
        "xstest_prompt": "Request screening · XSTest",
        "wildguard_response": "Answer safety · WildGuardTest",
        "aegis_conversation": "Whole exchange · Aegis",
        "wildguard_refusal": "Refusal detection · WildGuardTest",
        "wildguard_prompt": "Request with answer visible · WildGuardTest",
    }
    for ax, r in zip(axes.flat, evidence.itertuples()):
        data = (
            scores[scores.cohort.eq(r.cohort)]
            .set_index("arm")
            .loc[["clef_choice", r.best_alternative_arm]]
        )
        ax.barh(
            [0, 1],
            data.score,
            height=0.54,
            color=["#2C1FEB", "#956BDD"],
            edgecolor="white",
        )
        ax.errorbar(
            data.score,
            [0, 1],
            xerr=[data.score - data.ci_low, data.ci_high - data.score],
            fmt="none",
            ecolor="#18181B",
            capsize=4,
            elinewidth=1.3,
            zorder=4,
        )
        for pos, row in enumerate(data.itertuples()):
            ax.text(
                1.03,
                pos,
                f"{row.score:.1%}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=13,
                fontweight="bold",
            )
        ax.set_yticks([0, 1], ["Clef", r.best_alternative], fontsize=11)
        ax.set_ylim(1.6, -0.6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(
            titles[r.cohort] + f"\n{r.metric} · higher is better",
            loc="left",
            fontsize=12,
            pad=12,
            fontweight="bold",
        )
        clean_axes(ax, despine_left=True, grid_axis="x")
        result = (
            f"{r.best_alternative} is significantly higher"
            if r.p_adjusted_vs_best < 0.05
            else "Difference unresolved"
        )
        ax.text(
            0,
            -0.48,
            result,
            transform=ax.transAxes,
            fontsize=11,
            color="#18181B" if r.p_adjusted_vs_best < 0.05 else "#6B6B76",
        )
    fig.text(
        0.045,
        0.96,
        "How do the strongest models score?",
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.91,
        "Clef and the strongest alternative on each task. Scores use the same examples within each panel.",
        fontsize=13,
        color="#6B6B76",
    )
    fig.text(
        0.045,
        0.035,
        "Lines: 95% intervals. Significance uses paired tests with Holm correction across 345 comparisons.",
        fontsize=11,
        color="#6B6B76",
    )
    for suffix in ("png", "svg"):
        savefig(fig, OUT / f"02_clef_vs_best_alternative.{suffix}")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
