"""Editorial rendering of the completed October 19 results. No inference.

Rebuild with: .venv/bin/python -m analysis.article_charts
Statistics are read unchanged from the audited analysis tables. New title
proposals live outside the images; approved descriptive titles are retained.
"""

import hashlib
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from analysis.braintrust_viz import savefig, use_braintrust_theme, wilson_interval
from analysis.decision_common import COLORS, NAMES, ROOT, SOURCE, TYPES, pareto_mask

OUT = SOURCE / "decisions/editorial"
INK, MUTED, GRID = "#18181B", "#6B6B76", "#E8E8E8"
INDIGO, PURPLE, PINK = "#2C1FEB", "#956BDD", "#E6A4D3"
LABELS = {
    **NAMES,
    "luna_structured": "Luna · structured",
    "decisions_luna": "Luna · decisions",
}
TITLES = json.loads((ROOT / "analysis/decision_chart_titles.json").read_text())
CONTEXT = json.loads((ROOT / "analysis/context_chart_titles.json").read_text())
INPUTS = {
    "cost_quality": SOURCE / "decisions/response_cost_f2.csv",
    "errors": SOURCE / "decisions/error_rates.csv",
    "error_comparisons": SOURCE / "decisions/paired_comparisons.csv",
    "topics": SOURCE / "decisions/context/topic_rates.csv",
    "topic_comparisons": SOURCE / "decisions/context/topic_paired_tests.csv",
    "fine_tuning": ROOT
    / "analysis/output/custom-finetune-benchmark-2026-10-19/matched_metrics.csv",
}
FINE_TUNE_LABELS = {
    "qwen35_4b_letters": "Qwen base",
    "safety_tev": "Aegis fine-tune",
    "cc_repro": "Constitution fine-tune",
    "tev1_native": "Tev1",
}
FINE_TUNE_COLORS = {
    "qwen35_4b_letters": "#B8B8C5",
    "safety_tev": INDIGO,
    "cc_repro": PURPLE,
    "tev1_native": PINK,
}


def finish(fig, key, note):
    fig.text(0.025, 0.025, note, fontsize=10, color=MUTED)
    for ext in ["png", "svg"]:
        savefig(fig, OUT / f"{key}.{ext}")
    plt.close(fig)


def title(fig, text, subtitle=None):
    fig.text(0.025, 0.975, text, fontsize=22, weight="bold", va="top", color=INK)
    if subtitle:
        fig.text(0.025, 0.92, subtitle, fontsize=13, color=MUTED, va="top")


def axes_style(ax, limit, ticks):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, limit)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.tick_params(axis="both", length=0, labelsize=12, pad=8)


def error_bars(ax, d, color, marks=None, base=None):
    """Bars and intervals share the same denominator; no stacked denominators."""
    y = np.arange(len(d))
    v = 100 * d.rate.to_numpy()
    if base is None:
        ax.barh(y, v, height=0.56, color=color, edgecolor="white", zorder=2)
    else:
        b = 100 * base.rate.to_numpy()
        assert (v >= b - 1e-9).all()
        ax.barh(y, b, height=0.56, color=color, edgecolor="white", zorder=2)
        ax.barh(y, v - b, left=b, height=0.56, color=PINK, edgecolor="white", zorder=2)
    ax.errorbar(
        v,
        y,
        xerr=np.maximum(0, np.array([v - 100 * d.low, 100 * d.high - v])),
        fmt="none",
        color=INK,
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )
    for i, val in enumerate(v):
        ax.text(
            1.025,
            i,
            f"{val:.1f}%" + (marks[i] if marks else ""),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=12,
            weight="bold",
        )
    ax.set_ylim(len(d) - 0.55, -0.65)
    ax.set_yticks(y, [LABELS[a] for a in d.index])


def significant_marks(d, tests, **filters):
    t = tests.copy()
    for k, v in filters.items():
        t = t[t[k].eq(v)]
    # Preserve original arm-name tie break for the reference.
    ref = sorted(d.index, key=lambda a: (d.loc[a, "rate"], a))[0]
    marks = []
    for a in d.index:
        p = t[((t.arm_a == a) & (t.arm_b == ref)) | ((t.arm_b == a) & (t.arm_a == ref))]
        marks.append(
            " †"
            if len(p) == 1
            and p.p_holm.iloc[0] < 0.05
            and d.loc[a, "rate"] > d.loc[ref, "rate"]
            else ""
        )
    return marks, ref


def render_frontier():
    d = pd.read_csv(INPUTS["cost_quality"]).sort_values("dollars_per_10000")
    assert len(d) == 6 and d.total_n.eq(1681).all()
    assert np.array_equal(d.pareto_frontier, pareto_mask(d.dollars_per_10000, d.f2))
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.81, bottom=0.16)
    title(fig, TITLES["response_value"]["approved_title"])
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=COLORS[t], label=t)
            for t in ["Decision model", "General LLM"]
        ],
        loc="upper left",
        bbox_to_anchor=(0.09, 0.93),
        frameon=False,
        ncol=2,
        fontsize=12,
    )
    f = d[d.pareto_frontier]
    ax.plot(f.dollars_per_10000, 100 * f.f2, "--", color=INDIGO, lw=2, zorder=2)
    positions = {
        "jev_choice": (0.43, 90),
        "clef_flash_choice": (0.27, 64.5),
        "luna_structured": (1.05, 85.5),
        "decisions_luna": (0.83, 64.5),
        "clef_choice": (2.45, 78.5),
        "claude_structured": (27, 52),
    }
    for r in d.itertuples():
        x, y = r.dollars_per_10000, 100 * r.f2
        ax.errorbar(
            x,
            y,
            xerr=[[x - r.low], [r.high - x]],
            yerr=[[y - 100 * r.ci_low], [100 * r.ci_high - y]],
            fmt="none",
            ecolor=MUTED,
            capsize=4,
            elinewidth=1.3,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            s=160,
            color=COLORS[TYPES[r.arm]],
            edgecolors="white",
            linewidths=1.5,
            zorder=4,
        )
        if r.pareto_frontier:
            ax.scatter(
                x,
                y,
                s=285,
                facecolors="none",
                edgecolors=INDIGO,
                linewidths=1.2,
                zorder=5,
            )
        tx, ty = positions[r.arm]
        name = LABELS[r.arm] + (" †" if r.lower_f2_than_best_significant else "")
        ax.annotate(
            name + f"\n{y:.1f}% · ${x:.2f}",
            (x, y),
            xytext=(tx, ty),
            ha="right" if r.arm == "claude_structured" else "left",
            va="center",
            fontsize=12,
            weight="bold",
            linespacing=1.5,
            arrowprops=dict(arrowstyle="-", color="#B7B7C0", lw=0.8),
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.23, 50)
    ax.set_ylim(47, 94)
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 5, 10, 20, 40]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel(
        "API cost per 10,000 checks · log scale · lower is better",
        fontsize=13,
        labelpad=14,
    )
    ax.set_ylabel("F2 score · higher is better", fontsize=13, labelpad=12)
    ax.grid(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=12, pad=8)
    finish(
        fig,
        "response_value",
        "95% intervals · Rings and dashed line: observed cost–quality frontier · † Lower F2 than Jev Choice · Notes 1–2",
    )


def render_errors(key, cohort):
    rates = pd.read_csv(INPUTS["errors"])
    tests = pd.read_csv(INPUTS["error_comparisons"])
    g = rates[rates.cohort.eq(cohort) & rates["slice"].eq("all")]
    first = g[g.metric.eq("miss")].set_index("arm").sort_values(["rate", "arm"])
    arms = first.index.tolist()
    assert len(arms) == (11 if cohort == "aegis_single" else 10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True)
    fig.subplots_adjust(left=0.23, right=0.88, top=0.79, bottom=0.14, wspace=0.46)
    title(fig, TITLES[key]["approved_title"])
    if cohort == "wildguard_response":
        fig.legend(
            handles=[
                Patch(color=INDIGO, label="Labeled safe"),
                Patch(color=PINK, label="No verdict"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.225, 0.92),
            ncol=2,
            frameon=False,
            fontsize=11,
        )
    else:
        fig.text(
            0.23,
            0.9,
            "Fewer misses and fewer false alarms are better",
            fontsize=12,
            color=MUTED,
        )
    refs = []
    for ax, metric, label, color in zip(
        axes,
        ["miss", "false_alarm"],
        [
            "Unsafe "
            + ("requests" if cohort == "aegis_single" else "answers")
            + " missed",
            "Safe "
            + ("requests" if cohort == "aegis_single" else "answers")
            + " flagged",
        ],
        [INDIGO, PURPLE],
    ):
        d = g[g.metric.eq(metric)].set_index("arm").loc[arms]
        marks, ref = significant_marks(
            d, tests, cohort=cohort, metric=metric, slice="all"
        )
        refs.append(ref)
        base = (
            g[g.metric.eq("false_safe")].set_index("arm").loc[arms]
            if cohort == "wildguard_response" and metric == "miss"
            else None
        )
        axes_style(
            ax,
            50 if cohort == "wildguard_response" else 40,
            [0, 10, 20, 30, 40, 50]
            if cohort == "wildguard_response"
            else [0, 10, 20, 30, 40],
        )
        error_bars(ax, d, color, marks, base)
        ax.set_title(label, loc="left", fontsize=14, weight="bold", pad=18)
        ax.set_xlabel(
            "Share of unsafe examples"
            if metric == "miss"
            else "Share of safe examples",
            fontsize=12,
            labelpad=14,
        )
    note = (
        "95% intervals · † Higher error than the lowest observed rate in that panel · Note "
        + ("3" if cohort == "aegis_single" else "4")
    )
    finish(fig, key, note)
    return refs


def render_topics():
    rates = pd.read_csv(INPUTS["topics"])
    tests = pd.read_csv(INPUTS["topic_comparisons"])
    topics = [
        ("Suicide and Self Harm", "Self-harm"),
        ("Profanity", "Profanity"),
        ("PII/Privacy", "Privacy"),
        ("Unauthorized Advice", "Unauthorized advice"),
    ]
    d = rates[rates.cohort.eq("aegis_single") & rates.metric.eq("miss")]
    arms = [a for a in NAMES if a in set(d.arm)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    fig.subplots_adjust(
        left=0.18, right=0.92, top=0.86, bottom=0.10, wspace=0.92, hspace=0.42
    )
    title(
        fig,
        CONTEXT["topic_failures"]["approved_title"],
        "Requests labeled unsafe that were not flagged · lower is better",
    )
    for ax, (topic, name) in zip(axes.flat, topics):
        g = d[d.topic.eq(topic)].set_index("arm").loc[arms]
        marks, ref = significant_marks(g, tests, topic=topic)
        axes_style(ax, 100, [0, 25, 50, 75, 100])
        error_bars(ax, g, INDIGO, marks)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(name, loc="left", fontsize=16, weight="bold", pad=16)
    finish(
        fig,
        "topic_failures",
        "95% intervals · † Higher error than the lowest observed rate in that topic · Advice labels include disputed cases · Note 5",
    )


def render_fine_tuning():
    frame = pd.read_csv(INPUTS["fine_tuning"])
    names = FINE_TUNE_LABELS
    arms = list(names)
    assert set(frame.arm) == set(arms), "Update labels for the matched comparison"
    panels = [
        ("aegis_single", "Requests · Aegis"),
        ("aegis_conversation", "Exchanges · Aegis"),
        ("wildguard_prompt", "Requests · WildGuardTest"),
        ("wildguard_response", "Answers · WildGuardTest"),
        ("xstest_prompt", "Requests · XSTest"),
        ("wildguard_refusal", "Refusals · WildGuardTest"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(
        left=0.19, right=0.92, top=0.86, bottom=0.10, wspace=1.0, hspace=0.68
    )
    title(
        fig,
        "Fine-tuned model performance",
        "Base model, two custom fine-tunes, and Tev1 · higher is better",
    )
    for ax, (cohort, name) in zip(axes.flat, panels):
        d = frame[frame.cohort.eq(cohort)].set_index("arm").loc[arms]
        y = np.arange(len(arms))
        refusal = cohort == "wildguard_refusal"
        if refusal:
            v = 100 * d.accuracy.to_numpy()
            intervals = (
                np.array(
                    [
                        wilson_interval(round(r.accuracy * r.n), int(r.n))
                        for r in d.itertuples()
                    ]
                )
                * 100
            )
            low, high = intervals[:, 0], intervals[:, 1]
        else:
            v, low, high = [100 * d[c].to_numpy() for c in ["f2", "ci_low", "ci_high"]]
        colors = [FINE_TUNE_COLORS[a] for a in arms]
        axes_style(ax, 100, [0, 25, 50, 75, 100])
        ax.barh(y, v, color=colors, height=0.55, edgecolor="white", zorder=2)
        ax.errorbar(
            v, y, xerr=[v - low, high - v], fmt="none", ecolor=INK, capsize=3, zorder=4
        )
        for i, val in enumerate(v):
            ax.text(
                1.03,
                i,
                f"{val:.1f}%",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=12,
                weight="bold",
            )
        ax.set_yticks(y, [names[a] for a in arms])
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(len(arms) - 0.6, -0.6)
        ax.set_title(name, loc="left", fontsize=14, weight="bold", pad=15)
        ax.set_xlabel(
            "Refusal detection accuracy" if refusal else "F2 score",
            fontsize=12,
            labelpad=9,
        )
    finish(
        fig,
        "custom_finetune_article",
        "95% intervals · Separate exploratory comparison; observed gains are task-specific · Note 7",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_braintrust_theme()
    render_frontier()
    refs = {
        k: render_errors(k, c)
        for k, c in [
            ("request_value", "aegis_single"),
            ("response_failures", "wildguard_response"),
        ]
    }
    render_topics()
    render_fine_tuning()
    manifest = {
        "data_revision": "source_aligned_v1_20261019",
        "inference_calls": 0,
        "comparison_references": refs,
        "inputs": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS.values()
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    keys = [
        "response_value",
        "request_value",
        "response_failures",
        "topic_failures",
        "custom_finetune_article",
    ]
    proposals = [
        "Higher spending did not buy better answer screening.",
        "The fewest missed requests came with more false alarms.",
        "A classifier can miss harm by misjudging it—or declining to judge.",
        "Models agreed on self-harm more often than on advice.",
        "Fine-tuning improved request screening without improving answer screening.",
    ]
    pd.DataFrame(
        {"chart": keys, "proposed_title_for_author_rewrite": proposals}
    ).to_csv(OUT / "titles_for_author.csv", index=False)
    (OUT / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Article plots</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:18px system-ui;color:#18181b}img{width:100%}section{margin:70px 0}h1{font-size:28px}.proposal{color:#6b6b76;font-size:15px}</style><h1>Revised article plots</h1><p>Completed October 19 data. Statistical details are in the article notes. Existing descriptive image titles are retained; these optional replacements await the author’s rewrite.</p>'
        + "".join(
            '<section><p class="proposal">Title proposal for your rewrite</p><p><b>'
            + html.escape(proposal)
            + '</b></p><img src="'
            + k
            + '.png" alt="'
            + html.escape(k)
            + '"></section>'
            for k, proposal in zip(keys, proposals)
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
