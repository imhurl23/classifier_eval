"""Decision-oriented bar charts from completed revision results; no inference.

Titles are rendered ONLY from approved_title in decision_chart_titles.json.
Proposed titles are exported separately for the author's rewrite. All statistical
details remain in CSVs; plots show Wilson intervals and a single paired-test mark.
"""

import html
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from analysis.braintrust_viz import (
    clean_axes,
    savefig,
    use_braintrust_theme,
    wilson_interval,
)
from analysis.decision_common import COLORS, NAMES, ROOT, SOURCE, TYPES, pareto_mask
from analysis.plot_safety_usecases import COHORTS, select

OUT = SOURCE / "decisions"
TITLE_FILE = ROOT / "analysis/decision_chart_titles.json"
METRICS = {
    "miss": ("Unsafe not flagged", "per 100 unsafe examples"),
    "false_safe": ("Unsafe labeled safe", "per 100 unsafe examples"),
    "false_alarm": ("Safe labeled unsafe", "per 100 safe examples"),
    "no_verdict": ("No usable decision", "per 100 examples"),
    "refusal_miss": ("Refusals missed", "per 100 refusals"),
    "refusal_false_alarm": ("Compliance labeled refusal", "per 100 non-refusals"),
}


def endpoint(frame, cohort, metric, slice_name="all"):
    group = select(frame, COHORTS[cohort])
    if slice_name != "all":
        group = group[group.adversarial.eq(slice_name == "adversarial")]
    if metric in {"miss", "false_safe"}:
        group = group[group.source_gold.eq("unsafe")]
        event = (
            ~(group.status.eq("ok") & group.label.eq("unsafe"))
            if metric == "miss"
            else group.status.eq("ok") & group.label.eq("safe")
        )
    elif metric == "false_alarm":
        group = group[group.source_gold.eq("safe")]
        event = group.status.eq("ok") & group.label.eq("unsafe")
    elif metric == "no_verdict":
        event = ~group.status.eq("ok")
    elif metric == "refusal_miss":
        group = group[group.source_gold.eq("refusal")]
        event = ~(group.status.eq("ok") & group.label.eq("refusal"))
    elif metric == "refusal_false_alarm":
        group = group[group.source_gold.eq("not_refusal")]
        event = group.status.eq("ok") & group.label.eq("refusal")
    else:
        raise ValueError(metric)
    return group.assign(event=event.astype(int))


def summarize_endpoint(frame, cohort, metric, slice_name="all"):
    group = endpoint(frame, cohort, metric, slice_name)
    rows = []
    for arm, g in group.groupby("arm"):
        k, n = int(g.event.sum()), len(g)
        low, high = wilson_interval(k, n)
        rows.append(
            dict(
                cohort=cohort,
                metric=metric,
                slice=slice_name,
                arm=arm,
                model_type=TYPES[arm],
                k=k,
                n=n,
                rate=k / n,
                low=low,
                high=high,
            )
        )
    return pd.DataFrame(rows)


def paired_tests(frame, endpoints):
    rows = []
    for cohort, metric, slice_name in endpoints:
        g = endpoint(frame, cohort, metric, slice_name)
        matrix = g.pivot(
            index=["source_dataset", "item_id"], columns="arm", values="event"
        )
        assert not matrix.isna().any().any(), (cohort, metric, slice_name)
        for a, b in combinations(sorted(matrix.columns), 2):
            av, bv = matrix[a].astype(bool), matrix[b].astype(bool)
            ab = int((av & ~bv).sum())
            ba = int((~av & bv).sum())
            p = binomtest(ab, ab + ba).pvalue if ab + ba else 1.0
            rows.append(
                dict(
                    cohort=cohort,
                    metric=metric,
                    slice=slice_name,
                    arm_a=a,
                    arm_b=b,
                    a_only_errors=ab,
                    b_only_errors=ba,
                    n=len(matrix),
                    p_raw=p,
                )
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result.p_raw, method="holm")[1]
    return result


def cost_stats(frame, cohort):
    rows = []
    for arm, g in select(frame, COHORTS[cohort]).groupby("arm"):
        if not g.provider.eq("hosted").all():
            continue
        # No zero substitution for missing usage. Quotes apply to this case mix
        # and historical rates, not provider invoices or a production traffic mix.
        observed = g.token_cost_usd.dropna()
        if observed.empty:
            continue
        costs = observed.to_numpy(float) * 10000
        mean = costs.mean()
        se = costs.std(ddof=1) / np.sqrt(len(costs))
        rows.append(
            dict(
                cohort=cohort,
                arm=arm,
                n=len(costs),
                total_calls=len(g),
                usage_coverage=len(costs) / len(g),
                dollars_per_10000=mean,
                low=max(0.0, mean - 1.96 * se),
                high=mean + 1.96 * se,
            )
        )
    return pd.DataFrame(rows)


def chart_specs():
    return {
        "request_value": [
            ("aegis_single", "miss", "all"),
            ("aegis_single", "false_alarm", "all"),
            ("aegis_single", "cost", "all"),
        ],
        "response_value": [
            ("wildguard_response", "miss", "all"),
            ("wildguard_response", "false_alarm", "all"),
            ("wildguard_response", "cost", "all"),
        ],
        "response_failures": [
            ("wildguard_response", "false_safe", "all"),
            ("wildguard_response", "false_alarm", "all"),
            ("wildguard_response", "no_verdict", "all"),
        ],
        "benign_requests": [
            ("aegis_single", "false_alarm", "all"),
            ("xstest_prompt", "false_alarm", "all"),
        ],
        "adversarial_answers": [
            ("wildguard_response", "miss", "vanilla"),
            ("wildguard_response", "miss", "adversarial"),
        ],
        "exchange_review": [
            ("aegis_conversation", "miss", "all"),
            ("aegis_conversation", "false_alarm", "all"),
            ("aegis_conversation", "no_verdict", "all"),
        ],
        "refusal_audit": [
            ("wildguard_refusal", "refusal_miss", "all"),
            ("wildguard_refusal", "refusal_false_alarm", "all"),
        ],
    }


def render_cost_frontier(source, costs, title_record, out):
    cohort = "wildguard_response"
    f2 = pd.read_csv(source / "f2_metrics.csv")
    data = f2[f2.cohort.eq(cohort)].merge(
        costs[costs.cohort.eq(cohort)], on=["arm", "cohort"], validate="one_to_one"
    )
    data = data.sort_values("dollars_per_10000").copy()
    assert (
        len(data) == 6
        and data.total_n.nunique() == 1
        and data.usage_coverage.eq(1).all()
    )
    data["pareto_frontier"] = pareto_mask(data.dollars_per_10000, data.f2)
    pairs = pd.read_csv(source / "f2_pairwise_significance.csv")
    best = data.loc[data.f2.idxmax(), "arm"]

    def lower_significant(row):
        p = pairs[
            pairs.cohort.eq(cohort)
            & (
                (pairs.arm_a.eq(row.arm) & pairs.arm_b.eq(best))
                | (pairs.arm_b.eq(row.arm) & pairs.arm_a.eq(best))
            )
        ]
        return bool(
            len(p) == 1 and p.p_holm_all_290.iloc[0] < 0.05 and row.f2 < data.f2.max()
        )

    data["lower_f2_than_best_significant"] = data.apply(lower_significant, axis=1)
    data.to_csv(out / "response_cost_f2.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.82, bottom=0.25)
    frontier = data[data.pareto_frontier]
    ax.plot(
        frontier.dollars_per_10000,
        100 * frontier.f2,
        linestyle="--",
        color="#6B6B76",
        linewidth=1.5,
        zorder=2,
    )
    offsets = {
        "jev_choice": (15, 30),
        "clef_flash_choice": (-3, -58),
        "decisions_luna": (14, -38),
        "luna_structured": (15, 31),
        "clef_choice": (18, 14),
        "claude_structured": (-15, -54),
    }
    for r in data.itertuples():
        x, y = r.dollars_per_10000, 100 * r.f2
        ax.errorbar(
            x,
            y,
            xerr=[[max(0, x - r.low)], [max(0, r.high - x)]],
            yerr=[[100 * (r.f2 - r.ci_low)], [100 * (r.ci_high - r.f2)]],
            fmt="none",
            ecolor="#6B6B76",
            capsize=4,
            elinewidth=1.3,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            s=140,
            color=COLORS[TYPES[r.arm]],
            edgecolors="white",
            linewidths=1.3,
            zorder=4,
        )
        if r.pareto_frontier:
            ax.scatter(
                x,
                y,
                s=230,
                facecolors="none",
                edgecolors="#18181B",
                linewidths=1.1,
                zorder=5,
            )
        label = NAMES[r.arm] + (" †" if r.lower_f2_than_best_significant else "")
        ax.annotate(
            label,
            (x, y),
            xytext=offsets[r.arm],
            textcoords="offset points",
            ha="right" if r.arm == "claude_structured" else "left",
            va="center",
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#9F9FA9", lw=0.7),
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.24, 50)
    ax.set_ylim(45, 96)
    ticks = [0.25, 0.5, 1, 2, 5, 10, 20, 40]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Estimated API cost per 10,000 checks (log scale)", fontsize=13)
    ax.set_ylabel("F2 score", fontsize=13)
    clean_axes(ax, grid_axis="both")
    title = title_record.get("approved_title")
    if title:
        fig.text(0.025, 0.98, title, fontsize=21, fontweight="bold", va="top")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=COLORS[t], label=t)
            for t in ["Decision model", "General LLM"]
        ]
        + [
            Line2D(
                [],
                [],
                color="#6B6B76",
                linestyle="--",
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#18181B",
                label="Observed Pareto frontier",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.095, 0.93),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.11, 0.79, "Higher F2 and lower cost are better", fontsize=11, color="#6B6B76"
    )
    fig.text(
        0.025,
        0.14,
        "WildGuardTest answer harm · 1,681 examples per model. Lines: 95% intervals; the frontier uses observed point estimates.",
        fontsize=10,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.095,
        "† Lower F2 than Jev Choice after paired tests and Holm correction. Unmarked differences remain unresolved.",
        fontsize=10,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.05,
        "Cost: recorded token usage at October 12 rates; excludes retries, fees and GPU hosting. F2 gives recall more weight than precision.",
        fontsize=10,
        color="#6B6B76",
    )
    for ext in ["png", "svg"]:
        savefig(fig, out / f"response_value.{ext}")
    plt.close(fig)


def render(key, specs, rates, costs, tests, title_record, out):
    first = rates[
        (rates.cohort == specs[0][0])
        & (rates.metric == specs[0][1])
        & (rates["slice"] == specs[0][2])
    ]
    has_cost = any(s[1] == "cost" for s in specs)
    if has_cost:
        c = costs[costs.cohort == specs[0][0]].sort_values("dollars_per_10000")
        arms = c.arm.tolist()
    else:
        first = first.assign(
            type_order=first.arm.map(lambda a: list(COLORS).index(TYPES[a]))
        )
        arms = first.sort_values(["type_order", "rate", "arm"]).arm.tolist()
        # Cross-dataset panels compare only interfaces tested in every panel.
        for cohort, metric, slice_name in specs:
            available = set(
                rates.loc[
                    rates.cohort.eq(cohort)
                    & rates.metric.eq(metric)
                    & rates["slice"].eq(slice_name),
                    "arm",
                ]
            )
            arms = [a for a in arms if a in available]
    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(16 if len(specs) == 3 else 13, max(7.5, 0.48 * len(arms) + 3.0)),
        sharey=True,
    )
    fig.subplots_adjust(left=0.20, right=0.92, top=0.82, bottom=0.25, wspace=0.40)
    labels = []
    for ax, (cohort, metric, slice_name) in zip(axes, specs):
        if metric == "cost":
            data = costs[costs.cohort.eq(cohort)].set_index("arm").loc[arms]
            value, low, high = (
                data[c].to_numpy() for c in ["dollars_per_10000", "low", "high"]
            )
            name, unit = "Estimated API cost", "USD per 10,000 checks"
            reference = None
            notes = [""] * len(arms)
        else:
            data = (
                rates[
                    rates.cohort.eq(cohort)
                    & rates.metric.eq(metric)
                    & rates["slice"].eq(slice_name)
                ]
                .set_index("arm")
                .loc[arms]
            )
            value, low, high = (
                100 * data[c].to_numpy() for c in ["rate", "low", "high"]
            )
            name, unit = METRICS[metric]
            if key == "benign_requests":
                name = (
                    "Aegis · safe requests"
                    if cohort == "aegis_single"
                    else "XSTest · safe requests"
                )
            if key == "adversarial_answers":
                name = (
                    "Ordinary prompts"
                    if slice_name == "vanilla"
                    else "Adversarial prompts"
                )
            reference = sorted(arms, key=lambda a: (data.loc[a, "rate"], a))[0]
            notes = []
            for arm in arms:
                pair = tests[
                    tests.cohort.eq(cohort)
                    & tests.metric.eq(metric)
                    & tests["slice"].eq(slice_name)
                    & (
                        (tests.arm_a.eq(arm) & tests.arm_b.eq(reference))
                        | (tests.arm_a.eq(reference) & tests.arm_b.eq(arm))
                    )
                ]
                significant = (
                    len(pair) == 1
                    and pair.p_holm.iloc[0] < 0.05
                    and data.loc[arm, "rate"] > data.loc[reference, "rate"]
                )
                notes.append(" †" if significant else "")
            position = ax.get_position()
            fig.text(
                position.x0,
                0.135,
                "Compared with " + NAMES[reference],
                fontsize=9,
                color="#6B6B76",
            )
        y = np.arange(len(arms))
        ax.barh(
            y,
            value,
            color=[COLORS[TYPES[a]] for a in arms],
            height=0.59,
            edgecolor="white",
            zorder=2,
        )
        ax.errorbar(
            value,
            y,
            xerr=np.maximum(0, np.array([value - low, high - value])),
            fmt="none",
            color="#18181B",
            elinewidth=1.1,
            capsize=3,
            zorder=4,
        )
        maximum = max(float(high.max()), 1.0 if metric == "cost" else 2.0)
        ax.set_xlim(0, maximum * 1.18)
        for i, v in enumerate(value):
            txt = (f"${v:.2f}" if metric == "cost" else f"{v:.1f}") + notes[i]
            labels.append(
                ax.text(
                    1.02,
                    i,
                    txt,
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                )
            )
        ax.set_yticks(y, [NAMES[a] for a in arms])
        ax.set_ylim(len(arms) - 0.6, -0.6)
        ax.set_xlabel(unit, fontsize=11)
        ax.text(0, 1.05, name, transform=ax.transAxes, fontsize=12, fontweight="bold")
        if metric != "cost":
            ns = data.n.unique()
            assert len(ns) == 1
            ax.text(
                0,
                1.005,
                f"n = {int(ns[0]):,} per model",
                transform=ax.transAxes,
                fontsize=9,
                color="#6B6B76",
            )
        clean_axes(ax, despine_left=True, grid_axis="x")
    if key in {"benign_requests", "adversarial_answers"}:
        common_limit = max(ax.get_xlim()[1] for ax in axes)
        for ax in axes:
            ax.set_xlim(0, common_limit)
    title = title_record.get("approved_title")
    if title:
        fig.text(
            0.025, 0.98, title, ha="left", va="top", fontsize=20, fontweight="bold"
        )
    fig.legend(
        handles=[
            Patch(color=v, label=k)
            for k, v in COLORS.items()
            if any(TYPES[a] == k for a in arms)
        ],
        loc="upper left",
        bbox_to_anchor=(0.20, 0.94),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    foot = "Lower is better. Lines: 95% intervals. † Higher error than the named model after paired tests; unmarked gaps remain unresolved."
    foot2 = "An unmarked difference is unresolved, not proof of equal performance. Model types describe the tested models; no type averages are shown."
    if has_cost:
        foot2 = "Cost uses calls with recorded usage and October 12 rates; GPU costs are separate. Missing decisions count as unsafe not flagged."
    if key == "adversarial_answers":
        foot2 = "Different examples occur in the two groups. This is a descriptive slice comparison, not a controlled measure of attack impact."
    if key == "benign_requests":
        foot2 = "Only interfaces tested on both datasets are shown; Jev Noul was not tested on XSTest. Unmarked gaps are unresolved, not equal."
    fig.text(0.025, 0.075, foot, fontsize=9.5, color="#6B6B76")
    fig.text(0.025, 0.035, foot2, fontsize=9.5, color="#6B6B76")
    for ext in ("png", "svg"):
        savefig(fig, out / f"{key}.{ext}")
    plt.close(fig)


def main(source=SOURCE, out=OUT):
    source, out = Path(source), Path(out)
    if not (source / "summary.json").exists():
        raise ValueError("Wait for complete verified revision analysis")
    frame = pd.read_csv(source / "judgments.csv.gz", low_memory=False)
    assert len(frame) == 89251 and frame.job_id.is_unique
    rows = [
        json.loads(l)
        for l in (ROOT / "data/processed/source-full/000bf1bcb6dc/sample.jsonl").open()
    ]
    adversarial = {r["id"]: r["metadata"].get("wildguard_adversarial") for r in rows}
    frame["adversarial"] = frame.item_id.map(adversarial)
    specs = chart_specs()
    endpoints = sorted(
        {tuple(s) for values in specs.values() for s in values if s[1] != "cost"}
    )
    rates = pd.concat(
        [summarize_endpoint(frame, *s) for s in endpoints], ignore_index=True
    )
    tests = paired_tests(frame, endpoints)
    costs = pd.concat(
        [cost_stats(frame, k) for k in ["aegis_single", "wildguard_response"]],
        ignore_index=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    rates.to_csv(out / "error_rates.csv", index=False)
    tests.to_csv(out / "paired_comparisons.csv", index=False)
    costs.to_csv(out / "api_costs.csv", index=False)
    titles = json.loads(TITLE_FILE.read_text())
    active = {k: v for k, v in titles.items() if v["placement"] == "main"}
    pd.DataFrame([{"chart": k, **v} for k, v in active.items()]).to_csv(
        out / "titles_for_author.csv", index=False
    )
    pd.DataFrame(
        [{"chart": k, **v} for k, v in titles.items() if k not in active]
    ).to_csv(out / "excluded_charts.csv", index=False)
    archive = out / "archive"
    archive.mkdir(exist_ok=True)
    for key in set(titles) - set(active):
        for ext in ("png", "svg"):
            path = out / f"{key}.{ext}"
            if path.exists():
                path.replace(archive / path.name)
    use_braintrust_theme()
    for key in active:
        if key == "response_value":
            render_cost_frontier(source, costs, titles[key], out)
        else:
            render(key, specs[key], rates, costs, tests, titles[key], out)
    # The reviewer sees proposals outside images. They never become figure titles
    # unless the author explicitly supplies approved_title.
    blocks = []
    for key, record in active.items():
        blocks.append(
            '<section><p class="tag">'
            + html.escape(record["placement"].upper())
            + " · "
            + html.escape(key)
            + "</p>"
            "<p><b>Decision:</b> " + html.escape(record["decision"]) + "</p>"
            "<p><b>Title proposal for your rewrite:</b> "
            + html.escape(record["proposed_title"])
            + "</p>"
            "<p><b>What the results show:</b> "
            + html.escape(record["finding"])
            + "</p>"
            "<p><b>How to use this:</b> "
            + html.escape(record["use_in_decision"])
            + "</p>"
            '<img src="'
            + key
            + '.png" alt="'
            + html.escape(record["decision"])
            + '"></section>'
        )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Analysis review</title>'
        "<style>body{font:17px system-ui;color:#18181b;max-width:1400px;margin:40px auto;padding:0 28px}section{margin:45px 0 70px}img{width:100%}.tag{font-size:13px;color:#6b6b76}</style>"
        "<h1>Analysis review</h1><p>Read the evidence by the decision it can inform. Figure titles are awaiting your rewrite.</p>"
        "<p>Four findings: lower cost can coexist with better detection; fewer misses can come with more false alarms; classifiers can fail to return a verdict; and source-topic boundaries can dominate disagreement.</p>"
        "<p>These benchmarks support candidate selection for harm classification. They do not validate tool authorization, legal compliance, or deployment without further evaluation.</p>"
        "<p>The policies were revised after examining disagreements with source labels. These results are exploratory, not a fresh held-out test. Labels remain unchanged; XSTest results are reused. The intervals capture sampling uncertainty, not uncertainty in the labels or policy interpretation.</p>"
        "<p>Costs use recorded token usage and historical October 12 rates. They exclude retries, gateway fees, and GPU deployment costs. Error rates use the published labels as the reference. No alert precision or production cost is implied without the intended traffic mix.</p>"
        + "".join(blocks)
    )
    report = dict(
        charts=len(active),
        total_narrative_charts=len(active) + 1,
        main_charts=len(active),
        excluded_charts=[k for k in titles if k not in active],
        significance_family=len(tests),
        statistical_test="Exact paired McNemar; original global Holm family retained despite editorial cuts",
        intervals="Wilson 95% for error rates; normal mean 95% for recorded token-cost sampling variation",
        titles_rendered=[k for k, v in active.items() if v.get("approved_title")],
        caveats=[
            "Sampling intervals do not capture annotation or policy uncertainty.",
            "Cost intervals do not capture pricing changes, retries, or gateway fees.",
            "No deployment guarantee or model-family causal claim; underlying models recur across interfaces.",
        ],
    )
    report["response_value_chart"] = (
        "F2 versus cost scatter; observed Pareto frontier; paired F2 tests use the saved 290-comparison Holm family"
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Safety decisions: revised analysis\n\nStart with the decision each comparison can inform. "
            "Title proposals are separate from the images and await the author’s rewrite. "
            "Policies were revised after inspection of disagreements; labels are unchanged and XSTest is reused."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "root=Path.cwd()\nwhile not (root/'analysis').exists():\n    root=root.parent\n"
            "out=root/'analysis/output/source-policy-revision-2026-10-19/decisions'\n"
            "display(pd.read_csv(out/'titles_for_author.csv')[['chart','decision','proposed_title']])"
        ),
    ]
    for key, record in active.items():
        nb.cells += [
            nbformat.v4.new_markdown_cell(
                "**Proposed title for your rewrite:** "
                + record["proposed_title"]
                + "\n\n**What the results show:** "
                + record["finding"]
                + "\n\n**How to use this:** "
                + record["use_in_decision"]
            ),
            nbformat.v4.new_code_cell(f"display(Image(filename=str(out/'{key}.png')))"),
        ]
    nb.cells += [
        nbformat.v4.new_markdown_cell(
            "Supporting counts, costs, and paired tests remain in `error_rates.csv`, `api_costs.csv`, and `paired_comparisons.csv`. The original test family is retained after chart selection."
        )
    ]
    nbformat.write(nb, ROOT / "analysis/source_policy_revision_exploration.ipynb")
    from analysis.decision_context import main as build_context_evidence

    build_context_evidence()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
