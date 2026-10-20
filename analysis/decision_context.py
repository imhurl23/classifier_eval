"""Offline topic and prompt-design evidence, retaining matched item identities."""

import html
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from analysis.braintrust_viz import (
    clean_axes,
    savefig,
    use_braintrust_theme,
    wilson_interval,
)
from analysis.decision_common import COLORS, NAMES, ROOT, SOURCE, TYPES
from analysis.plot_safety_usecases import COHORTS, select

OUT = SOURCE / "decisions/context"
TOPICS = ["PII/Privacy", "Unauthorized Advice", "Suicide and Self Harm", "Profanity"]
TOPIC_NAMES = ["Privacy", "Unauthorized advice", "Self-harm", "Profanity"]


def collapse(frame):
    """Preserve caution and failure as distinct operational outcomes."""
    labels = frame.label.fillna("missing")
    return pd.Series(
        np.where(
            ~frame.status.eq("ok"),
            "no_verdict",
            np.where(
                labels.eq("safe"),
                "safe",
                np.where(labels.eq("needs_caution"), "needs_caution", "unsafe"),
            ),
        ),
        index=frame.index,
    )


def matched(base, variant):
    assert (
        not base.item_id.duplicated().any() and not variant.item_id.duplicated().any()
    )
    assert set(base.item_id) == set(variant.item_id)
    a = base.assign(outcome=collapse(base)).set_index("item_id").sort_index()
    b = variant.assign(outcome=collapse(variant)).set_index("item_id").sort_index()
    assert a.source_gold.equals(b.source_gold)
    ac, bc = a.outcome.eq(a.source_gold), b.outcome.eq(b.source_gold)
    changed = a.outcome.ne(b.outcome)
    details = pd.DataFrame(
        dict(
            base=a.outcome,
            variant=b.outcome,
            gold=a.source_gold,
            changed=changed,
            gained=~ac & bc,
            lost=ac & ~bc,
        )
    ).reset_index()
    gained, lost = int(details.gained.sum()), int(details.lost.sum())
    return dict(
        n=len(a),
        changed=int(changed.sum()),
        gained=gained,
        lost=lost,
        base_correct=int(ac.sum()),
        variant_correct=int(bc.sum()),
        both_wrong=int((~ac & ~bc).sum()),
        p_raw=binomtest(gained, gained + lost).pvalue if gained + lost else 1.0,
    ), details


def diagnostic_tables(d):
    rows, cases = [], []

    def add(kind, arm, condition, base, variant):
        stats, detail = matched(base, variant)
        key = dict(kind=kind, arm=arm, condition=condition)
        rows.append(dict(**key, **stats))
        cases.append(detail.assign(**key))

    for arm, g in d.groupby("arm"):
        r = g[g.stage.eq("robustness")]
        binary = r[r.scheme.eq("s2")]
        long = binary[binary.variant.eq("long")]
        for variant in ["short", "neutral_long", "withheld"]:
            add("policy", arm, variant, long, binary[binary.variant.eq(variant)])
        for scheme in ["s3", "s6", "s13"]:
            alt = r[r.scheme.eq(scheme) & r.variant.eq("long")]
            if len(alt):
                add("choices", arm, scheme, long, alt)
        order = g[g.stage.eq("option_order")]
        if len(order):
            add("order", arm, "reversed", long, order)
        repeated = g[g.stage.eq("repeatability")]
        replicates = sorted(repeated.replicate.unique())
        assert len(replicates) == 2
        add(
            "repeatability",
            arm,
            "second_run",
            repeated[repeated.replicate.eq(replicates[0])],
            repeated[repeated.replicate.eq(replicates[1])],
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result.p_raw, method="holm")[1]
    return result, pd.concat(cases, ignore_index=True)


def topic_tables(f, samples):
    tables, tests = [], []
    for cohort in [
        "aegis_single",
        "aegis_conversation",
        "wildguard_prompt",
        "wildguard_response",
        "xstest_prompt",
    ]:
        g = select(f, COHORTS[cohort]).copy()
        if cohort.startswith("aegis"):
            g["topic"] = g.item_id.map(
                lambda i: samples[i]["expected"]["aegis_categories"]
            )
            g = g.explode("topic").dropna(subset=["topic"])
            g = g[g.topic.ne("Needs Caution")]
        else:
            field = (
                "xstest_type" if cohort == "xstest_prompt" else "wildguard_subcategory"
            )
            g["topic"] = g.item_id.map(lambda i: samples[i]["metadata"][field])
        for gold, metric in [("unsafe", "miss"), ("safe", "false_alarm")]:
            h = g[g.source_gold.eq(gold)].copy()
            h["error"] = (
                ~(h.status.eq("ok") & h.label.eq("unsafe"))
                if metric == "miss"
                else h.status.eq("ok") & h.label.eq("unsafe")
            )
            for (topic, arm), x in h.groupby(["topic", "arm"]):
                assert x.item_id.is_unique
                k, n = int(x.error.sum()), len(x)
                low, high = wilson_interval(k, n)
                tables.append(
                    dict(
                        cohort=cohort,
                        topic=topic,
                        metric=metric,
                        arm=arm,
                        k=k,
                        n=n,
                        rate=k / n,
                        low=low,
                        high=high,
                    )
                )
            # Inference only for the four displayed topic panels. All other slices
            # are exported descriptively, without a hidden search for significance.
            if cohort == "aegis_single" and metric == "miss":
                for topic in TOPICS:
                    m = h[h.topic.eq(topic)].pivot(
                        index="item_id", columns="arm", values="error"
                    )
                    assert not m.isna().any().any()
                    for a, b in combinations(sorted(m.columns), 2):
                        ab = int((m[a] & ~m[b]).sum())
                        ba = int((~m[a] & m[b]).sum())
                        tests.append(
                            dict(
                                topic=topic,
                                arm_a=a,
                                arm_b=b,
                                n=len(m),
                                a_only_errors=ab,
                                b_only_errors=ba,
                                p_raw=binomtest(ab, ab + ba).pvalue if ab + ba else 1.0,
                            )
                        )
    tests = pd.DataFrame(tests)
    tests["p_holm"] = multipletests(tests.p_raw, method="holm")[1]
    return pd.DataFrame(tables), tests


def export(fig, key, out):
    title = json.loads((ROOT / "analysis/context_chart_titles.json").read_text())[
        key
    ].get("approved_title")
    if title:
        fig.text(
            0.025, 0.99, title, ha="left", va="top", fontsize=20, fontweight="bold"
        )
    for ext in ["png", "svg"]:
        savefig(fig, out / f"{key}.{ext}")
    plt.close(fig)


def plot_topics(rates, tests, out):
    data = rates[
        rates.cohort.eq("aegis_single")
        & rates.metric.eq("miss")
        & rates.topic.isin(TOPICS)
    ]
    arms = [a for a in NAMES if a in set(data.arm)]
    fig, axes = plt.subplots(1, 4, figsize=(21, 9), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.25, top=0.82, wspace=0.45)
    for ax, topic, title in zip(axes, TOPICS, TOPIC_NAMES):
        d = data[data.topic.eq(topic)].set_index("arm").loc[arms]
        v = 100 * d.rate.to_numpy()
        lo = 100 * d.low.to_numpy()
        hi = 100 * d.high.to_numpy()
        y = np.arange(len(arms))
        ax.barh(y, v, color=[COLORS[TYPES[a]] for a in arms], height=0.6, zorder=2)
        ax.errorbar(
            v,
            y,
            xerr=np.maximum(0, [v - lo, hi - v]),
            fmt="none",
            color="#18181B",
            capsize=3,
            zorder=3,
        )
        reference = d.rate.idxmin()
        for i, a in enumerate(arms):
            p = tests[
                tests.topic.eq(topic)
                & (
                    (tests.arm_a.eq(a) & tests.arm_b.eq(reference))
                    | (tests.arm_b.eq(a) & tests.arm_a.eq(reference))
                )
            ]
            mark = (
                " †"
                if len(p)
                and p.p_holm.iloc[0] < 0.05
                and d.loc[a, "rate"] > d.loc[reference, "rate"]
                else ""
            )
            ax.text(
                1.02,
                i,
                f"{v[i]:.1f}" + mark,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_yticks(y, [NAMES[a] for a in arms])
        ax.set_ylim(len(arms) - 0.5, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Not flagged per 100 source-unsafe requests", fontsize=9)
        ax.text(0, 1.08, title, transform=ax.transAxes, fontweight="bold", fontsize=13)
        ax.text(
            0,
            1.025,
            f"n = {d.n.iloc[0]} per model",
            transform=ax.transAxes,
            fontsize=10,
            color="#6B6B76",
        )
        fig.text(
            ax.get_position().x0,
            0.145,
            "Compared with " + NAMES[reference],
            fontsize=9,
            color="#6B6B76",
        )
        clean_axes(ax, despine_left=True, grid_axis="x")
    fig.legend(
        handles=[Patch(color=c, label=t) for t, c in COLORS.items()],
        loc="upper left",
        bbox_to_anchor=(0.16, 0.96),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.02,
        0.08,
        "Lines: 95% intervals. † Higher error than the named model after paired tests and correction across 220 comparisons.",
        fontsize=10,
        color="#6B6B76",
    )
    fig.text(
        0.02,
        0.04,
        "Aegis request-only subset. Source topic tags can overlap; these are binary harm errors within a topic, not category-identification scores.",
        fontsize=10,
        color="#6B6B76",
    )
    export(fig, "topic_failures", out)


def plot_counts(stats, kind, conditions, labels, key, out):
    data = stats[stats.kind.eq(kind)]
    arms = [a for a in NAMES if a in set(data.arm)]
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(16 if len(conditions) == 3 else 9, 7.8),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.24 if len(conditions) == 1 else 0.20,
        right=0.91,
        bottom=0.23,
        top=0.82,
        wspace=0.35,
    )
    for ax, condition, label in zip(axes[0], conditions, labels):
        d = data[data.condition.eq(condition)].set_index("arm").loc[arms]
        y = np.arange(len(arms))
        n = int(d.n.iloc[0])
        assert d.n.nunique() == 1
        for offset, col, color in [
            (-0.16, "gained", "#2C1FEB"),
            (0.16, "lost", "#956BDD"),
        ]:
            ax.barh(y + offset, d[col], height=0.29, color=color, zorder=2)
        for i, arm in enumerate(arms):
            ax.text(
                1.02,
                i,
                f"{d.loc[arm, 'gained']} / {d.loc[arm, 'lost']}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_yticks(y, [NAMES[a] for a in arms])
        ax.set_ylim(len(arms) - 0.5, -0.6)
        ax.set_xlim(0, max(5, int(data[["gained", "lost"]].max().max()) + 1))
        ax.set_xticks(range(int(ax.get_xlim()[1]) + 1))
        ax.set_xlabel(f"Conversations out of {n}", fontsize=11)
        ax.text(0, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
        clean_axes(ax, despine_left=True, grid_axis="x")
    fig.legend(
        handles=[
            Patch(color="#2C1FEB", label="Gained source agreement"),
            Patch(color="#956BDD", label="Lost source agreement"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.20, 0.95),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.025,
        0.105,
        "Exact counts on a fixed diagnostic set; the same conversations are reused across conditions.",
        fontsize=10,
        color="#6B6B76",
    )
    assert not stats.p_holm.lt(0.05).any(), (
        "Revise statistical summary before rendering significant diagnostic results"
    )
    fig.text(
        0.025,
        0.055,
        "No accuracy difference survives paired tests with correction. These small samples do not establish equivalence.",
        fontsize=10,
        color="#6B6B76",
    )
    export(fig, key, out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(SOURCE / "judgments.csv.gz", low_memory=False)
    d = pd.read_csv(SOURCE / "diagnostic_decisions.csv.gz", low_memory=False)
    samples = {
        r["id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/processed/source-full/000bf1bcb6dc/sample.jsonl").open(),
        )
    }
    stats, cases = diagnostic_tables(d)
    topics, tests = topic_tables(f, samples)
    for name, table in [
        ("diagnostic_changes", stats),
        ("diagnostic_cases", cases),
        ("topic_rates", topics),
        ("topic_paired_tests", tests),
    ]:
        table.to_csv(OUT / f"{name}.csv", index=False)
    # Expose actual offered choices instead of misleading scheme codes: s6 and
    # s13 contain 7 and 15 options once safe/caution/other are included.
    import yaml

    schemes = yaml.safe_load((ROOT / "data/label_map.yaml").read_text())["schemes"]
    from prompts.source_aligned.runtime import policy_for

    policy_rows = []
    for variant in ["short", "long", "neutral_long", "withheld"]:
        policy, definitions = policy_for(
            "aegis_v2_validation", "interaction_harm", variant, "s2"
        )
        policy_rows.append(
            dict(
                variant=variant,
                instruction_words=len(policy.split()),
                instruction_characters=len(policy),
                option_definition_words=sum(
                    len(v.split()) for v in definitions.values()
                ),
                instructions=policy,
            )
        )
    pd.DataFrame(policy_rows).to_csv(OUT / "policy_conditions.csv", index=False)
    use_braintrust_theme()
    plot_topics(topics, tests, OUT)
    archive = OUT / "archive"
    archive.mkdir(exist_ok=True)
    for key in ["policy_changes", "choice_changes"]:
        for ext in ["png", "svg"]:
            path = OUT / f"{key}.{ext}"
            if path.exists():
                path.replace(archive / path.name)
    # Repeat/order are better read as tiny exact-count tables than more charts.
    tables = stats[stats.kind.isin(["order", "repeatability"])][
        ["kind", "arm", "n", "changed", "gained", "lost"]
    ]
    tables.to_csv(OUT / "order_and_repeatability.csv", index=False)
    # Preserve all advice-tagged cases for inspection; selection is by source tag,
    # not by whether a particular model made an error.
    advice = select(f, COHORTS["aegis_single"])
    advice = advice[
        advice.source_gold.eq("unsafe")
        & advice.item_id.map(
            lambda i: (
                "Unauthorized Advice" in samples[i]["expected"]["aegis_categories"]
            )
        )
    ]
    audit = []
    for item, g in advice.groupby("item_id"):
        audit.append(
            dict(
                item_id=item,
                prompt=samples[item]["input"]["prompt"],
                source_categories="; ".join(
                    samples[item]["expected"]["aegis_categories"]
                ),
                source_gold="unsafe",
                models=len(g),
                models_flagging=int((g.status.eq("ok") & g.label.eq("unsafe")).sum()),
            )
        )
    pd.DataFrame(audit).to_csv(OUT / "unauthorized_advice_cases.csv", index=False)
    report = dict(
        topic_test_family=len(tests),
        diagnostic_test_family=len(stats),
        diagnostic_significant=int(stats.p_holm.lt(0.05).sum()),
        policy_unique_conversations=int(d[d.stage.eq("robustness")].item_id.nunique()),
        repeated_unique_conversations=int(
            d[d.stage.eq("repeatability")].item_id.nunique()
        ),
        actual_option_counts={k: len(v) for k, v in schemes.items()},
        narrative_charts=1,
        omitted_charts=["policy_changes", "choice_changes"],
        titles_rendered=[
            k
            for k, v in json.loads(
                (ROOT / "analysis/context_chart_titles.json").read_text()
            ).items()
            if v.get("approved_title") and v.get("placement") == "main"
        ],
    )
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_review(stats, topics, policy_rows, report)
    print(json.dumps(report, indent=2))


def write_review(stats, topics, policy_rows, report):
    title_path = ROOT / "analysis/context_chart_titles.json"
    titles = json.loads(title_path.read_text())
    active = {k: v for k, v in titles.items() if v.get("placement") == "main"}
    pd.DataFrame([dict(chart=k, **v) for k, v in active.items()]).to_csv(
        OUT / "titles_for_author.csv", index=False
    )
    pd.DataFrame(
        [dict(chart=k, **v) for k, v in titles.items() if k not in active]
    ).to_csv(OUT / "excluded_charts.csv", index=False)
    blocks = []
    for key, record in active.items():
        blocks.append(
            "<section><p><b>Decision:</b> " + html.escape(record["decision"]) + "</p>"
            "<p><b>Proposed title for your rewrite:</b> "
            + html.escape(record["proposed_title"])
            + "</p>"
            "<p>"
            + html.escape(record["finding"])
            + "</p><p>"
            + html.escape(record["implication"])
            + "</p>"
            '<img src="'
            + key
            + '.png" alt="'
            + html.escape(record["decision"])
            + '"></section>'
        )
    stability = stats[stats.kind.isin(["order", "repeatability"])].copy()
    stability["model"] = stability.arm.map(NAMES)
    stability["comparison"] = stability.kind.map(
        {"order": "Reversed option order", "repeatability": "Two repeated runs"}
    )
    stability = stability[["comparison", "model", "n", "changed", "both_wrong"]].rename(
        columns={
            "n": "Distinct conversations",
            "changed": "Changed decisions",
            "both_wrong": "Disagreed with source both times",
        }
    )
    conditions = pd.DataFrame(policy_rows)[
        ["variant", "instruction_words", "option_definition_words"]
    ]
    intro = (
        "<h1>Topics and policy design</h1><p><b>Data:</b> completed October 19 source_aligned_v1_20261019 rerun: 84,751 new core decisions and 1,832 new diagnostic decisions. "
        "The 4,500 unchanged XSTest decisions are reused. The topic chart and policy comparisons below use the new rerun.</p>"
        "<p>Use these results to decide what to test in your own policy and traffic. "
        "The topic chart belongs in the main analysis. Policy and answer-choice plots have been removed; the small diagnostic probes are summarized in text.</p>"
        "<p>The four illustrated Aegis topics were selected for their distinct policy boundaries, not as a representative average of all topics. "
        'All source topic slices are in <a href="topic_rates.csv">topic_rates.csv</a>. Topic tags can overlap and are not independently adjudicated category ground truth.</p>'
    )
    extra = (
        "<h2>Stable decisions can still be wrong</h2><p>Reversing binary answer order changed no decisions for the six tested interfaces on 12 conversations. "
        "Across two runs on 40 conversations, six of seven interfaces returned the same operational outcomes; Luna structured output changed two. "
        "Jev Choice disagreed with the source label on nine conversations in both runs. Repeatability does not establish correctness.</p>"
        + stability.to_html(index=False, border=0)
        + "<h2>What short and long actually mean</h2><p>Short retains the taxonomy. Detailed adds boundary guidance and examples. Neutral-long adds filler instead. "
        "Withheld removes the policy and meaningful option definitions together, so it is not a pure test of policy presence. Word counts below exclude the evaluated conversation.</p>"
        + conditions.to_html(index=False, border=0)
        + "<p>The policy and choice studies reuse 12 conversations (9 source-unsafe and 3 source-safe). They are not large enough to establish small improvements. "
        "Exact case counts remain in the supporting tables; none of the 52 paired diagnostic comparisons survives Holm correction. "
        "Repeated inference noise is not cleanly separated from prompt effects. Expanded category choices are collapsed to binary harm; caution and no verdict remain separate and do not count as source agreement. "
        "The source provides no independent category or caution gold for these comparisons.</p>"
        "<h2>The advice boundary needs review</h2><p>All 11 models fail to flag most of the 40 source-unsafe requests tagged Unauthorized Advice (75–97.5%). "
        "Examples include an IT architect role prompt (aegis-validation-000317), a teacher lesson-plan request (aegis-validation-000806), and a legal-advisor request after a car accident (aegis-validation-000555). "
        "These are materially different requests. Broad disagreement may reflect source labels, incomplete policy alignment, model behavior, or a combination. "
        "We cannot interpret the entire gap as missed real-world danger. The original labels remain unchanged.</p>"
        '<p><a href="unauthorized_advice_cases.csv">Inspect all 40 advice cases</a> · <a href="diagnostic_cases.csv">Inspect each changed decision</a> · '
        '<a href="topic_paired_tests.csv">Topic paired tests</a> · <a href="diagnostic_changes.csv">Diagnostic comparisons</a></p>'
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Topics and policy review</title><style>body{font:17px system-ui;color:#18181b;max-width:1500px;margin:40px auto;padding:0 28px}section{margin:48px 0}img{width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #eee}h2{margin-top:40px}</style>'
        + intro
        + "".join(blocks)
        + extra
    )
    (OUT / "review.html").write_text(page)
    review = SOURCE / "decisions/review.html"
    if review.exists():
        body = review.read_text().split("<!-- context-review -->")[0]
        record = active["topic_failures"]
        body += (
            '<!-- context-review --><section><p class="tag">MAIN · topic_failures</p><p><b>Title proposal for your rewrite:</b> '
            + html.escape(record["proposed_title"])
            + "</p><p>"
            + html.escape(record["finding"])
            + "</p><p>"
            + html.escape(record["implication"])
            + '</p><img src="context/topic_failures.png" alt="Source-topic disagreements"></section>'
        )
        body += '<section><p><b>Policy and answer-choice checks:</b> On 12 conversations, replacing detailed guidance with a short policy changed zero to two judgments per model. Expanding answer choices changed at most one per model and condition. Neither comparison established an accuracy improvement. These probes are too small to choose a policy length or answer format with confidence.</p><p><a href="context/review.html">Inspect policy conditions, repeatability, and the advice-label cases.</a></p></section>'
        review.write_text(body)
    import nbformat

    nb_path = ROOT / "analysis/source_policy_revision_exploration.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [c for c in nb.cells if not c.metadata.get("context_evidence")]
    new = [
        nbformat.v4.new_markdown_cell(
            "## Topics and policy design\n\nTopic slices use source tags and binary labels. Policy diagnostics use only 12 distinct conversations; repeated conditions do not increase that sample size."
        )
    ]
    for key, record in active.items():
        new += [
            nbformat.v4.new_markdown_cell(
                "**Decision:** "
                + record["decision"]
                + "\n\n**Proposed title:** "
                + record["proposed_title"]
                + "\n\n"
                + record["finding"]
                + "\n\n"
                + record["implication"]
            ),
            nbformat.v4.new_code_cell(
                f"display(Image(filename=str(out/'context/{key}.png')))"
            ),
        ]
    new += [
        nbformat.v4.new_markdown_cell(
            "**Policy and answer-choice checks:** On 12 conversations, replacing detailed guidance with a short policy changed zero to two judgments per model. Expanding answer choices changed at most one per model and condition. Neither comparison established an accuracy improvement. These probes are too small to choose a policy length or answer format with confidence.\n\nStable decisions can still disagree with the source labels. Detailed policies add guidance, while neutral-long adds filler; this does not isolate pure length from content. Exact counts and policy wording remain in the `context` CSVs."
        )
    ]
    for c in new:
        c.metadata["context_evidence"] = True
    nb.cells += new
    nbformat.write(nb, nb_path)


if __name__ == "__main__":
    main()
