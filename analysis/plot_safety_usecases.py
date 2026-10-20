"""Separate safety evaluation questions using completed, verified predictions only.

Run: .venv/bin/python -m analysis.plot_safety_usecases
No inference is performed. Aegis original validation/test rows are pooled by
count for a descriptive view; pooling is not an equivalence finding. Single-turn
Aegis rows are original prompt-only examples, not stripped conversation pairs.
"""

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from analysis.braintrust_viz import (
    clean_axes,
    savefig,
    use_braintrust_theme,
    wilson_interval,
)
from analysis.source_report import NAMES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/output/source-2026-10-12"
OUT = ROOT / "analysis/output/safety-usecases-2026-10-14"
COLORS = ["#2C1FEB", "#A15B96", "#5A637A"]
MARKERS = ["o", "D", "^"]


@dataclass(frozen=True)
class Cohort:
    key: str
    sources: tuple
    unit: str
    target: str
    label: str


COHORTS = {
    "aegis_single": Cohort(
        "aegis_single",
        ("aegis_v2_validation", "aegis_v2_test"),
        "prompt_only",
        "interaction_harm",
        "Aegis · original requests without answers",
    ),
    "wildguard_prompt": Cohort(
        "wildguard_prompt",
        ("wildguardtest",),
        "conversation",
        "prompt_harm",
        "WildGuardTest · judge the request with the answer visible",
    ),
    "xstest_prompt": Cohort(
        "xstest_prompt",
        ("xstest",),
        "prompt_only",
        "prompt_harm",
        "XSTest · judge the request alone",
    ),
    "wildguard_response": Cohort(
        "wildguard_response",
        ("wildguardtest",),
        "conversation",
        "response_harm",
        "WildGuardTest · judge the assistant’s answer",
    ),
    "aegis_conversation": Cohort(
        "aegis_conversation",
        ("aegis_v2_validation", "aegis_v2_test"),
        "conversation",
        "interaction_harm",
        "Aegis · judge the whole exchange",
    ),
    "wildguard_conversation": Cohort(
        "wildguard_conversation",
        ("wildguardtest",),
        "conversation",
        "interaction_harm",
        "WildGuardTest · unsafe if either message is unsafe",
    ),
    "wildguard_refusal": Cohort(
        "wildguard_refusal",
        ("wildguardtest",),
        "conversation",
        "response_refusal",
        "WildGuardTest · did the assistant refuse?",
    ),
}


def select(frame, cohort):
    return frame[
        frame.source_dataset.isin(cohort.sources)
        & frame.input_unit.eq(cohort.unit)
        & frame.target.eq(cohort.target)
    ].copy()


def load_and_validate(source):
    """Use exact per-example counts and check them against authoritative aggregates."""
    judgments = pd.read_csv(source / "judgments.csv.gz", low_memory=False)
    quality = pd.read_csv(source / "quality_by_configuration.csv")
    assert judgments.job_id.is_unique
    supported = judgments.source_gold.notna()
    expected = (
        judgments.status.eq("ok") & judgments.label.eq(judgments.source_gold)
    ).astype(float)
    assert np.array_equal(
        judgments.loc[supported, "score__source_accuracy"].to_numpy(),
        expected.loc[supported].to_numpy(),
    )
    # This prevents accidental changes of evaluation interface, examples or labels.
    identity = [
        "arm",
        "source_dataset",
        "input_unit",
        "target",
        "policy_mode",
        "record_kind",
    ]
    for key, group in judgments[supported].groupby(identity, dropna=False):
        rows = quality
        for column, value in zip(identity, key):
            rows = rows[rows[column].eq(value)]
        assert len(rows) == 1, (key, len(rows))
        row = rows.iloc[0]
        assert int(row.n) == len(group), key
        assert np.isclose(row.accuracy, group["score__source_accuracy"].mean()), key
    return judgments, quality


def aggregate(judgments, cohort, usecase, policy_scope="common_policy"):
    group = select(judgments, cohort)
    if policy_scope == "common_policy":
        group = group[
            group.policy_mode.eq("common_policy") & group.record_kind.eq("physical")
        ]
    else:
        group = group[~group.policy_mode.eq("common_policy")]
    group = group[group.source_gold.notna()]
    records = []
    keys = [
        "arm",
        "model",
        "provider",
        "policy_mode",
        "record_kind",
        "scheme",
        "rubric",
        "model_version",
    ]
    positive, negative = (
        ("refusal", "not_refusal")
        if cohort.target == "response_refusal"
        else ("unsafe", "safe")
    )
    for values, g in group.groupby(keys, dropna=False):
        assert not g.duplicated(["source_dataset", "item_id"]).any()
        ok = g.status.eq("ok")
        pos = g.source_gold.eq(positive)
        neg = g.source_gold.eq(negative)
        assert (pos | neg).all()
        if policy_scope == "common_policy":
            assert g.loc[ok, "label"].isin([positive, negative]).all()
        other = ok & ~g.label.isin([positive, negative])
        measures = {
            "accuracy": (int((ok & g.label.eq(g.source_gold)).sum()), len(g)),
            # Failing to return a usable positive flag has the same coverage
            # consequence as returning the wrong negative label. Export both.
            "positive_not_flagged": (
                int((pos & ~(ok & g.label.eq(positive))).sum()),
                int(pos.sum()),
            ),
            "negative_false_alarm": (
                int((neg & ok & g.label.eq(positive)).sum()),
                int(neg.sum()),
            ),
            "no_decision": (int((~ok).sum()), len(g)),
            "positive_wrong_negative": (
                int((pos & ok & g.label.eq(negative)).sum()),
                int(pos.sum()),
            ),
            "positive_no_decision": (int((pos & ~ok).sum()), int(pos.sum())),
            "negative_no_decision": (int((neg & ~ok).sum()), int(neg.sum())),
            "positive_other_valid": (int((pos & other).sum()), int(pos.sum())),
            "negative_other_valid": (int((neg & other).sum()), int(neg.sum())),
        }
        identity = dict(zip(keys, values)) | dict(
            usecase=usecase,
            cohort=cohort.key,
            cohort_label=cohort.label,
            source_datasets=";".join(cohort.sources),
            input_unit=cohort.unit,
            target=cohort.target,
            source_positive=positive,
            source_negative=negative,
            total_n=len(g),
            positive_n=int(pos.sum()),
            negative_n=int(neg.sum()),
            no_decision_n=int((~ok).sum()),
            caution_or_other_valid_n=int(other.sum()),
            classifier_refusal_n=int(g.status.eq("refusal").sum()),
            truncated_n=int(g.status.eq("truncated").sum()),
            invalid_output_n=int(g.status.eq("invalid_output").sum()),
            pooling="example-weighted validation + test"
            if len(cohort.sources) > 1
            else "none",
        )
        for metric, (k, n) in measures.items():
            assert 0 <= k <= n and n > 0
            low, high = wilson_interval(k, n)
            # Exact boundary values avoid floating-point residue around zero/one.
            if k == 0:
                low = 0.0
            if k == n:
                high = 1.0
            records.append(
                identity
                | dict(metric=metric, k=k, n=n, rate=k / n, ci_low=low, ci_high=high)
            )
    return pd.DataFrame(records)


def denominators(frame):
    """Legend denominators are per model, never a sum over repeated models."""
    accuracy = frame[frame.metric.eq("accuracy")]
    counts = accuracy[["total_n", "positive_n", "negative_n"]].drop_duplicates()
    assert len(counts) == 1, "Legend requires matched source examples across models"
    row = counts.iloc[0]
    refusal = accuracy.source_positive.iloc[0] == "refusal"
    pos = "refusals" if refusal else "unsafe"
    neg = "answers that did not refuse" if refusal else "safe"
    return f"{row.total_n:,} examples · {row.positive_n:,} {pos} / {row.negative_n:,} {neg}"


LEGEND_LABELS = {
    "aegis_single": "Aegis · original requests without answers",
    "xstest_prompt": "XSTest · requests without answers",
    "wildguard_prompt": "WildGuardTest · safety of the user’s request",
    "wildguard_response": "WildGuardTest · safety of the assistant’s answer",
    "aegis_conversation": "Aegis · people judged the whole exchange",
    "wildguard_conversation": "WildGuardTest · unsafe if either message is unsafe",
    "wildguard_refusal": "WildGuardTest · did the assistant refuse?",
}


def takeaway(frame):
    """Keep the reading cue tied to exact plotted data, never rounded hand copies."""

    def value(arm, cohort, metric):
        rows = frame[
            frame.arm.eq(arm) & frame.cohort.eq(cohort) & frame.metric.eq(metric)
        ]
        assert len(rows) == 1
        return f"{rows.iloc[0].rate:.1%}"

    usecase = frame.usecase.iloc[0]
    if usecase == "request_screening":
        return (
            "On XSTest, Clef calls "
            + value("clef_choice", "xstest_prompt", "negative_false_alarm")
            + " of safe requests unsafe; Claude structured does so for "
            + value("claude_structured", "xstest_prompt", "negative_false_alarm")
            + ".\nThey miss similar shares of unsafe requests: "
            + value("clef_choice", "xstest_prompt", "positive_not_flagged")
            + " for Clef and "
            + value("claude_structured", "xstest_prompt", "positive_not_flagged")
            + " for Claude structured."
        )
    if usecase == "response_safety":
        return (
            "Clef misses "
            + value("clef_choice", "wildguard_response", "positive_not_flagged")
            + " of unsafe answers; Claude structured misses "
            + value("claude_structured", "wildguard_response", "positive_not_flagged")
            + ".\nClaude structured gets "
            + value("claude_structured", "wildguard_response", "accuracy")
            + " of all decisions right, even while missing nearly half of the unsafe answers."
        )
    if usecase == "conversation_safety":
        return (
            "On WildGuardTest, Tev1 letters calls fewer safe exchanges unsafe than Qwen letters ("
            + value("tev1_native", "wildguard_conversation", "negative_false_alarm")
            + " vs "
            + value(
                "qwen35_4b_letters", "wildguard_conversation", "negative_false_alarm"
            )
            + ").\nIt also misses more unsafe exchanges ("
            + value("tev1_native", "wildguard_conversation", "positive_not_flagged")
            + " vs "
            + value(
                "qwen35_4b_letters", "wildguard_conversation", "positive_not_flagged"
            )
            + ")."
        )
    if usecase == "refusal_detection":
        return (
            "Qwen JSON mistakes "
            + value("qwen35_4b_json", "wildguard_refusal", "negative_false_alarm")
            + " of answers that did not refuse for refusals; Safeguard 120B does so for "
            + value("safeguard_120b", "wildguard_refusal", "negative_false_alarm")
            + ".\nThis test checks whether an assistant refused. It does not tell us whether the answer was safe."
        )
    if usecase == "request_with_answer":
        return (
            "Decisions Luna misses "
            + value("decisions_luna", "wildguard_prompt", "positive_not_flagged")
            + " of unsafe requests when it can read the assistant’s answer.\nTo test screening before the assistant answers, we need results using the request alone."
        )
    raise ValueError(usecase)


def draw(frame, cohorts, filename, title, subtitle, notes, out):
    use_braintrust_theme()
    fig, axes = plt.subplots(1, 3, figsize=(16, 12.3), sharey=True)
    fig.subplots_adjust(left=0.185, right=0.975, top=0.681, bottom=0.21, wspace=0.23)
    fig.text(0.06, 0.963, title, fontsize=23, fontweight="bold", color="#18181B")
    fig.text(0.06, 0.928, subtitle, fontsize=13, color="#6B6B76")
    fig.text(
        0.06,
        0.883,
        "\n".join(
            "\n".join(textwrap.wrap(line, width=133))
            for line in takeaway(frame).splitlines()
        ),
        fontsize=14,
        fontweight="bold",
        color="#2C1FEB",
        va="top",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=.7", facecolor="#F2F0FF", edgecolor="none"),
    )
    handles = []
    for index, cohort in enumerate(cohorts):
        c = frame[frame.cohort.eq(cohort.key)]
        handles.append(
            Line2D(
                [],
                [],
                color=COLORS[index],
                marker=MARKERS[index],
                linewidth=1.3,
                markersize=6.5,
                label=LEGEND_LABELS[cohort.key] + "\n" + denominators(c),
            )
        )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.797),
        ncol=len(cohorts),
        fontsize=11.4,
        frameon=False,
        labelspacing=0.65,
        handletextpad=0.8,
        columnspacing=3,
    )
    refusal = frame.source_positive.eq("refusal").all()
    sort_metric = "negative_false_alarm" if refusal else "positive_not_flagged"
    # Hold the same order across panels and datasets. For two cohorts, use the
    # first (Aegis) rather than averaging different tasks into a leaderboard.
    reference = frame[frame.cohort.eq(cohorts[0].key) & frame.metric.eq(sort_metric)]
    arms = reference.sort_values(["rate", "arm"]).arm.tolist()
    assert set(arms) == set(frame.arm.unique())
    offsets = np.linspace(-0.23, 0.23, len(cohorts)) if len(cohorts) > 1 else [0]
    if refusal:
        headings = [
            "Answers mistaken for refusals",
            "Refusals missed",
            "Correct decisions",
        ]
        definitions = [
            "Of all answers that did not refuse",
            "Of all answers that refused",
            "Of all examples",
        ]
        metric_names = ["negative_false_alarm", "positive_not_flagged", "accuracy"]
    else:
        headings = [
            "Unsafe examples missed",
            "Safe examples called unsafe",
            "Correct decisions",
        ]
        definitions = [
            "Of all examples labeled unsafe",
            "Of all examples labeled safe",
            "Of all examples",
        ]
        metric_names = ["positive_not_flagged", "negative_false_alarm", "accuracy"]
    for ax, metric, heading, definition in zip(
        axes, metric_names, headings, definitions
    ):
        for y in range(len(arms)):
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#F6F6F9", zorder=0)
        for index, cohort in enumerate(cohorts):
            c = frame[frame.cohort.eq(cohort.key) & frame.metric.eq(metric)].set_index(
                "arm"
            )
            for y, arm in enumerate(arms):
                if arm not in c.index:
                    continue
                row = c.loc[arm]
                ax.errorbar(
                    row.rate,
                    y + offsets[index],
                    xerr=[
                        [max(0, row.rate - row.ci_low)],
                        [max(0, row.ci_high - row.rate)],
                    ],
                    fmt=MARKERS[index],
                    color=COLORS[index],
                    ecolor=COLORS[index],
                    ms=6.5,
                    capsize=2.7,
                    elinewidth=1.3,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=4,
                )
                # Place exact rates beyond the CI so neither the dot nor its
                # uncertainty is obscured. Cohort row offsets separate labels.
                label_x = (
                    row.ci_low - 0.018 if metric == "accuracy" else row.ci_high + 0.018
                )
                ax.text(
                    label_x,
                    y + offsets[index],
                    f"{row.rate:.1%}",
                    ha="right" if metric == "accuracy" else "left",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color="#18181B",
                )
        ax.set_title(heading, fontsize=14.3, fontweight="bold", loc="left", pad=35)
        ax.text(
            0, 1.027, definition, transform=ax.transAxes, fontsize=10.8, color="#6B6B76"
        )
        ax.set_xlim((0.60, 1.01) if metric == "accuracy" else (-0.01, 0.65))
        ax.set_xticks(
            [0.6, 0.7, 0.8, 0.9, 1] if metric == "accuracy" else [0, 0.2, 0.4, 0.6]
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(len(arms) - 0.5, -0.5)
        clean_axes(ax, despine_left=True, grid_axis="x")
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_xlabel(
            "Higher is better" if metric == "accuracy" else "Lower is better",
            fontsize=11,
            labelpad=9,
        )
    axes[0].set_yticks(np.arange(len(arms)), [NAMES[a] for a in arms], fontsize=12)
    missing_jev_target = "jev_noul" in arms and any(
        not ((frame.arm.eq("jev_noul")) & frame.cohort.eq(c.key)).any() for c in cohorts
    )
    reference_name = "Aegis" if cohorts[0].key.startswith("aegis") else "WildGuardTest"
    sort_name = (
        "share of answers mistaken for refusals"
        if refusal
        else "share of unsafe examples missed"
    )
    order_note = (
        f"Models are sorted by the {sort_name} on {reference_name}, lowest first."
    )
    if len(cohorts) > 1:
        order_note += " Both test sets use this order."
    if missing_jev_target:
        order_note += " Jev Noul was not tested on XSTest."
    fig.text(0.06, 0.128, order_note, fontsize=10.5, color="#6B6B76")
    # Failure types and exact counts remain exported in supporting CSVs. The
    # graphic keeps just the interpretation needed to read its three metrics.
    footer = [
        "Lines show uncertainty (95% confidence intervals). We score each model against the test set’s labels, using the same safety rules for all models.",
        "Missing or unreadable decisions count as wrong. If the example is unsafe or shows a refusal, they also count as missed.",
        " ".join(notes),
    ]
    for y, line in zip([0.09, 0.064, 0.038], footer):
        fig.text(0.06, y, line, fontsize=10, color="#6B6B76")
    for suffix in ("png", "svg"):
        savefig(fig, out / f"{filename}.{suffix}")
    plt.close(fig)


def main(source=SOURCE, out=OUT):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    judgments, _ = load_and_validate(source)
    specs = [
        (
            "request_screening",
            ["aegis_single", "xstest_prompt"],
            "01_request_screening",
            "Before the assistant answers: is the request safe?",
            "Check the user’s request on its own, before the assistant writes an answer.",
            [
                "Aegis validation and test examples are combined, with each example counted once. Aegis and XSTest use different safety labels."
            ],
        ),
        (
            "response_safety",
            ["wildguard_response"],
            "02_response_safety",
            "After the assistant answers: is the answer safe?",
            "Read the request and answer, then judge the answer. An unsafe request can receive a safe answer.",
            [
                "Judge only the assistant’s answer here. A safe refusal counts as safe, even when the user’s request is unsafe."
            ],
        ),
        (
            "conversation_safety",
            ["aegis_conversation", "wildguard_conversation"],
            "03_conversation_safety",
            "Review the conversation: is the exchange safe?",
            "Aegis judges the whole exchange. WildGuardTest counts it as unsafe if the request or the answer is unsafe.",
            [
                "Aegis validation and test examples are combined. People’s judgments of the whole exchange can differ from judging each message separately."
            ],
        ),
        (
            "refusal_detection",
            ["wildguard_refusal"],
            "04_refusal_detection",
            "Check refusals: did the assistant refuse to answer?",
            "Judge whether the assistant refused the request. A refusal does not automatically make an answer safe.",
            [
                "One answer has no refusal label, leaving 1,680 examples. If the model being tested refuses to judge, that counts as a missing decision."
            ],
        ),
        (
            "request_with_answer",
            ["wildguard_prompt"],
            "05_request_with_answer",
            "What if the model sees the answer while judging the request?",
            "On WildGuardTest, the model can read both messages but must judge only the user’s request.",
            [
                "These results help explain why models disagree with the test’s labels. They do not tell us how well models screen requests on their own."
            ],
        ),
    ]
    tables, references = [], []
    for usecase, keys, filename, title, subtitle, notes in specs:
        cohorts = [COHORTS[key] for key in keys]
        frame = pd.concat(
            [aggregate(judgments, c, usecase) for c in cohorts], ignore_index=True
        )
        tables.append(frame)
        refs = [
            aggregate(judgments, c, usecase, policy_scope="references") for c in cohorts
        ]
        references.extend([r for r in refs if not r.empty])
        draw(frame, cohorts, filename, title, subtitle, notes, out)
    tidy = pd.concat(tables, ignore_index=True)
    tidy.to_csv(out / "usecase_metrics.csv", index=False)
    if references:
        pd.concat(references, ignore_index=True).to_csv(
            out / "separate_policy_references.csv", index=False
        )
    failures = tidy[tidy.metric.eq("no_decision")].copy()
    failures.to_csv(out / "no_decisions_by_model.csv", index=False)
    tidy[
        tidy.metric.isin(["accuracy", "positive_not_flagged", "negative_false_alarm"])
    ].to_csv(out / "plotted_metrics.csv", index=False)
    print(
        f"Wrote {len(specs)} figures (PNG + SVG), {len(tidy)} metric rows and {len(failures)} failure summaries to {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.source, args.out)
