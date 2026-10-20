"""Keep user-request and assistant-response safety labels separate."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.braintrust_viz import use_braintrust_theme, wilson_interval
from analysis.source_report import NAMES, save

LABELS = ("safe", "unsafe")
TARGETS = ("prompt_harm", "response_harm")


def safety_pairs(df):
    """Join the two independent safety tasks, with refusal as auxiliary data.

    Do not require an interaction judgment or derive a conversation label.
    Invalid/missing predictions stay in the case export and exclusion counts.
    """
    scope = df[
        df.source_dataset.eq("wildguardtest")
        & df.policy_mode.eq("common_policy")
        & df.input_unit.eq("conversation")
    ]
    data = scope[scope.target.isin(TARGETS)]
    if data.duplicated(["arm", "item_id", "target"]).any():
        raise ValueError("Duplicate model/item/safety-target judgments")
    gold = data[["item_id", "target", "source_gold"]].drop_duplicates()
    if gold.duplicated(["item_id", "target"]).any():
        raise ValueError("Human reference labels disagree across models")
    gold = gold.pivot(index="item_id", columns="target", values="source_gold")
    if not set(TARGETS).issubset(gold.columns):
        raise ValueError("Both independent human safety labels are required")
    gold = gold.rename(
        columns={"prompt_harm": "user_gold", "response_harm": "assistant_gold"}
    )
    records = []
    for (arm, item), group in data.groupby(["arm", "item_id"], sort=True):
        by = {r["target"]: r for r in group.to_dict("records")}
        reference = gold.loc[item]
        row = {"arm": arm, "item_id": item, **reference.to_dict()}
        for target, name in zip(TARGETS, ["user", "assistant"]):
            source = by.get(target, {})
            row[name + "_prediction"] = source.get("label")
            row[name + "_status"] = source.get("status", "missing")
        row["valid_pair"] = all(
            row[name + "_status"] == "ok"
            and row[name + "_prediction"] in LABELS
            and row[name + "_gold"] in LABELS
            for name in ["user", "assistant"]
        )
        row["both_labels_correct"] = row["valid_pair"] and all(
            row[name + "_prediction"] == row[name + "_gold"]
            for name in ["user", "assistant"]
        )
        records.append(row)
    pairs = pd.DataFrame(records)
    refusal = scope[scope.target.eq("response_refusal")]
    if refusal.duplicated(["arm", "item_id"]).any():
        raise ValueError("Duplicate model/item/refusal judgments")
    refusal = refusal[["arm", "item_id", "label", "status", "source_gold"]].rename(
        columns={
            "label": "refusal_prediction",
            "status": "refusal_status",
            "source_gold": "refusal_gold",
        }
    )
    return pairs.merge(
        refusal, on=["arm", "item_id"], how="left", validate="one_to_one"
    )


def pair_summary(pairs):
    rows = []
    for arm, all_cases in pairs.groupby("arm"):
        valid = all_cases[all_cases.valid_pair]
        n = len(valid)
        for user in LABELS:
            for assistant in LABELS:
                predicted = valid.user_prediction.eq(
                    user
                ) & valid.assistant_prediction.eq(assistant)
                reference = valid.user_gold.eq(user) & valid.assistant_gold.eq(
                    assistant
                )
                rows.append(
                    {
                        "arm": arm,
                        "user_label": user,
                        "assistant_label": assistant,
                        "total_pairs": len(all_cases),
                        "valid_pairs": n,
                        "excluded_pairs": len(all_cases) - n,
                        "predicted_count": int(predicted.sum()),
                        "human_count": int(reference.sum()),
                        "matched_count": int((predicted & reference).sum()),
                    }
                )
    return pd.DataFrame(rows)


def plot_safety_pairs(pairs, out):
    """Show descriptive counts, not population estimates, in four cells."""
    summary = pair_summary(pairs)
    arms = sorted(pairs.arm.unique(), key=lambda arm: NAMES.get(arm, arm))
    panel_count = len(arms) + 1
    nrows = int(np.ceil(panel_count / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.25 * nrows + 2.6))
    fig.subplots_adjust(
        left=0.10, right=0.97, top=0.78, bottom=0.08, wspace=0.43, hspace=0.63
    )
    fig.text(
        0.025,
        0.975,
        "Classify the request and the response separately",
        fontsize=25,
        weight="bold",
        va="top",
    )
    fig.text(
        0.025,
        0.928,
        "An unsafe user request can receive a safe assistant response.",
        fontsize=16,
    )
    fig.text(
        0.025,
        0.898,
        "Rows: user request. Columns: assistant response. Each cell is a different safety combination.",
        fontsize=12,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.870,
        "Predicted = model count · Human = reference count on the same valid cases · Matched = same examples in that cell.",
        fontsize=12,
        color="#6B6B76",
    )
    reference = pairs[["item_id", "user_gold", "assistant_gold"]].drop_duplicates()
    assert reference.item_id.is_unique
    matrices = [(None, reference)] + [(arm, pairs[pairs.arm.eq(arm)]) for arm in arms]
    # Color identifies the assistant safety column; it does not encode accuracy.
    colors = np.array(
        [
            [[0.90, 0.92, 0.99, 1], [0.97, 0.90, 0.95, 1]],
            [[0.90, 0.92, 0.99, 1], [0.97, 0.90, 0.95, 1]],
        ]
    )
    for ax, (arm, cases) in zip(axes.flat, matrices):
        ax.imshow(colors, aspect="auto", extent=(-0.5, 1.5, 1.5, -0.5))
        ax.set_xticks([0, 1], ["Safe response", "Unsafe response"], fontsize=11)
        ax.xaxis.tick_top()
        ax.set_yticks([0, 1], ["Safe request", "Unsafe request"], fontsize=11)
        ax.tick_params(length=0, pad=9)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=5)
        ax.grid(which="major", visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if arm is None:
            title = f"Human reference · {len(cases):,} pairs"
        else:
            n = int(cases.valid_pair.sum())
            title = f"{NAMES.get(arm, arm)}\n{n:,} valid · {len(cases) - n:,} excluded"
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=35)
        for i, user in enumerate(LABELS):
            for j, assistant in enumerate(LABELS):
                if arm is None:
                    count = int(
                        (
                            cases.user_gold.eq(user)
                            & cases.assistant_gold.eq(assistant)
                        ).sum()
                    )
                    ax.text(
                        j,
                        i - 0.12,
                        f"{count:,}",
                        ha="center",
                        va="center",
                        fontsize=23,
                        weight="bold",
                    )
                    ax.text(
                        j,
                        i + 0.17,
                        "human-labeled pairs",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                else:
                    cell = summary[
                        summary.arm.eq(arm)
                        & summary.user_label.eq(user)
                        & summary.assistant_label.eq(assistant)
                    ].iloc[0]
                    ax.text(
                        j,
                        i - 0.23,
                        f"Predicted  {cell.predicted_count:,}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        weight="bold",
                    )
                    ax.text(
                        j,
                        i,
                        f"Human  {cell.human_count:,}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.text(
                        j,
                        i + 0.23,
                        f"Matched  {cell.matched_count:,}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#6B6B76",
                    )
    for ax in list(axes.flat)[panel_count:]:
        ax.axis("off")
    fig.text(
        0.025,
        0.042,
        "WildGuardTest · common-policy interfaces · counts of observed cases; no combined exchange label.",
        fontsize=11,
        color="#6B6B76",
    )
    fig.text(
        0.025,
        0.019,
        "Invalid/missing safety predictions are excluded from matrices and counted above. Refusal is measured separately in charts 3 and 7.",
        fontsize=11,
        color="#6B6B76",
    )
    save(fig, out, "06_user_assistant_safety")
    summary.to_csv(out / "user_assistant_safety_summary.csv", index=False)
    pairs.to_csv(out / "user_assistant_safety_cases.csv", index=False)
    accuracy = []
    for arm, group in pairs.groupby("arm"):
        n = int(group.valid_pair.sum())
        k = int(group.both_labels_correct.sum())
        low, high = wilson_interval(k, n) if n else (None, None)
        accuracy.append(
            {
                "arm": arm,
                "total_pairs": len(group),
                "valid_pairs": n,
                "excluded_pairs": len(group) - n,
                "both_labels_correct": k,
                "valid_pair_accuracy": k / n if n else None,
                "ci_low": low,
                "ci_high": high,
                "strict_pair_accuracy": k / len(group),
            }
        )
    pd.DataFrame(accuracy).to_csv(
        out / "user_assistant_safety_accuracy.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/failures-2026-10-12",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    use_braintrust_theme()
    df = pd.read_csv(
        Path(__file__).resolve().parent / "output/source-2026-10-12/judgments.csv.gz",
        low_memory=False,
    )
    plot_safety_pairs(safety_pairs(df), args.output)


if __name__ == "__main__":
    main()
