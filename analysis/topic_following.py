"""Summarize policy-following results without merging them with safety scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from analysis.braintrust_viz import header, make_figure, rate_cmap, use_braintrust_theme
from analysis.source_report import NAMES, heatmap_ink, markdown_table, save
from data.jsonl import read_jsonl
from data.topic_following import OUT


def summarize(results, jobs):
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate manifest job IDs")
    by_job = {r["job_id"]: r for r in results}
    if len(by_job) != len(results):
        raise ValueError("Duplicate result job IDs")
    if set(by_job) - {j["job_id"] for j in jobs}:
        raise ValueError("Results contain jobs outside the manifest")
    rows = []
    for job in jobs:
        r = by_job.get(job["job_id"], {})
        if r and (
            r["expected"] != job["expected"]
            or r["request_sha256"] != job["request_sha256"]
        ):
            raise ValueError("Result/manifest mismatch")
        valid = r.get("status") == "ok" and r.get("label") in {"engage", "redirect"}
        rows.append(
            {
                "job_id": job["job_id"],
                "arm": job["arm"],
                "case_id": job["case_id"],
                **job["metadata"],
                "expected": job["expected"],
                "label": r.get("label"),
                "model": r.get("model"),
                "model_version": r.get("model_version"),
                "status": r.get("status", "missing"),
                "valid": valid,
                "correct": valid and r.get("label") == job["expected"],
            }
        )
    data = pd.DataFrame(rows)
    cohorts = []
    pairs = []
    for (arm, cohort), g in data.groupby(["arm", "cohort"]):
        n = len(g)
        k = int(g.correct.sum())
        cohorts.append(
            {
                "arm": arm,
                "cohort": cohort,
                "n": n,
                "correct": k,
                "strict_accuracy": k / n,
                "invalid_or_missing": int((~g.valid).sum()),
                "engage_n": int(g.expected.eq("engage").sum()),
                "redirect_n": int(g.expected.eq("redirect").sum()),
            }
        )
    for (arm, pair_id), g in data[data.cohort.eq("paired_policy")].groupby(
        ["arm", "pair_id"]
    ):
        if len(g) != 2 or set(g.policy_variant) != {"allowed", "excluded"}:
            raise ValueError("Incomplete policy pair")
        allowed = g[g.policy_variant.eq("allowed")].iloc[0]
        excluded = g[g.policy_variant.eq("excluded")].iloc[0]
        requests = [
            next(j["request"] for j in jobs if j["job_id"] == r["job_id"])
            for r in g.to_dict("records")
        ]
        if (
            requests[0]["text"] != requests[1]["text"]
            or requests[0]["option_order"] != requests[1]["option_order"]
        ):
            raise ValueError("Paired history, current turn, or offered order changed")
        valid = bool(g.valid.all())
        correct = bool(g.correct.all())
        pairs.append(
            {
                "arm": arm,
                "pair_id": pair_id,
                "domain": allowed.domain,
                "allowed_label": allowed.label,
                "excluded_label": excluded.label,
                "allowed_status": allowed.status,
                "excluded_status": excluded.status,
                "both_valid": valid,
                "decision_changed": valid and allowed.label != excluded.label,
                "both_correct": correct,
            }
        )
    return data, pd.DataFrame(cohorts), pd.DataFrame(pairs)


def plot_grid(pairs, output):
    arms = sorted(pairs.arm.unique())
    nrows = int(np.ceil(len(arms) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.5 * nrows + 2.3), squeeze=False)
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.77, bottom=0.07, wspace=0.45, hspace=0.50
    )
    fig.text(
        0.025,
        0.975,
        "Does the decision change when the policy changes?",
        fontsize=24,
        weight="bold",
        va="top",
    )
    fig.text(
        0.025,
        0.905,
        "Same request, history, and choice order. Only the policy changes.",
        fontsize=15,
    )
    fig.text(
        0.025,
        0.857,
        "Expected: engage when permitted, redirect when excluded. * = wrong or failed. These are benign topic probes.",
        fontsize=12,
        color="#6B6B76",
    )
    for ax, arm in zip(axes.flat, arms):
        group = pairs[pairs.arm.eq(arm)].sort_values("pair_id")
        values = np.array(
            [
                [
                    0
                    if r[k] == "engage" and r[s] == "ok"
                    else 1
                    if r[k] == "redirect" and r[s] == "ok"
                    else 2
                    for k, s in [
                        ("allowed_label", "allowed_status"),
                        ("excluded_label", "excluded_status"),
                    ]
                ]
                for r in group.to_dict("records")
            ]
        )
        ax.imshow(
            values,
            cmap=ListedColormap(["#D4E0FB", "#E6A4D3", "#E8E8E8"]),
            vmin=0,
            vmax=2,
            aspect="auto",
        )
        for i, row in enumerate(group.to_dict("records")):
            for j, (key, status, expected) in enumerate(
                [
                    ("allowed_label", "allowed_status", "engage"),
                    ("excluded_label", "excluded_status", "redirect"),
                ]
            ):
                label = row[key] if row[status] == "ok" else row[status]
                ax.text(
                    j,
                    i,
                    str(label) + (" *" if label != expected else ""),
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
        ax.set_title(
            f"{NAMES.get(arm, arm)} · {int(group.both_correct.sum())}/{len(group)} pairs correct",
            loc="left",
            fontsize=13,
            weight="bold",
            pad=12,
        )
        ax.set_yticks(range(len(group)), group.domain, fontsize=11)
        ax.set_xticks([0, 1], ["Permitted\nengage", "Excluded\nredirect"], fontsize=11)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks(np.arange(0.5, len(group) - 0.5), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(arms) :]:
        ax.axis("off")
    save(fig, output, "policy_switches")


def report(directory, output):
    directories = [directory] if isinstance(directory, Path) else directory
    jobs = []
    results = []
    archived_attempts = []
    for folder in directories:
        jobs.extend(read_jsonl(folder / "manifest.jsonl"))
        if (folder / "results.jsonl").exists():
            results.extend(read_jsonl(folder / "results.jsonl"))
        if (folder / "api_failure_attempts.jsonl").exists():
            archived_attempts.extend(read_jsonl(folder / "api_failure_attempts.jsonl"))
    requests_by_case = {}
    for job in jobs:
        prior = requests_by_case.setdefault(job["case_id"], job["request_sha256"])
        if prior != job["request_sha256"]:
            raise ValueError("Runs use different requests for the same case ID")
    data, cohorts, pairs = summarize(results, jobs)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("cases", data),
        ("cohort_metrics", cohorts),
        ("policy_pairs", pairs),
    ]:
        frame.to_csv(output / (name + ".csv"), index=False)
    label_metrics = []
    domain_metrics = []
    for (arm, cohort, label), g in data.groupby(["arm", "cohort", "expected"]):
        label_metrics.append(
            {
                "arm": arm,
                "cohort": cohort,
                "reference_label": label,
                "n": len(g),
                "matched": int(g.correct.sum()),
                "agreement": float(g.correct.mean()),
                "invalid_or_missing": int((~g.valid).sum()),
            }
        )
    for (arm, cohort, domain), g in data.groupby(["arm", "cohort", "domain"]):
        domain_metrics.append(
            {
                "arm": arm,
                "cohort": cohort,
                "domain": domain,
                "n": len(g),
                "matched": int(g.correct.sum()),
                "agreement": float(g.correct.mean()),
                "invalid_or_missing": int((~g.valid).sum()),
            }
        )
    pd.DataFrame(label_metrics).to_csv(output / "label_metrics.csv", index=False)
    pd.DataFrame(domain_metrics).to_csv(output / "domain_metrics.csv", index=False)
    if archived_attempts:
        pd.DataFrame(
            [
                {
                    k: r.get(k)
                    for k in ["job_id", "arm", "case_id", "status", "latency_ms"]
                }
                for r in archived_attempts
            ]
        ).to_csv(output / "retry_history.csv", index=False)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    disagreements = [
        {
            "case_id": r.case_id,
            "arm": r.arm,
            "provisional_reference": r.expected,
            "predicted_label": r.label,
            "status": r.status,
            "request": jobs_by_id[r.job_id]["request"],
        }
        for r in data[
            data.cohort.isin(["source_topic_following", "synthetic_topic_following"])
            & ~data.correct
        ].itertuples()
    ]
    from data.topic_following import write_jsonl

    write_jsonl(output / "source_disagreements.jsonl", disagreements)
    use_braintrust_theme()
    synthetic = pd.DataFrame(
        [r for r in domain_metrics if r["cohort"] == "synthetic_topic_following"]
    )
    if len(synthetic):
        domains = sorted(synthetic.domain.unique())
        arms = sorted(synthetic.arm.unique())
        fig, ax = make_figure(width=18, height=10, left=0.18, right=0.95, top=0.77)
        matrix = np.zeros((len(domains), len(arms)))
        for i, domain in enumerate(domains):
            for j, arm in enumerate(arms):
                matrix[i, j] = (
                    synthetic[synthetic.domain.eq(domain) & synthetic.arm.eq(arm)]
                    .iloc[0]
                    .agreement
                )
        cmap = rate_cmap()
        ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for i, domain in enumerate(domains):
            for j, arm in enumerate(arms):
                r = synthetic[synthetic.domain.eq(domain) & synthetic.arm.eq(arm)].iloc[
                    0
                ]
                ax.text(
                    j,
                    i,
                    f"{r.agreement:.0%}\n{int(r.matched)}/{int(r.n)}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=heatmap_ink(r.agreement, cmap, 0, 1),
                )
        ax.set_yticks(range(len(domains)), domains, fontsize=12)
        ax.set_xticks(
            range(len(arms)),
            [NAMES.get(a, a).replace(" ", "\n") for a in arms],
            fontsize=11,
        )
        ax.set_xticks(np.arange(0.5, len(arms) - 0.5), minor=True)
        ax.set_yticks(np.arange(0.5, len(domains) - 0.5), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        header(
            ax,
            "Policy following across nine domains",
            "Five synthetic dialogues per domain · original-turn and distractor references are provisional",
            "Cells show exact reference agreement and matched / total cases. Unresolved anchors are excluded; failures count as mismatches.",
            title_size=22,
            subtitle_size=13,
            key_size=11,
        )
        save(fig, output, "domain_reference_agreement")
    if len(pairs) and len(pairs.arm.unique()) > 3:
        plot_grid(pairs, output)
    elif len(pairs):
        pairs = pairs.sort_values(["arm", "pair_id"])
        values = np.array(
            [
                [
                    0
                    if r[k] == "engage" and r[s] == "ok"
                    else 1
                    if r[k] == "redirect" and r[s] == "ok"
                    else 2
                    for k, s in [
                        ("allowed_label", "allowed_status"),
                        ("excluded_label", "excluded_status"),
                    ]
                ]
                for r in pairs.to_dict("records")
            ]
        )
        fig, ax = make_figure(
            width=12,
            height=max(6, len(pairs) * 0.42 + 3),
            left=0.29,
            right=0.90,
            top=0.77,
        )
        ax.imshow(
            values,
            cmap=ListedColormap(["#D4E0FB", "#E6A4D3", "#E8E8E8"]),
            vmin=0,
            vmax=2,
            aspect="auto",
        )
        for i, row in enumerate(pairs.to_dict("records")):
            for j, (key, status, expected) in enumerate(
                [
                    ("allowed_label", "allowed_status", "engage"),
                    ("excluded_label", "excluded_status", "redirect"),
                ]
            ):
                label = row[key] if row[status] == "ok" else row[status]
                ax.text(
                    j,
                    i,
                    str(label) + (" *" if label != expected else ""),
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                )
        ax.set_yticks(
            range(len(pairs)),
            [NAMES.get(r.arm, r.arm) + " · " + r.domain for r in pairs.itertuples()],
            fontsize=10,
        )
        ax.set_xticks(
            [0, 1],
            [
                "Policy permits the topic\nExpected: engage",
                "Policy excludes the topic\nExpected: redirect",
            ],
            fontsize=12,
        )
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks(np.arange(0.5, len(pairs) - 0.5), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        header(
            ax,
            "Does the decision change when the policy changes?",
            "Same user request, history, and option order · benign topic probes",
            "Each cell is one recorded decision. * = wrong or failed. Engage/redirect is separate from content safety.",
            title_size=20,
            subtitle_size=12,
            key_size=10,
        )
        save(fig, output, "policy_switches")
    lines = [
        "# Policy-following diagnostic",
        "",
        "The task is engage/redirect under a supplied scenario policy. It does not assign safe/unsafe labels.",
        "",
        "The human-test cohort covers banking. The synthetic cohort samples five dialogues from each of nine domains. Synthetic first-turn and distractor references are provisional; banking positive controls were reviewed by this agent. Unreviewed originals and unresolved anchors are excluded. These cohorts are reported separately.",
        "",
        "## Results",
        "",
    ]
    display = cohorts.copy()
    display["Model"] = display.arm.map(lambda a: NAMES.get(a, a))
    display["Cohort"] = display.cohort
    display["cohort_order"] = display.cohort.map(
        {
            "synthetic_topic_following": 0,
            "source_topic_following": 1,
            "paired_policy": 2,
        }
    )
    display = display.sort_values(["cohort_order", "Model"])
    display["Correct / n"] = display.correct.astype(str) + " / " + display.n.astype(str)
    display["Agreement"] = display.strict_accuracy.map(lambda x: f"{x:.1%}")
    display["Cohort"] = display.cohort.map(
        {
            "paired_policy": "Authored policy pairs",
            "source_topic_following": "Banking human test (provisional)",
            "synthetic_topic_following": "Nine-domain synthetic (provisional)",
        }
    )
    display["Invalid / missing"] = display.invalid_or_missing
    lines.append(
        markdown_table(
            display,
            ["Model", "Cohort", "Correct / n", "Agreement", "Invalid / missing"],
        )
    )
    lines += [
        "",
        "These are exact counts for the selected diagnostic cases. Paired conditions share inputs and source controls share dialogues; no population confidence interval is reported.",
    ]
    lines += [
        "",
        f"Execution: {len(results):,}/{len(jobs):,} results recorded; {sum(r.get('status') != 'ok' for r in results)} final execution/output failures; {len(archived_attempts)} archived failed attempts. {sum(bool(r.get('reused_from')) for r in results)} matching pilot results were reused.",
    ]
    if len(synthetic):
        lines += [
            "",
            "## Nine-domain results",
            "",
            "![Domain reference agreement](domain_reference_agreement.png)",
            "",
            "Each domain contributes five synthetic dialogues. Case counts differ because unmatched distractor anchors are excluded. Agreement is against provisional topic references, not human safety labels.",
            "",
            "## Engage and redirect separately",
            "",
        ]
        labels = pd.DataFrame(
            [r for r in label_metrics if r["cohort"] == "synthetic_topic_following"]
        )
        labels["Model"] = labels.arm.map(lambda a: NAMES.get(a, a))
        labels["Reference"] = labels.reference_label
        labels["Matched / n"] = (
            labels.matched.astype(str) + " / " + labels.n.astype(str)
        )
        labels["Agreement"] = labels.agreement.map(lambda x: f"{x:.1%}")
        lines.append(
            markdown_table(labels, ["Model", "Reference", "Matched / n", "Agreement"])
        )
    if len(pairs):
        lines += [
            "",
            "## Paired policies",
            "",
            "![Policy switches](policy_switches.png)",
            "",
        ]
        for arm, g in pairs.groupby("arm"):
            lines.append(
                f"- {NAMES.get(arm, arm)}: {int(g.both_correct.sum())}/{len(g)} pairs correct in both conditions; {int(g.decision_changed.sum())}/{int(g.both_valid.sum())} valid pairs changed decision; {int((~g.both_valid).sum())} invalid or missing pairs."
            )
    lines += [
        "",
        "The paired cases are controlled probes authored for this project, not the Aegis paper’s novel-category benchmark. This small diagnostic is not a held-out generalization result.",
        "",
        "## Files",
        "",
        "- `cases.csv`: each expected and predicted engage/redirect decision.",
        "- `cohort_metrics.csv`: reference agreement and invalid counts, by model and cohort.",
        "- `domain_metrics.csv` and `label_metrics.csv`: results by domain and reference label.",
        "- `policy_pairs.csv`: paired decisions, validity, and whether both answers are correct.",
        "- `source_disagreements.jsonl`: source disagreements with full policies and preceding context for review.",
        "",
        "[Source dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)",
        "",
    ]
    if disagreements:
        review = []
        source_ids = list(dict.fromkeys(d["case_id"] for d in disagreements))
        for case_id in source_ids[:10]:
            group = [d for d in disagreements if d["case_id"] == case_id]
            request = json.loads(group[0]["request"]["text"])
            review.append(
                {
                    "Case": case_id,
                    "Current user turn": request["current_user_turn"],
                    "Reference": group[0]["provisional_reference"],
                    "Predictions": "; ".join(
                        NAMES.get(d["arm"], d["arm"]) + ": " + str(d["predicted_label"])
                        for d in group
                    ),
                }
            )
        lines += [
            "## Source labels to review",
            "",
            markdown_table(pd.DataFrame(review), list(review[0])),
            "",
            f"Showing up to 10 of {len(source_ids)} disputed cases. All policies and contexts are in `source_disagreements.jsonl`. Inspect the supplied policy before treating reference disagreements as failures.",
            "",
        ]
    (output / "report.md").write_text("\n".join(lines))
    summary = {
        "output": str(output),
        "manifest_requests": len(jobs),
        "recorded_results": len(results),
        "missing_results": len(jobs) - len(results),
        "arms": len(data.arm.unique()),
        "unique_cases": data.case_id.nunique(),
        "reused_results": sum(bool(r.get("reused_from")) for r in results),
        "archived_failed_attempts": len(archived_attempts),
        "final_invalid_or_errors": int((~data.valid).sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, nargs="+", default=[OUT / "pilot"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/topic-following",
    )
    args = p.parse_args()
    print(json.dumps(report(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
