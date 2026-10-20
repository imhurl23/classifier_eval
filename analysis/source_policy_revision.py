"""Analyze the completed policy revision without changing archived results."""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from analysis import plot_safety_f2 as plots
from analysis.f2_significance import estimate
from analysis.plot_safety_usecases import COHORTS, aggregate, select
from analysis.source_report import normalize, quality
from run import source_policy_revision as run

ROOT = run.ROOT
OUT = ROOT / "analysis/output/source-policy-revision-2026-10-19"
OLD = ROOT / "analysis/output/source-2026-10-12/judgments.csv.gz"
COHORT_KEYS = (
    "aegis_single",
    "aegis_conversation",
    "wildguard_prompt",
    "wildguard_response",
    "wildguard_refusal",
    "xstest_prompt",
)


def load():
    summary = run.verify()
    samples = {r["id"]: r for r in run.records(run.SOURCE / "sample.jsonl")}
    old = pd.read_csv(OLD, low_memory=False)
    rows, diagnostics = [], []
    for arm in run.ARMS:
        folder = run.OUT / arm
        receipt = folder / "verification.json"
        if not receipt.exists() or not json.loads(receipt.read_text())["complete"]:
            raise ValueError("Analysis requires complete coverage: " + arm)
        counts = {}
        seen = set()
        for record in run.records(folder / "results.jsonl"):
            assert record["job_id"] not in seen and record["actual_request_matches"]
            seen.add(record["job_id"])
            counts[record["stage"]] = counts.get(record["stage"], 0) + 1
            sample = samples[record["item_id"]]
            provider = "hosted" if arm in run.HOSTED else "baseten"
            normalized = normalize(
                record, sample, provider, run.RATES, run.RATES, run.VERSION
            )
            normalized.update(
                policy_version=run.VERSION,
                source_aligned=True,
                reused=False,
                stage=record["stage"],
                variant=record["variant"],
                replicate=record["replicate"],
                reversed_options=record["reversed_options"],
            )
            (rows if record["stage"] == "core" else diagnostics).append(normalized)
        assert counts == summary["counts"][arm], (arm, counts)
    new = pd.DataFrame(rows)
    reused = old[
        old.source_dataset.eq("xstest")
        & old.policy_mode.eq("common_policy")
        & old.record_kind.eq("physical")
    ].copy()
    reused["reused"] = True
    reused["source_aligned"] = False
    reused["policy_version"] = "original_unchanged_xstest"
    reused["stage"] = "core"
    frame = pd.concat([new, reused], ignore_index=True)
    assert len(frame) == 89251 and frame.job_id.is_unique
    assert not frame.duplicated(["arm", "source_dataset", "item_id", "target"]).any()
    return frame, pd.DataFrame(diagnostics), old, summary


def pairwise(frame):
    records = []
    for key in COHORT_KEYS:
        data = select(frame, COHORTS[key])
        estimates, ids = {}, {}
        for arm, group in data.groupby("arm"):
            group = group.sort_values(["source_dataset", "item_id"])
            estimates[arm] = estimate(group)
            ids[arm] = list(zip(group.source_dataset, group.item_id))
        for a, b in combinations(sorted(estimates), 2):
            assert ids[a] == ids[b]
            fa, ia, _ = estimates[a]
            fb, ib, _ = estimates[b]
            se = np.std(ia - ib, ddof=1) / np.sqrt(len(ia))
            delta = fa - fb
            p = 2 * norm.sf(abs(delta / se)) if se else float(delta == 0)
            records.append(
                dict(
                    cohort=key,
                    arm_a=a,
                    arm_b=b,
                    n=len(ia),
                    f2_a=fa,
                    f2_b=fb,
                    difference=delta,
                    se_paired=se,
                    p_unadjusted=p,
                    ci_low=delta - 1.95996398454 * se,
                    ci_high=delta + 1.95996398454 * se,
                )
            )
    result = pd.DataFrame(records)
    assert len(result) == 290
    result["p_holm_all_290"] = multipletests(result.p_unadjusted, method="holm")[1]
    result["significant_holm_05"] = result.p_holm_all_290.lt(0.05)
    return result


def main():
    frame, diagnostics, old, manifest = load()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "judgments.csv.gz", index=False)
    diagnostics.to_csv(OUT / "diagnostic_decisions.csv.gz", index=False)
    quality(frame).to_csv(OUT / "quality_by_configuration.csv", index=False)
    metrics = pd.concat(
        [aggregate(frame, COHORTS[k], k) for k in COHORT_KEYS], ignore_index=True
    )
    metrics.to_csv(OUT / "usecase_metrics.csv", index=False)
    f2 = plots.compute(metrics)
    assert len(f2) == 62
    f2.to_csv(OUT / "f2_metrics.csv", index=False)
    pairs = pairwise(frame)
    pairs.to_csv(OUT / "f2_pairwise_significance.csv", index=False)
    # Compare the same model, example, and target. This is a descriptive policy
    # revision comparison, not a fresh held-out confirmation or causal estimate.
    keys = ["arm", "item_id", "target"]
    prior = old[old.policy_mode.eq("common_policy") & old.record_kind.eq("physical")]
    comparison = frame.merge(
        prior[keys + ["label", "status", "source_gold"]],
        on=keys,
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert len(comparison) == len(frame)
    assert comparison.source_gold_new.equals(comparison.source_gold_old)
    comparison["correct_new"] = comparison.status_new.eq(
        "ok"
    ) & comparison.label_new.eq(comparison.source_gold_new)
    comparison["correct_old"] = comparison.status_old.eq(
        "ok"
    ) & comparison.label_old.eq(comparison.source_gold_old)
    comparison["label_changed"] = comparison.label_new.fillna("missing").ne(
        comparison.label_old.fillna("missing")
    )
    comparison[
        keys
        + [
            "source_dataset",
            "input_unit",
            "source_gold_new",
            "label_old",
            "label_new",
            "status_old",
            "status_new",
            "correct_old",
            "correct_new",
            "label_changed",
            "reused",
        ]
    ].to_csv(OUT / "old_vs_new_decisions.csv.gz", index=False)
    changes = []
    for (arm, source, unit, target), group in comparison.groupby(
        ["arm", "source_dataset", "input_unit", "target"]
    ):
        changes.append(
            dict(
                arm=arm,
                source_dataset=source,
                input_unit=unit,
                target=target,
                n=len(group),
                accuracy_old=group.correct_old.mean(),
                accuracy_new=group.correct_new.mean(),
                gained=int((group.correct_new & ~group.correct_old).sum()),
                lost=int((~group.correct_new & group.correct_old).sum()),
                changed_labels=int(group.label_changed.sum()),
            )
        )
    pd.DataFrame(changes).to_csv(OUT / "policy_revision_changes.csv", index=False)
    examples = [
        "aegis-validation-000415",
        "aegis-test-001047",
        "aegis-validation-000807",
        "wildguard-000292",
        "wildguard-000425",
    ]
    comparison[comparison.item_id.isin(examples)].drop(
        columns=["parent_job_ids"], errors="ignore"
    ).to_csv(OUT / "audited_cases_old_vs_new.csv", index=False)
    diagnostic_summary = (
        diagnostics.groupby(["stage", "arm", "scheme", "variant"], dropna=False)
        .agg(
            decisions=("item_id", "size"),
            valid_rate=("status", lambda x: x.eq("ok").mean()),
            source_binary_agreement=("score__source_accuracy", "mean"),
        )
        .reset_index()
    )
    diagnostic_summary.to_csv(OUT / "diagnostic_summary.csv", index=False)
    result = dict(
        policy_version=run.VERSION,
        configurations=len(f2),
        comparisons=len(pairs),
        significant_comparisons=int(pairs.significant_holm_05.sum()),
        physical_new_core=int((~frame.reused).sum()),
        reused_xstest=int(frame.reused.sum()),
        diagnostic_decisions=len(diagnostics),
        caveats=manifest["caveats"],
        holm_family=290,
        original_holm_family=345,
    )
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    from analysis.decision_story import main as build_decision_story

    build_decision_story()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
