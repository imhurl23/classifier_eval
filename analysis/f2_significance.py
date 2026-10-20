"""Exploratory paired delta-method tests of F2 differences.

Run: .venv/bin/python -m analysis.f2_significance
The sampling unit is a dataset/item_id, shared across models. Holm correction
covers all model pairs across the seven question/test cohorts together.
Approximate inference assumes independent examples, conditional on saved runs.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from analysis.plot_safety_f2 import OUT
from analysis.plot_safety_usecases import COHORTS, SOURCE, select
from analysis.source_report import NAMES


def estimate(g):
    positive = "refusal" if g.target.iloc[0] == "response_refusal" else "unsafe"
    actual = g.source_gold.eq(positive).to_numpy()
    predicted = (g.status.eq("ok") & g.label.eq(positive)).to_numpy()
    observations = np.column_stack(
        [actual & predicted, actual & ~predicted, ~actual & predicted]
    ).astype(float)
    tp, fn, fp = observations.mean(axis=0)
    den = 5 * tp + 4 * fn + fp
    score = 5 * tp / den
    gradient = np.array([5 * (4 * fn + fp), -20 * tp, -5 * tp]) / den**2
    influence = (observations - observations.mean(axis=0)) @ gradient
    return score, influence, observations


def main():
    judgments = pd.read_csv(SOURCE / "judgments.csv.gz", low_memory=False)
    f2 = pd.read_csv(OUT / "f2_metrics.csv")
    rng = np.random.default_rng(20261015)
    records = []
    leaders = []
    for cohort in f2.cohort.unique():
        g = select(judgments, COHORTS[cohort])
        g = g[
            g.policy_mode.eq("common_policy")
            & g.record_kind.eq("physical")
            & g.source_gold.notna()
        ]
        estimates = {}
        keys = {}
        for arm, rows in g.groupby("arm"):
            rows = rows.sort_values(["source_dataset", "item_id"])
            assert not rows.duplicated(["source_dataset", "item_id"]).any()
            keys[arm] = list(zip(rows.source_dataset, rows.item_id))
            estimates[arm] = estimate(rows)
            expected = f2[f2.cohort.eq(cohort) & f2.arm.eq(arm)].iloc[0]
            assert np.isclose(estimates[arm][0], expected.f2)
        leader = max(estimates, key=lambda a: estimates[a][0])
        for a, b in combinations(sorted(estimates), 2):
            assert keys[a] == keys[b]
            fa, ia, _ = estimates[a]
            fb, ib, _ = estimates[b]
            n = len(ia)
            delta = fa - fb
            se = np.std(ia - ib, ddof=1) / np.sqrt(n)
            p = 2 * norm.sf(abs(delta / se)) if se else (1.0 if delta == 0 else 0.0)
            records.append(
                dict(
                    cohort=cohort,
                    arm_a=a,
                    arm_b=b,
                    n=n,
                    f2_a=fa,
                    f2_b=fb,
                    difference=delta,
                    se_paired=se,
                    p_unadjusted=p,
                    ci_low=delta - norm.ppf(0.975) * se,
                    ci_high=delta + norm.ppf(0.975) * se,
                )
            )
        # Independent numerical check of the leader/runner-up delta-method SE
        # against paired row bootstrap, compressed by their joint outcomes.
        runner = sorted(estimates, key=lambda a: estimates[a][0], reverse=True)[1]
        joint = np.column_stack([estimates[leader][2], estimates[runner][2]])
        patterns, counts = np.unique(joint, axis=0, return_counts=True)
        draws = rng.multinomial(len(joint), counts / counts.sum(), size=30000)
        sums = draws @ patterns
        sa = 5 * sums[:, 0] / (5 * sums[:, 0] + 4 * sums[:, 1] + sums[:, 2])
        sb = 5 * sums[:, 3] / (5 * sums[:, 3] + 4 * sums[:, 4] + sums[:, 5])
        analytic = np.std(
            estimates[leader][1] - estimates[runner][1], ddof=1
        ) / np.sqrt(len(joint))
        ratio = np.std(sa - sb, ddof=1) / analytic
        assert 0.9 < ratio < 1.1, (cohort, ratio)
        leaders.append(
            dict(
                cohort=cohort,
                leader=leader,
                runner_up=runner,
                bootstrap_se_ratio=ratio,
                delta_bootstrap_low=np.quantile(sa - sb, 0.025),
                delta_bootstrap_high=np.quantile(sa - sb, 0.975),
            )
        )
    result = pd.DataFrame(records)
    result["p_holm_all_345"] = multipletests(result.p_unadjusted, method="holm")[1]
    result["significant_holm_05"] = result.p_holm_all_345.lt(0.05)
    z = norm.ppf(1 - 0.05 / (2 * len(result)))
    result["simultaneous_ci_low"] = result.difference - z * result.se_paired
    result["simultaneous_ci_high"] = result.difference + z * result.se_paired
    result.to_csv(OUT / "f2_pairwise_significance.csv", index=False)
    pd.DataFrame(leaders).to_csv(OUT / "f2_leader_bootstrap_checks.csv", index=False)
    print(
        f"{int(result.significant_holm_05.sum())}/{len(result)} pairs significant after global Holm correction."
    )
    for r in leaders:
        c = r["cohort"]
        lead = r["leader"]
        runner = r["runner_up"]
        row = result[
            result.cohort.eq(c)
            & (
                (result.arm_a.eq(lead) & result.arm_b.eq(runner))
                | (result.arm_b.eq(lead) & result.arm_a.eq(runner))
            )
        ].iloc[0]
        indist = []
        beaten = []
        for x in result[
            result.cohort.eq(c) & (result.arm_a.eq(lead) | result.arm_b.eq(lead))
        ].itertuples():
            other = x.arm_b if x.arm_a == lead else x.arm_a
            (beaten if x.significant_holm_05 else indist).append(NAMES[other])
        print(
            c,
            "\n leader:",
            NAMES[lead],
            "\n runner:",
            NAMES[runner],
            "gap:",
            abs(row.difference) * 100,
            "pp; Holm p:",
            row.p_holm_all_345,
            "\n significantly beats:",
            ", ".join(beaten),
            "\n no significant difference:",
            ", ".join(indist),
        )
    print("\nSELECTED COMPARISONS:")
    for c, a, b in [
        ("wildguard_response", "clef_choice", "claude_structured"),
        ("wildguard_response", "clef_choice", "luna_structured"),
        ("xstest_prompt", "claude_structured", "clef_choice"),
        ("wildguard_refusal", "safeguard_120b", "jev_choice"),
        ("wildguard_conversation", "clef_choice", "qwen35_4b_letters"),
    ]:
        x = result[
            result.cohort.eq(c)
            & (
                (result.arm_a.eq(a) & result.arm_b.eq(b))
                | (result.arm_a.eq(b) & result.arm_b.eq(a))
            )
        ].iloc[0]
        sign = 1 if x.arm_a == a else -1
        print(
            c,
            NAMES[a],
            "-",
            NAMES[b],
            "delta",
            sign * x.difference * 100,
            "95% CI",
            sorted([sign * x.ci_low * 100, sign * x.ci_high * 100]),
            "Holm p",
            x.p_holm_all_345,
        )


if __name__ == "__main__":
    main()
