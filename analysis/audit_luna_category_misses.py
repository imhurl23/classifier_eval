"""Offline audit of chart 04 against original labels and saved wire payloads.

Run with: .venv/bin/python -m analysis.audit_luna_category_misses
No model calls or credentials are used.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from adapters.openai_decisions import parse_multi_choice
from run.offline_review import digest, request_for

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/processed/source-full/000bf1bcb6dc"
OUT = ROOT / "analysis/output/failures-2026-10-12/luna-audit"


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def main():
    samples = {r["id"]: r for r in records(RUN / "sample.jsonl")}
    results = [
        r
        for r in records(RUN / "results.jsonl")
        if r["target"] == "prompt_harm"
        and samples[r["item_id"]]["metadata"]["source"] == "wildguardtest"
    ]
    luna = [r for r in results if r["arm"] == "decisions_luna"]
    assert len(luna) == len({r["item_id"] for r in luna}) == 1681
    manifest = {
        r["job_id"]: r
        for r in records(RUN / "manifest.jsonl")
        if r["arm"] == "decisions_luna" and r["target"] == "prompt_harm"
    }
    original = pd.read_parquet(ROOT / "data/raw/wildguard/test/wildguard_test.parquet")
    tidy = pd.read_csv(
        ROOT / "analysis/output/source-2026-10-12/judgments.csv.gz", low_memory=False
    )
    tidy = tidy.set_index("job_id")
    assert tidy.index.is_unique
    checked = []
    for r in luna:
        sample = samples[r["item_id"]]
        source = original.iloc[sample["metadata"]["source_id"]]
        gold = {"harmful": "unsafe", "unharmful": "safe"}[source.prompt_harm_label]
        assert source.prompt == sample["input"]["prompt"]
        assert source.response == sample["input"]["response"]
        assert source.subcategory == sample["metadata"]["wildguard_subcategory"]
        assert gold == sample["expected"]["prompt_s2"] == r["source_gold"]
        request, response = r["raw"]["request"], r["raw"]["response"]
        assert request == manifest[r["job_id"]]["request"]
        assert (
            digest(request)
            == r["request_sha256"]
            == manifest[r["job_id"]]["request_sha256"]
        )
        assert request == request_for(
            "decisions_luna", sample, "prompt_harm", "prompt_harm_s2"
        )
        assert request["input"] == sample["input"]["text"]
        assert len(request["questions"]) == len(response["answers"]) == 1
        parsed = parse_multi_choice(response, request, 0)["safety"]
        assert parsed.status == r["status"] == "ok"
        assert parsed.label == r["label"] == response["answers"][0]["choice"]
        assert parsed.probs == r["probs"]
        assert parsed.probs[parsed.label] == max(parsed.probs.values())
        t = tidy.loc[r["job_id"]]
        assert t.label == r["label"] and t.source_gold == gold and t.status == "ok"
        assert t["score__source_accuracy"] == int(r["label"] == gold)
        checked.append(
            {
                "item_id": r["item_id"],
                "job_id": r["job_id"],
                "category": source.subcategory,
                "source_gold": gold,
                "prediction": r["label"],
                "p_unsafe": r["probs"]["unsafe"],
                "first_option": request["questions"][0]["choices"][0]["value"],
                "prompt": source.prompt,
            }
        )
    cases = pd.DataFrame(checked)
    slices = pd.read_csv(ROOT / "analysis/output/failures-2026-10-12/slice_stats.csv")
    slices = slices[
        (slices.arm == "decisions_luna")
        & (slices.target == "prompt_harm")
        & (slices.source_dataset == "wildguardtest")
        & (slices.policy_mode == "common_policy")
        & (slices.slice_type == "source_tags")
    ]
    category_rows = []
    for category, g in cases[cases.source_gold == "unsafe"].groupby("category"):
        k, n = int(g.prediction.eq("safe").sum()), len(g)
        saved = slices[slices.slice_value == category]
        assert len(saved) == 1
        saved = saved.iloc[0]
        assert saved.unsafe_miss_k == k and saved.unsafe_miss_n == n
        assert abs(saved.unsafe_miss_rate - k / n) < 1e-12
        category_rows.append(
            {"category": category, "missed": k, "unsafe_n": n, "miss_rate": k / n}
        )
    comparisons = []
    for arm in sorted({r["arm"] for r in results}):
        group = [r for r in results if r["arm"] == arm and r["source_gold"] == "unsafe"]
        comparisons.append(
            {
                "arm": arm,
                "unsafe_n": len(group),
                "unsafe_labeled_safe": sum(
                    r["status"] == "ok" and r["label"] == "safe" for r in group
                ),
            }
        )
    summary = {
        "audit_scope": "Offline artifact integrity; does not validate server implementation or policy alignment.",
        "checked_requests": len(luna),
        "checked_category_cells": len(category_rows),
        "checks": [
            "original parquet prompt/response/category and prompt-only gold",
            "frozen manifest request and hash; current request reconstruction",
            "saved API answer, parser replay, recorded label and probabilities",
            "tidy CSV label and score; unique category aggregation and counts",
        ],
        "confusion": dict(Counter(cases.source_gold + " -> " + cases.prediction)),
        "unsafe_misses_by_first_option": [
            {
                "first_option": option,
                "missed": int(g.prediction.eq("safe").sum()),
                "unsafe_n": len(g),
            }
            for option, g in cases[cases.source_gold == "unsafe"].groupby(
                "first_option"
            )
        ],
        "hosted_comparisons": comparisons,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUT / "checked_cases.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUT / "category_recount.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
