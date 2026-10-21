"""Offline, allowlisted HF companion export; never uploads or changes frozen data."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from data.jsonl import read_jsonl
from data.topic_following import paired_cases, reconstruct, synthetic_cases

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MAIN = ROOT / "data/processed/source-full/000bf1bcb6dc/sample.jsonl"
TOPIC = ROOT / "data/processed/topic-following/multi-domain/cases.jsonl"
MAIN_SHA256 = "4ed143001f47e9839d476f6998635047a710ca6fbfbde5fba36d842ae90948e9"
TOPIC_SHA256 = "a723aaac32d2433703272fbf64e167265565796932ed32e0206511f4863e1984"
PUBLIC_NOTICES = (
    "CC-BY-4.0.txt",
    "aegis-source-card.md",
    "xstest-source-card.md",
    "topic-source-card.md",
    "wildguard-NOTICE.md",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


def load_sources():
    sources = json.loads((HERE / "sources.json").read_text())
    upstream = {}
    for key, source in sources.items():
        if file_sha(HERE / source["source_card_file"]) != source["source_card_sha256"]:
            raise ValueError(f"Changed license evidence: {key}")
        for filename, spec in source["files"].items():
            path = ROOT / spec["local_path"]
            if file_sha(path) != spec["sha256"]:
                raise ValueError(f"Changed pinned upstream file: {key}/{filename}")
            if filename.endswith(".csv"):
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            elif filename.endswith(".parquet"):
                import pandas as pd

                rows = json.loads(pd.read_parquet(path).to_json(orient="records"))
            elif filename.endswith(".jsonl"):
                rows = read_jsonl(path)
            else:
                rows = json.loads(path.read_text())
            upstream[key, filename] = rows
    return sources, upstream


def attribution(key, filename, index, source_id, sources, raw, *, locator=None):
    source = sources[key]
    return {
        "source_key": key,
        "source_title": source["title"],
        "creator": source["creator"],
        "source_url": source["url"],
        "revision": source["revision"],
        "source_file": filename,
        "source_file_sha256": source["files"][filename]["sha256"],
        "source_file_url": f"{source['url']}/blob/{source['revision']}/{filename}",
        "row_index": index,
        "row_index_base": 0,
        "source_id": str(source_id),
        "source_id_kind": "upstream_id"
        if key in {"aegis", "xstest"}
        else "zero_based_row_index",
        "source_record_sha256": digest(raw),
        "locator": locator or {},
        "license": source["license"],
        "license_url": source["license_url"],
        "citation_url": source["citation_url"],
        "source_card_url": source["source_card_url"],
        "terms_url": source.get("terms_url"),
        "rights_scope": "Immediate provider's published license; no inferred original author or additional third-party rights.",
    }


def main_record(row, sources, upstream):
    meta = row["metadata"]
    source = meta["source"]
    if source in {"aegis_v2_test", "aegis_v2_validation"}:
        key, split = "aegis", source.removeprefix("aegis_v2_")
        filename = split + ".json"
        match = re.fullmatch(r"aegis-(test|validation)-(\d{6})", row["id"])
        if not match or match[1] != split:
            raise ValueError("Invalid Aegis identity")
        index = int(match[2])
    elif source == "xstest":
        key, split, filename = "xstest", "test", "xstest_prompts.csv"
        match = re.fullmatch(r"xstest-(\d{6})", row["id"])
        if not match:
            raise ValueError("Invalid XSTest identity")
        index = int(match[1])
    elif source == "wildguardtest":
        key, split, filename = "wildguard", "test", "test/wildguard_test.parquet"
        match = re.fullmatch(r"wildguard-(\d{6})", row["id"])
        if not match:
            raise ValueError("Invalid WildGuard identity")
        index = int(match[1])
    else:
        raise ValueError(f"Unknown source: {source}")
    raw = upstream[key, filename][index]
    source_id = raw["id"] if key in {"aegis", "xstest"} else index
    if str(meta["source_id"]) != str(source_id) or meta["split"] != split:
        raise ValueError(f"Source identity mismatch: {row['id']}")
    prompt, response = raw["prompt"], raw.get("response") or None
    if (
        row["input"]["prompt"] != prompt
        or (row["input"].get("response") or None) != response
    ):
        raise ValueError(f"Source text mismatch: {row['id']}")
    categories = []
    if key == "aegis":
        categories = [
            x.strip()
            for x in (raw.get("violated_categories") or "").split(",")
            if x.strip()
        ]
        if (
            row["expected"]["s2"] != raw["prompt_label"]
            or row["expected"]["aegis_categories"] != categories
        ):
            raise ValueError("Aegis source label/category mismatch")
        labels = {
            k: raw.get(k)
            for k in (
                "prompt_label",
                "response_label",
                "prompt_label_source",
                "response_label_source",
            )
        }
        scope = "Human interaction label (single turn when no response); response label has its own provenance."
    elif key == "xstest":
        if row["expected"]["s2"] != raw["label"]:
            raise ValueError("XSTest label mismatch")
        labels = {"label": raw["label"], "type": raw["type"]}
        scope = "Original prompt label; not a generated-response score."
    else:
        categories = [raw["subcategory"]] if raw.get("subcategory") else []
        mapping = {"harmful": "unsafe", "unharmful": "safe"}
        for target in ("prompt", "response"):
            if row["expected"][target + "_s2"] != mapping[raw[target + "_harm_label"]]:
                raise ValueError("WildGuard label mismatch")
        labels = {
            k: raw.get(k)
            for k in (
                "prompt_harm_label",
                "response_harm_label",
                "response_refusal_label",
            )
        }
        scope = "Separate human prompt harm, response harm and refusal targets; no new adjudication."
    attr = attribution(key, filename, index, source_id, sources, raw)
    attr["source_split"] = split
    content = {"prompt": prompt, "response": response}
    return {
        "id": row["id"],
        "attribution": attr,
        "source_labels": labels,
        "label_scope": scope,
        "source_categories": categories,
        "content": content,
        "transformations": [
            "Selected from the frozen study cohort.",
            "Prompt and response text preserved verbatim; empty response normalized to null.",
            "Original source labels retained without relabeling.",
        ],
        "redacted": prompt.strip().casefold() == "redacted"
        or raw.get("reconstruction_id_if_redacted") is not None,
    }


def topic_records(rows, sources, upstream):
    human_file = "canttalkaboutthis_topic_control_human_test_set.jsonl"
    synthetic_file = "canttalkaboutthis_topic_control_mixtral.jsonl"
    human, _, _ = reconstruct(upstream["topic", human_file])
    # Actual per-domain sample size is recorded by the frozen topic builder.
    summary = json.loads(TOPIC.with_name("summary.json").read_text())
    synthetic, _, _ = synthetic_cases(
        upstream["topic", synthetic_file], summary["synthetic_dialogues_per_domain"]
    )
    canonical = {r["id"]: r for r in human + synthetic + paired_cases()}
    for row in rows:
        if row != canonical.get(row["id"]):
            raise ValueError(f"Topic reconstruction mismatch: {row['id']}")
        meta = row["metadata"]
        if meta["cohort"] == "paired_policy":
            # Authored probes are tracked but not licensed/released as third-party data.
            attr = {
                "source_key": "study_authored",
                "creator": "Braintrust classifier_eval study",
                "source_id": row["id"],
                "source_id_kind": "study_id",
                "license": "not_assigned",
                "source_file": "data/topic_following.py",
                "source_file_sha256": file_sha(ROOT / "data/topic_following.py"),
                "source_record_sha256": digest(row),
            }
        else:
            filename = (
                synthetic_file
                if meta["cohort"] == "synthetic_topic_following"
                else human_file
            )
            index = meta["dialogue_index"]
            raw = upstream["topic", filename][index]
            locator = {
                k: meta[k]
                for k in ("kind", "turn_index", "anchor_turn_index", "anchor_match")
                if k in meta
            }
            if meta["kind"] == "distractor":
                locator["distractor_index"] = int(row["id"].rsplit("-", 1)[1])
            attr = attribution(
                "topic", filename, index, index, sources, raw, locator=locator
            )
            attr["source_split"] = (
                "human_test" if filename == human_file else "synthetic"
            )
        yield {
            "id": row["id"],
            "attribution": attr,
            "source_labels": {},
            "derived_labels": {
                "topic_action": row["expected"],
                "provenance": meta["label_provenance"],
            },
            "label_scope": "Study topic-control reference; not human content-safety gold.",
            "source_categories": [],
            "content": row["input"],
            "redacted": False,
            "transformations": [
                "Frozen topic selection.",
                "History ends at the intervention; later turns omitted.",
                "Topic action labels retain source/agent/provisional provenance.",
            ],
        }


def candidates():
    if file_sha(MAIN) != MAIN_SHA256 or file_sha(TOPIC) != TOPIC_SHA256:
        raise ValueError("Frozen study membership or content changed")
    sources, upstream = load_sources()
    main = [main_record(r, sources, upstream) for r in read_jsonl(MAIN)]
    if len(main) != 5252:
        raise ValueError("Frozen main cohort size changed")
    rows = main + list(topic_records(read_jsonl(TOPIC), sources, upstream))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate study IDs")
    for row in rows:
        row["review_sha256"] = digest(row)
    return rows, sources


def validate_reviews(reviews, rows, policy):
    by_id = {r["id"]: r for r in rows}
    result = {}
    for review in reviews:
        item_id = review["id"]
        if item_id in result or item_id not in by_id:
            raise ValueError("Duplicate or unknown review ID")
        if (
            review["review_sha256"] != by_id[item_id]["review_sha256"]
            or review["policy_version"] != policy["version"]
        ):
            raise ValueError(f"Stale review: {item_id}")
        if review["decision"] not in {"publish", "withhold"}:
            raise ValueError("Invalid review decision")
        if (
            not review.get("reviewer")
            or not review.get("reviewed_on")
            or not review.get("reason")
        ):
            raise ValueError("Missing review provenance")
        if review.get("reviewer_type") not in {"agent", "human"}:
            raise ValueError("Review must distinguish agent and human")
        if not isinstance(review.get("excluded_topics"), list) or set(
            review["excluded_topics"]
        ) - set(policy["excluded_review_topics"]):
            raise ValueError("Unknown review topic")
        result[item_id] = review
    return result


def distribution_reasons(row, review, policy):
    reasons = []
    key = row["attribution"]["source_key"]
    if key == "wildguard":
        reasons.append("upstream_access_only")
    if key == "study_authored":
        reasons.append("contribution_license_not_assigned")
    if row["redacted"]:
        reasons.append("provider_redacted_or_separately_licensed")
    for category in row["source_categories"]:
        if category in policy["excluded_source_categories"]:
            reasons.append(policy["excluded_source_categories"][category])
    # Actual identifier-shaped text is held regardless of source safety labels.
    text = unicodedata.normalize("NFKC", json.dumps(row["content"], ensure_ascii=False))
    if re.search(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}-\d{2}-\d{4}\b|-----BEGIN .*PRIVATE KEY-----",
        text,
    ):
        reasons.append("identifier_or_secret_pattern")
    if review is None:
        reasons.append("not_reviewed")
    else:
        reasons.extend(review["excluded_topics"])
        if review["decision"] != "publish":
            reasons.append("review_withheld")
    return sorted(set(reasons))


def build(output, reviews_path):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    policy = json.loads((HERE / "release_policy.json").read_text())
    rows, sources = candidates()
    reviews = validate_reviews(read_jsonl(reviews_path), rows, policy)
    references, released = [], []
    for row in rows:
        review = reviews.get(row["id"])
        reasons = distribution_reasons(row, review, policy)
        # Explicit public schema: no arbitrary input metadata, excerpts or review notes.
        references.append(
            {
                "id": row["id"],
                "attribution": row["attribution"],
                "review_sha256": row["review_sha256"],
                "text_released": not reasons,
                "text_exclusion_reasons": reasons,
            }
        )
        if not reasons:
            if row["attribution"]["license"] != "CC-BY-4.0":
                raise ValueError("Text license not allowlisted")
            released.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "attribution",
                        "content",
                        "source_labels",
                        "label_scope",
                        "transformations",
                        "review_sha256",
                    )
                }
            )
            if "derived_labels" in row:
                released[-1]["derived_labels"] = row["derived_labels"]
            released[-1]["review"] = {
                k: review[k]
                for k in ("reviewer", "reviewer_type", "reviewed_on", "policy_version")
            }
    counts = Counter(
        reason for r in references for reason in r["text_exclusion_reasons"]
    )
    summary = {
        "policy_version": policy["version"],
        "source_rows": len(rows),
        "main_study_rows": 5252,
        "topic_rows": len(rows) - 5252,
        "released_text_rows": len(released),
        "withheld_text_rows": len(rows) - len(released),
        "exclusion_reasons_nonexclusive": dict(sorted(counts.items())),
        "released_by_source": dict(
            Counter(r["attribution"]["source_key"] for r in released)
        ),
        "reviewed_rows": len(reviews),
        "review_type": "See each row; agent review is not human adjudication.",
        "benchmark_scores": "Not recomputed. Published full-cohort metrics must not be attributed to this filtered text subset.",
        "inputs_sha256": {
            "main_sample": file_sha(MAIN),
            "topic_sample": file_sha(TOPIC),
            "reviews": file_sha(reviews_path),
            "policy": file_sha(HERE / "release_policy.json"),
            "sources": file_sha(HERE / "sources.json"),
            "exporter": file_sha(Path(__file__)),
        },
        "out_of_scope": [
            "HarmBench",
            "HH-RLHF full-dialogue artifacts",
            "training data",
            "model weights",
            "provider outputs and per-example predictions",
            "working repository and Git history",
        ],
    }
    output.mkdir(parents=True)
    write_rows(output / "source_references.jsonl", references)
    write_rows(output / "reviewed_text.jsonl", released)
    write_json(output / "release_summary.json", summary)
    public_sources = {
        k: {
            field: value
            for field, value in s.items()
            if field not in {"files", "source_card_file"}
        }
        for k, s in sources.items()
    }
    write_json(output / "source_attribution.json", public_sources)
    for name in ("DATASET_CARD.md", "LICENSE.md", "release_policy.json"):
        shutil.copyfile(
            HERE / name, output / ("README.md" if name == "DATASET_CARD.md" else name)
        )
    (output / "notices").mkdir()
    for name in PUBLIC_NOTICES:
        shutil.copyfile(HERE / "notices" / name, output / "notices" / name)
    write_json(
        output / "checksums.json",
        {
            str(p.relative_to(output)): file_sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=HERE / "reviews.jsonl")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.reviews), indent=2))


if __name__ == "__main__":
    main()
