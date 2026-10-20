"""Offline aggregate export for Every Eval Ever; never runs inference or uploads.

The CSV is the complete, frozen main F2 comparison. Metadata explicitly retains
the study's task variants, unresolved registry identities, and reused XSTest run.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import datastore_path_components

COLLECTION = "decision-interface-safety"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "analysis/output/source-policy-revision-2026-10-19/f2_metrics.csv"
DEFAULT_METADATA = Path(__file__).with_name("source_metadata.json")


def make_log(row, metadata, retrieved_timestamp):
    """Preserve one task/configuration per log, including same-model interfaces."""
    cohort = metadata["cohorts"][row["cohort"]]
    model = metadata["models"][row["model"]]
    counts = {
        key: int(row[key])
        for key in (
            "total_n",
            "positive_n",
            "negative_n",
            "no_decision_n",
            "tp",
            "fn",
            "fp",
        )
    }
    if any(v < 0 for v in counts.values()):
        raise ValueError("Negative count")
    if counts["total_n"] != counts["positive_n"] + counts["negative_n"]:
        raise ValueError("Class counts do not sum to total_n")
    if counts["tp"] + counts["fn"] != counts["positive_n"]:
        raise ValueError("Positive confusion counts do not match positive_n")
    if (
        counts["fp"] > counts["negative_n"]
        or counts["no_decision_n"] > counts["total_n"]
    ):
        raise ValueError("Count exceeds its denominator")
    if (counts["total_n"], counts["positive_n"], row["source_positive"]) != (
        cohort["total_n"],
        cohort["positive_n"],
        cohort["positive_label"],
    ):
        raise ValueError("Cohort definition differs from frozen metadata")
    denominator = 5 * counts["tp"] + 4 * counts["fn"] + counts["fp"]
    expected = 5 * counts["tp"] / denominator if denominator else 0.0
    score, low, high = (float(row[k]) for k in ("f2", "ci_low", "ci_high"))
    if not all(math.isfinite(v) for v in (score, low, high)):
        raise ValueError("Non-finite score or interval")
    if not math.isclose(score, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Saved F2 disagrees with confusion counts")
    if not 0 <= low <= score <= high <= 1:
        raise ValueError("Invalid score/interval bounds")
    identity = "/".join(
        row[k] for k in ("model", "arm", "cohort", "scheme", "rubric", "model_version")
    )
    details = {k: str(v) for k, v in counts.items()}
    details.update(
        positive_label=row["source_positive"],
        bootstrap_seed=row["bootstrap_seed"],
        no_decision_handling=(
            "No verdict on a positive example counts as a false negative; "
            "no verdict on a negative example is not a false positive. "
            "F2 is not strict label accuracy."
        ),
    )
    return EvaluationLog.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "evaluation_id": f"{COLLECTION}/{metadata['study_version']}/{identity}",
            "retrieved_timestamp": retrieved_timestamp,
            "source_metadata": {
                "source_name": "Decision-interface safety evaluation",
                "source_type": "evaluation_run",
                "source_organization_name": "Braintrust",
                "evaluator_relationship": "third_party",
                "additional_details": {
                    "source_code_commit": metadata["source_code_commit"],
                    "source_csv_sha256": metadata["source_csv_sha256"],
                    "study_version": metadata["study_version"],
                    "limitations": json.dumps(metadata["caveats"]),
                    "publication_status": "prepared; public source and release license pending",
                },
            },
            "eval_library": {
                "name": "classifier_eval",
                "version": metadata["source_code_commit"],
                "additional_details": {"registry_status": "unverified_custom_harness"},
            },
            "model_info": {
                "name": row["model"],
                "id": model["id"],
                "developer": model["developer"],
                "inference_platform": {
                    "jev_choice": "TypeSafe",
                    "jev_noul": "TypeSafe",
                    "clef_choice": "Cloudflare Workers AI",
                    "clef_flash_choice": "Cloudflare Workers AI",
                    "decisions_luna": "OpenAI",
                    "luna_structured": "Braintrust AI Gateway",
                    "claude_structured": "Braintrust AI Gateway",
                }.get(row["arm"], "Baseten"),
                "additional_details": {
                    "raw_model_id": row["model"],
                    "served_version": row["model_version"],
                    "interface": row["arm"],
                    "deployment_type": "self_deployed"
                    if row["provider"] == "baseten"
                    else "externally_managed",
                    "model_availability": model["model_availability"],
                    "registry_status": model["registry_status"],
                    "registry_revision": metadata["registry_revision"],
                },
            },
            "evaluation_results": [
                {
                    "evaluation_result_id": f"{row['cohort']}.f2",
                    "evaluation_name": f"decision_interface_safety.{row['cohort']}",
                    "source_data": cohort["source_data"],
                    "metric_config": {
                        "metric_id": "f2",
                        "metric_name": "F2",
                        "metric_kind": "f_beta",
                        "metric_unit": "proportion",
                        "metric_parameters": {"beta": 2},
                        "lower_is_better": False,
                        "score_type": "continuous",
                        "min_score": 0,
                        "max_score": 1,
                        "evaluation_description": cohort["description"],
                        "additional_details": {
                            "formula": "5*TP / (5*TP + 4*FN + FP)",
                            "registry_status": "unverified_metric_id_and_study_task_variant",
                            "scoring": "Deterministic comparison to published human labels; no LLM judge.",
                        },
                    },
                    "score_details": {
                        "score": score,
                        "details": details,
                        "uncertainty": {
                            "num_samples": counts["total_n"],
                            "num_bootstrap_samples": int(row["bootstrap_replicates"]),
                            "confidence_interval": {
                                "lower": low,
                                "upper": high,
                                "confidence_level": 0.95,
                                "method": "Percentile multinomial bootstrap of example-level confusion outcomes",
                            },
                        },
                    },
                    "generation_config": {
                        "additional_details": {
                            "arm": row["arm"],
                            "scheme": row["scheme"],
                            "rubric": row["rubric"],
                            "policy_text": cohort["policy_text"],
                            "option_definitions": json.dumps(
                                cohort["option_definitions"]
                            ),
                            "provenance": cohort["provenance"],
                            "policy_scope": "Shared semantic instructions; provider-specific request wrappers and output formats.",
                        },
                    },
                }
            ],
        }
    )


def convert_rows(rows, metadata, output_dir, retrieved_timestamp):
    outputs, failures, seen = [], [], set()
    for index, row in enumerate(rows, 1):
        try:
            log = make_log(row, metadata, retrieved_timestamp)
            if log.evaluation_id in seen:
                raise ValueError("Duplicate task/configuration identity")
            seen.add(log.evaluation_id)
            collection, developer, model = datastore_path_components(
                COLLECTION, log.model_info.id, log.model_info.developer
            )
            outputs.append(
                EvaluationLogOutput(log, output_dir / collection, developer, model)
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(SourceRecordFailure(f"CSV row {index}", str(exc), row))
    return SourceConversionResult(COLLECTION, len(rows), outputs, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="An empty directory named data"
    )
    args = parser.parse_args()
    if args.output_dir.name != "data" or (
        args.output_dir.exists() and any(args.output_dir.iterdir())
    ):
        parser.error(
            "Output must be an empty directory named data; use a fresh path for each export"
        )
    metadata = json.loads(args.metadata.read_text())
    if (
        hashlib.sha256(args.input_csv.read_bytes()).hexdigest()
        != metadata["source_csv_sha256"]
    ):
        parser.error("Input CSV differs from the reviewed source snapshot")
    with args.input_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != metadata["expected_rows"]:
        parser.error("Incomplete source table")
    result = convert_rows(rows, metadata, args.output_dir, str(time.time()))
    report = save_failure_report(
        result, default_failure_report_path(args.output_dir / COLLECTION)
    )
    result.raise_if_incomplete()
    paths = save_evaluation_logs(result.records)
    print(f"Exported {len(paths)} records; coverage report: {report}")


if __name__ == "__main__":
    main()
