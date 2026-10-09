"""Diagnostic Baseten guard pilots on the reserved eight paired interactions."""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments

import argparse
from collections import Counter
from datetime import datetime, timezone
import json

from adapters.base import Decision
from adapters.guards import BasetenGuard, granite, nemotron, qwen3guard, safeguard
from data.bt_auth import load_braintrust_key
from prompts.runtime import instructions_for, options_for
from run.matrix import matched_option_order
from run.pilot import load_local_service_keys, pilot_score, rows_for_pilot

ARMS = ("safeguard_120b", "granite_41_8b", "qwen3guard_8b", "nemotron_guard")


def union_decision(parts: dict[str, Decision]) -> Decision:
    if any(d.status != "ok" for d in parts.values()):
        return Decision(None, None, {k: d.raw for k, d in parts.items()}, "invalid_output",
                        sum(d.latency_ms for d in parts.values()))
    if any(d.label == "unsafe" for d in parts.values()):
        label = "unsafe"
    elif any(d.label == "needs_caution" for d in parts.values()):
        label = "needs_caution"
    else:
        label = "safe"
    return Decision(label, None, {"derived_from": {k: d.label for k, d in parts.items()},
                                  "calls": {k: d.raw for k, d in parts.items()}},
                    "ok", sum(d.latency_ms for d in parts.values()))


def run(arms: list[str], limit: int = 8):
    if any(a not in ARMS for a in arms):
        raise ValueError(f"Unsupported guard arm: {set(arms)-set(ARMS)}")
    import braintrust as bt
    load_local_service_keys()
    load_braintrust_key()
    experiment = init_experiments(project="decision-interface-safety",
                         experiment=f"guard_pilot__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                         metadata={**experiment_model_metadata(arms), "purpose": "diagnostic_pilot", "dataset": "dis-pilot",
                                   "preregistered": False, "arms": arms})
    counts = Counter()
    for arm in arms:
        guard = BasetenGuard(arm)
        for row in rows_for_pilot()[:limit]:
            prompt, response = row["input"]["prompt"], row["input"]["response"]
            decisions = {}
            if arm == "safeguard_120b":
                for target, rubric in (("interaction_harm", "r_short"),
                                       ("interaction_harm", "r_long"),
                                       ("interaction_harm", "r_neutral_long"),
                                       ("interaction_harm", "r_withheld"),
                                       ("prompt_harm", "prompt_harm_s2"),
                                       ("response_harm", "response_harm_s2"),
                                       ("response_refusal", "response_refusal_s2")):
                    labels, definitions = options_for("s2", rubric)
                    decisions[(target, rubric)] = safeguard(
                        guard, row["input"]["text"], instructions_for(rubric),
                        definitions, matched_option_order(row["id"], target, "s2", labels))
            elif arm == "granite_41_8b":
                parts = {target: granite(guard, prompt, response,
                                         instructions_for(f"{target}_s2") + "\nUnsafe criterion: " +
                                         options_for("s2", f"{target}_s2")[1]["unsafe"], target)
                         for target in ("prompt_harm", "response_harm")}
                decisions[("interaction_harm", "r_short")] = union_decision(parts)
                for target, d in parts.items():
                    decisions[(target, f"granite_byoc_{target}")] = d
            elif arm == "qwen3guard_8b":
                parts = {"prompt_harm": qwen3guard(guard, prompt),
                         "response_harm": qwen3guard(guard, prompt, response)}
                decisions[("interaction_harm", "native_s3")] = union_decision(parts)
                for target, d in parts.items():
                    decisions[(target, "native_s3")] = d
            else:
                parts = nemotron(guard, prompt, response)
                decisions[("interaction_harm", "native_s2")] = union_decision(parts)
                for target, d in parts.items():
                    decisions[(target, "native_s2")] = d
            for (target, rubric), decision in decisions.items():
                scores = pilot_score(decision, row, target) if decision.label != "needs_caution" else {}
                counts[(arm, target, rubric, decision.status)] += 1
                experiment.log(id=f"{arm}__{target}__{rubric}__{row['id']}",
                               input=row["input"], expected=row["expected"],
                               output={"label": decision.label, "status": decision.status},
                               scores=scores,
                               metadata={**model_metadata(arm), "target": target, "rubric": rubric,
                                         "model_version": guard.model_version,
                                         "dataset_record_id": row["id"],
                                         "raw": decision.raw, "usage": decision.usage},
                               metrics={"latency_ms": decision.latency_ms})
    experiment.flush()
    result = {"experiment_url": experiment.summarize().experiment_url, "experiment_urls": experiment.experiment_urls,
              "pilot_items": limit,
              "counts": {"|".join(k): v for k, v in counts.items()}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.arms, args.limit), indent=2))
