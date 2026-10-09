"""Diagnostic paired-interaction pilot. Never draws from the held-out dis-test set.

Run: .venv/bin/python -m run.pilot --arms jev_choice claude_structured
The eight dis-pilot records are reserved from WildGuardTest before the main study.
"""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from adapters.clef import ClefChoiceAdapter
from adapters.jev import JevChoiceAdapter, JevNoulAdapter
from adapters.tev1 import LettersAdapter
from data.bt_auth import load_braintrust_key
from prompts.runtime import instructions_for, options_for
from run.matrix import matched_option_order
from scorers.scorers import score_item, score_refusal

ROOT = Path(__file__).resolve().parents[1]
TASKS = (("interaction_harm", "r_short"),
         ("interaction_harm", "r_long"),
         ("interaction_harm", "r_neutral_long"),
         ("interaction_harm", "r_withheld"),
         ("prompt_harm", "prompt_harm_s2"),
         ("response_harm", "response_harm_s2"),
         ("response_refusal", "response_refusal_s2"))
SUPPORTED = {"jev_choice", "jev_noul", "clef_choice", "clef_flash_choice",
             "claude_structured", "luna_structured", "tev1_native", "qwen35_4b_letters",
             "qwen35_4b_json", "decisions_luna"}


def load_local_service_keys():
    """Populate optional provider credentials from this repo's untracked .env."""
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in {"BASETEN_API_KEY", "BASETEN_TEV1_MODEL_ID",
                    "BASETEN_QWEN35_MODEL_ID", "BASETEN_QWEN35_DEPLOYMENT_ID",
                    "BASETEN_TEV1_DEPLOYMENT_ID", "BASETEN_SAFEGUARD_MODEL_ID",
                    "BASETEN_GRANITE_MODEL_ID", "BASETEN_QWEN3GUARD_MODEL_ID",
                    "BASETEN_NEMOTRON_MODEL_ID", "CLOUDFLARE_ACCOUNT_ID",
                    "CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_API_KEY"}:
            os.environ[name] = value.strip().strip('"').strip("'")
    if "CLOUDFLARE_AUTH_TOKEN" not in os.environ and "CLOUDFLARE_API_KEY" in os.environ:
        os.environ["CLOUDFLARE_AUTH_TOKEN"] = os.environ["CLOUDFLARE_API_KEY"]


def rows_for_pilot():
    rows = [json.loads(line) for line in (ROOT / "data/processed/dis-pilot.jsonl").open()]
    if len(rows) != 8 or any(r["input"]["unit"] != "conversation" for r in rows):
        raise ValueError("Expected eight reserved paired pilot rows")
    return rows


def adapter_for(arm: str, definitions: dict[str, str]):
    load_local_service_keys()
    if arm == "jev_choice":
        return JevChoiceAdapter(definitions)
    if arm == "jev_noul":
        return JevNoulAdapter(definitions)
    if arm in {"clef_choice", "clef_flash_choice"}:
        return ClefChoiceAdapter(arm, definitions)
    if arm == "claude_structured":
        from adapters.claude import ClaudeStructuredAdapter
        load_braintrust_key()
        return ClaudeStructuredAdapter(definitions)
    if arm == "luna_structured":
        from adapters.luna import LunaStructuredAdapter
        return LunaStructuredAdapter(definitions)
    if arm == "decisions_luna":
        from adapters.openai_decisions import OpenAIDecisionsAdapter
        return OpenAIDecisionsAdapter(definitions)
    if arm in {"tev1_native", "qwen35_4b_letters"}:
        return LettersAdapter(arm, definitions)
    if arm == "qwen35_4b_json":
        from adapters.qwen_json import QwenJsonAdapter
        return QwenJsonAdapter(definitions)
    raise ValueError(f"Unsupported pilot arm {arm}")


def pilot_score(decision, row, target):
    if target == "response_refusal":
        scores = score_refusal({"status": decision.status, "label": decision.label}, row["metadata"])
        return {k: v for k, v in scores.items() if v is not None}
    output = {"status": decision.status, "label": decision.label, "probs": decision.probs}
    scores = score_item(output, row["expected"], row["metadata"], "s2", target)
    return {k: v for k, v in scores.items() if v is not None}


def run(arms: list[str], limit: int, log: bool = True,
        targets: set[str] | None = None):
    if any(arm not in SUPPORTED for arm in arms):
        raise ValueError(f"Pilot adapter unavailable: {set(arms) - SUPPORTED}")
    rows = rows_for_pilot()[:limit]
    experiment = None
    if log:
        import braintrust as bt
        load_braintrust_key()
        experiment = init_experiments(project="decision-interface-safety",
                             experiment=f"paired_pilot__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                             metadata={**experiment_model_metadata(arms), "purpose": "diagnostic_pilot", "preregistered": False,
                                       "dataset": "dis-pilot", "arms": arms})
    summary = Counter()
    for arm in arms:
        for target, rubric in TASKS:
            if targets is not None and target not in targets:
                continue
            if arm == "jev_noul" and target != "interaction_harm":
                continue
            labels, definitions = options_for("s2", rubric)
            instructions = instructions_for(rubric)
            adapter = adapter_for(arm, definitions)
            for row in rows:
                cell = f"{arm}__{target}__{rubric}__pilot"
                order = matched_option_order(row["id"], target, "s2", labels)
                decision = adapter.classify(row["input"]["text"], "s2", instructions, order)
                scores = pilot_score(decision, row, target)
                summary[(arm, target, rubric, decision.status)] += 1
                if experiment:
                    experiment.log(id=f"{cell}__{row['id']}", input=row["input"],
                                   expected=row["expected"],
                                   output={"label": decision.label, "probs": decision.probs,
                                           "status": decision.status}, scores=scores,
                                   metadata={**model_metadata(arm), "target": target, "rubric": rubric,
                                             "dataset_record_id": row["id"],
                                             "option_order": order, "option_order_protocol": "matched_v1",
                                             "source_metadata": row["metadata"],
                                             "raw": decision.raw, "usage": decision.usage,
                                             "model_version": adapter.model_version},
                                   metrics={"latency_ms": decision.latency_ms})
    if experiment:
        experiment.flush()
        url = experiment.summarize().experiment_url
    else:
        url = None
    result = {"experiment_url": url, "experiment_urls": experiment.experiment_urls if experiment else {}, "pilot_items": len(rows),
              "counts": {"|".join(key): value for key, value in summary.items()}}
    out = ROOT / "data/processed/paired_pilot_summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_batched_turns(arms: list[str], limit: int):
    """Check native multi-question paths on reserved pairs."""
    if any(a not in {"jev_choice", "clef_choice", "clef_flash_choice", "decisions_luna"} for a in arms):
        raise ValueError("Batched typed Choice requires Jev, Clef, or Decisions Luna")
    import braintrust as bt
    load_braintrust_key()
    experiment = init_experiments(project="decision-interface-safety",
                         experiment=f"batched_turn_pilot__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                         metadata={**experiment_model_metadata(arms), "purpose": "multi_question_diagnostic", "dataset": "dis-pilot",
                                   "preregistered": False, "arms": arms})
    summary = Counter()
    for arm in arms:
        adapter = adapter_for(arm, {"safe": "safe", "unsafe": "unsafe"})
        for row in rows_for_pilot()[:limit]:
            questions = {}
            for target, rubric in TASKS[-3:]:
                labels, definitions = options_for("s2", rubric)
                order = matched_option_order(row["id"], target, "s2", labels)
                questions[target] = {"instructions": instructions_for(rubric),
                                     "criteria": {label: definitions[label] for label in order}}
            results = adapter.classify_many(row["input"]["text"], questions)
            for target, decision in results.items():
                scores = pilot_score(decision, row, target)
                summary[(arm, target, decision.status)] += 1
                experiment.log(id=f"{arm}__{target}__{row['id']}", input=row["input"],
                               expected=row["expected"],
                               output={"label": decision.label, "probs": decision.probs,
                                       "status": decision.status}, scores=scores,
                               metadata={**model_metadata(arm), "target": target, "rubric": "batched_turn",
                                         "dataset_record_id": row["id"],
                                         "question_bundle": list(questions),
                                         "option_order": list(questions[target]["criteria"]),
                                         "option_order_protocol": "matched_v1",
                                         "raw": decision.raw, "usage": decision.usage})
    experiment.flush()
    return {"experiment_url": experiment.summarize().experiment_url, "experiment_urls": experiment.experiment_urls,
            "pilot_items": limit, "counts": {"|".join(k): v for k, v in summary.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["jev_choice", "claude_structured"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--targets", nargs="+", choices=sorted({t for t, _ in TASKS}))
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--batched-turns", action="store_true",
                        help="Use one native typed-choice request per pair for three separate questions")
    args = parser.parse_args()
    result = (run_batched_turns(args.arms, args.limit) if args.batched_turns else
              run(args.arms, args.limit, not args.no_log,
                  set(args.targets) if args.targets else None))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
