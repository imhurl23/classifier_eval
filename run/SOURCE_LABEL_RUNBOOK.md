# Source-labeled safety benchmark

The primary benchmark uses labels already published with the original prompt-response pairs. No new manual annotation is needed for supported tasks.

| Source | Supported gold |
| --- | --- |
| Aegis V2 paired validation | Human conversation/interaction binary harm |
| WildGuardTest | Human prompt harm, response harm, response refusal; interaction harm derived as OR of prompt/response harm |
| HH-RLHF full conversations | No safety gold; preferences remain supplemental diagnostics |

Do not transfer pair labels to expanded HH conversations, infer safety from chosen/rejected preferences, or score separate Aegis response/refusal targets as human gold. Unsupported scores stay absent.

## Current coverage

On 2026-10-09, 1,400 existing Aegis predictions were verified against their pinned requests and restored as seven primary single-interface experiments with 200 identical validation examples each. No inference was repeated. Inputs, outputs, span identity, and historical results were preserved; source-label provenance, dataset bindings, strict `source_accuracy`, and recorded token usage were added.

A fresh WildGuard scoring check completed 105 requests across 21 configurations on five identical examples. All calls completed. This small, deterministic stratified sample checks scoring and logging; it cannot support a reliable model ranking or a full-population claim. Native prompt/response/refusal questions remain distinct; derived interaction OR is explicitly labeled. Decisions Luna retains its existing Aegis comparison; it was omitted from new calls because endpoint-specific pricing is unresolved.

The new plan is `data/processed/source-labeled-benchmark/ce38af926b4d`; the run group is `source_labeled_wildguard__ce38af926b4d__20261009T154435Z`. The additional allocation was $25 within the previously authorized $1,000 envelope, with a $13.84 conservative request reserve and a separately retained infrastructure allowance. These are planning allowances, not reconciled invoice charges.

## Braintrust filters and scores

Open [Source-labeled safety benchmark](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/experiments?v=Source-labeled+safety+benchmark). Its saved filters select Aegis validation interaction harm. For another comparison, use all three filters:

```sql
metadata.study_status = 'current'
metadata.source_dataset = 'wildguardtest'
metadata.target = 'prompt_harm'
```

Other WildGuard targets: `response_harm`, `response_refusal`, `interaction_harm`. For Aegis: source dataset `aegis_v2_validation`, target `interaction_harm`. Compare models on the same source, target, and cohort. Group by `arm` to keep different interfaces for the same model separate.

Use `source_accuracy` as strict agreement with source gold. Invalid output and classifier refusal count as incorrect. `valid_output` and `refused` describe classifier behavior; they are not safety accuracy. Braintrust's “All scores (avg)” averages different quantities and must not be used as a model-quality metric. Probability scores are available only when the adapter returns probabilities; recall may be absent when the cohort contains no positives. Cost remains absent until billing is reconciled.

Primary configurations are `current`. HH, unsupported Aegis companion targets, repeatability, and batching checks are `supplemental`. Historical copies are `legacy`, and preserved originals are `aggregate_archive`. Each experiment contains one model/interface, target, scheme, and rubric; Model is filled from `metadata.model`.

## Prepare and execute future runs

```sh
.venv/bin/python -m run.source_benchmark --prepare --budget-usd 25 --max-items 200
.venv/bin/python -m run.source_benchmark --plan <returned-plan-directory> --wall-minutes 60
```

Preparation freezes code, prompts, selected original inputs, and exact requests. It checks the study's remaining planning allocation, excludes prior pilot/development exposure, uses the same cohort for every applicable interface, and reserves retries before inference. A budget may fit only a small scoring check. Never rewrite executed plan hashes.

Results and completion checkpoints are fsynced before logging to Braintrust. Completed jobs are never automatically repeated. `--resume` refuses ambiguous inflight calls until they are reconciled. The process lock prevents concurrent execution of the same plan. Future `run.development` calls also select scores from available source gold and mark unsupported configurations supplemental.

The existing restoration command is `.venv/bin/python -m run.restore_source_benchmark`; it validates the pinned Aegis prediction/request bindings, backs up records, and updates scores/provenance without inference. It does not demote new WildGuard primary experiments.

Run source checks with `.venv/bin/python -m pytest -q --ignore=data/processed`; frozen historical snapshots under processed data are artifacts, not another test suite.
