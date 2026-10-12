# Decision-interface safety eval

Implementation of [the study design](decision-interface-safety-eval-design.md). The active benchmark uses the published human labels on original Aegis and WildGuardTest prompt-response pairs. No new human labeling is required for these supported tasks.

The full hosted source-label run completed and verified all 66,566 requests across 5,252 local Aegis, WildGuardTest, and XSTest records. PolicyLM completed its 2,131 prompt-only comparisons. Baseten completed all 56,076 predictions plus 14,900 derived native judgments. All 66 Baseten configurations passed the final score/count/input/output/source-label/dataset audit, and all six GPU deployments were verified inactive with zero replicas. Together, the source-reference runs contain 124,773 physical predictions. The October 12 recovery corrected the deployed Tev/Qwen context limits and repaired three derived-result uploads without repeating completed model calls. The earlier Aegis validation check reused 1,400 verified predictions (200 per interface). HH-RLHF full conversations remain supplemental diagnostics because chosen/rejected preferences are not safety gold. See the [source-label runbook](run/SOURCE_LABEL_RUNBOOK.md) for supported targets, filters, rerun commands, and current coverage.

Review the [study policy target](prompts/policy_target.md), four draft rubrics, and their shared label definitions in the [rubric review index](prompts/README.md). They are local drafts, not frozen Braintrust prompt versions.

The review packet is [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md); a printable copy with tick boxes and write-in lines is [REVIEW_CHECKLIST.pdf](REVIEW_CHECKLIST.pdf). Regenerate it after editing the Markdown with `python3 review/build_printable.py` (needs pandoc and Chrome).

## Source data build

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.fetch_sources
.venv/bin/python data/build_datasets.py
.venv/bin/python data/build_datasets.py --push
.venv/bin/python -m pytest -q --ignore=data/processed
```

Braintrust access always loads `BRAINTRUST_API_KEY` from this repository's `.env`. `BASETEN_API_KEY` in the same file serves the six dedicated deployments (`deploy/baseten/`) and Baseten Loops training. Their model IDs are recorded in `.env`; the verified Tev1/Qwen deployment IDs are pinned there as well. Keys are never stored in the manifest or sent to stdout. Published dataset names include their content hash. `run_manifest.json` holds the returned Braintrust dataset versions and is ignored by Git because it is run-specific. Re-running `--push` republishes only datasets whose content hash changed; unchanged datasets keep their recorded name and version, so earlier experiments stay pinned to the data they ran on.

## Source audit and design implications

The pinned [Aegis V2 card](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/blob/main/README.md) describes a nine-column release with **binary** human `prompt_label`, `response`, `response_label`, and aggregate `violated_categories`. The [paper](https://aclanthology.org/2025.naacl-long.306.pdf) says the human vote was made at conversation level and assigned to `prompt_label`; separate response labels are not human gold. The builder therefore sends both turns on paired rows and scores the human conversation vote, while preserving response-label provenance and leaving unsupported fine gold and agreement fields null. Paired interactions are primary; one-turn Aegis records are secondary.

The [original XSTest prompt CSV](https://github.com/paul-rottger/xstest/blob/main/xstest_prompts.csv) supplies 450 binary-labeled prompts and is CC BY 4.0. Unsafe XSTest examples have no Aegis category gold. Aegis is also marked CC BY 4.0 on its dataset card.

[WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) (`wildguardtest` split, 1,725 prompt-response pairs, three human annotators each, ODC-BY) is the second primary test source. The builder preserves both turns and both human harm labels. It derives conversation harm as unsafe if either prompt or response was labeled harmful; that OR is a study-derived comparator, not a source field. The `adversarial` flag defines strata, while `subcategory` remains a candidate label only. The repo is gated: accept the AI2 terms for the **dataset** repo under the Hugging Face account whose token is in `~/.cache/huggingface/token`, then run `python -m data.fetch_sources`. Accuracy is reported per paired source before any paired-only pooled result.

The revised local `dis-test` has 4,057 records: 813 paired Aegis conversations, 1,681 paired WildGuardTest conversations, 1,113 one-turn Aegis records, and 450 XSTest controls. Eight additional WildGuardTest pairs are reserved in `dis-pilot` and excluded from test and controls. Separate [prompt-harm](prompts/rubrics/prompt_harm_s2.md), [response-harm](prompts/rubrics/response_harm_s2.md), and [refusal](prompts/rubrics/response_refusal_s2.md) questions use the complete paired interaction as context and judge distinct decisions.

The scheme names encode approximate granularity: `s6` has 7 options and `s13` has 15. The interaction grid has 151 required cells, or 199 with all three optional arms; the three fixed WildGuardTest companion decisions add 33 binary cells, plus one supplemental PolicyLM User-only cell. Inspect that plan with `.venv/bin/python -m run.matrix --companions`. The main grid remains gated until adapter contracts, rubric versions, and a preregistration are resolved.

## Diagnostic paired pilots

The eight reserved WildGuard pairs were evaluated with Jev Choice, Jev Noul, Claude, ordinary Luna structured output, Tev1, and its Qwen base. Choice, structured, and letter arms used four interaction rubric conditions plus separate User-harm, Assistant-harm, and refusal questions; Noul is binary interaction-only. Results and methodological caveats are in the [pilot report](run/pilot-report-2026-10-01.md). Jev's native three-question request was also tested on the same pairs. Clef and Clef-flash passed binary, 15-option Long, full paired, and batched-question pilots on 2026-10-02 with updated Workers AI credentials. Safeguard, Granite, Qwen3Guard, and Nemotron v3 have Baseten deployments and diagnostic paired pilots; all project Baseten deployments are currently deactivated. The direct OpenAI Decisions API arm (`decisions_luna`) was added on 2026-10-06; the historical pilot above predates that integration. It is now supported by the pilot runner and matrix, including batched turn questions.

## Clef and the Jev Decision Index

The [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) is a broad fixed benchmark of decision models. [Cloudflare's Clef announcement](https://blog.cloudflare.com/clef-decision-models/) already reports Clef and Clef-flash there. This study first measures **agreement with interaction-level harm labels** under a declared policy, using full user–assistant pairs when available. It then asks how decisions and label agreement change when the offered labels, policy information, rubric length, or option order change on the *same interactions*. Prompt-only XSTest remains a separate benign-lookalike control. Native fixed-policy guards are external references. Index scores are background context, not an outcome or validation target.

`clef_choice` and `clef_flash_choice` are included in the matrix and have a tested request/response adapter in [adapters/clef.py](adapters/clef.py). Live smoke tests need a Cloudflare Workers AI account ID and auth token, which are not present in `.env`; Braintrust operations always use that file's `BRAINTRUST_API_KEY`. The hosted Clef revision and 15-option behavior still require verification before a pilot.

## Jev smoke

The Braintrust gateway does not expose Jev to this organization's API key, so the Jev adapter uses the native [TypeSafe System One API](https://docs.typesafe.ai/api) with `TYPESAFE_API_KEY` from this repository's `.env`. Braintrust logging still uses `BRAINTRUST_API_KEY` from the same file.

Run `.venv/bin/python -m adapters.smoke_jev` to classify a balanced 20-item subset of `dis-dev` and log it to Braintrust. The 2026-10-01 smoke of pinned `jev-1.13.0` returned 20/20 valid Choice responses and 18/20 correct binary labels. The prompt is explicitly provisional, so this result checks adapter behavior rather than estimating final model performance. The experiment is [here](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/experiments/jev_choice__s2__smoke_provisional__20261001T191055Z).

## Tev1, its Qwen3.5-4B base, and the mini safety fine-tune

### Model fields and experiment organization

All experiment runners route through `run/experiment_groups.py`. Each Braintrust
experiment contains exactly one model/interface (`arm`), target, label scheme, and
rubric. The shared `run_group` identifies the study stage; it is not an experiment.
Repeatability replicates remain together within the same stage/configuration.
`metadata.model` uses the canonical model name from `run/model_metadata.py`.
Every span also records `arm` and `model`; `model_version` retains the served
revision. Register new arms in the shared mapping before running them. Runner
checkpoints retain the run-group name and an `experiment_urls` map for the children.

Primary source-labeled configurations set `metadata.study_status = "current"` and `benchmark_primary = true`. HH full-context, unsupported Aegis companion targets, repeatability, and batching checks are `supplemental`; other historical copies are `legacy`. Originals remain `aggregate_archive` with records and names preserved. The saved **Source-labeled safety benchmark** view selects Aegis validation interaction harm and charts `source_accuracy`. Filter by both `source_dataset` and `target` before comparing scores; do not pool unrelated tasks or datasets. The **Model** column reads `metadata.model` from the root span.

`python -m run.maintain_experiments` prepares a backup and metadata-only maintenance
plan in `data/processed/experiment-maintenance/`. Apply it with
`python -m run.maintain_experiments --apply <directory>`. Maintenance makes no
inference calls and verifies that inputs, outputs, expected values, scores,
metrics, tags, and span IDs are unchanged. Logging code changes require a new
plan/freeze for future runs; do not rewrite the hashes of previously executed plans.

`python -m run.split_experiments` backs up current records and prepares exact
configuration copies, using pinned job manifests where trace metadata lacks a
dimension. Apply with `python -m run.split_experiments --apply <directory>`.
This copies existing results without inference, verifies content and trace identity,
records source provenance, and only archives an original after all copies pass QA.
Interrupted migrations can resume from their receipt using the same command.

`tev1_native` and `qwen35_4b_letters` share [adapters/tev1.py](adapters/tev1.py), which follows the published Tev1 contract (system prompt, JSON `state`/`question`/`options`, one letter back, letter log-probs turned into a distribution). Both run from matching Baseten dedicated vLLM deployments in [deploy/baseten/](deploy/baseten/) so the fine-tune-vs-base contrast is not confounded with the serving stack; `provider="together"` is a serverless fallback. HF revisions are pinned in the adapter.

The optional `safety_tev` arm applies the Tev1 LoRA recipe to paired Aegis V2 `train` records. `python -m training.build_safety_sft` targets 4,000 train / 400 dev examples across schemes and question variants, with normalized-prompt overlap checks against eval sources and a 2,048-token request limit. The earlier `yqvek93` checkpoint was trained on prompts alone and is obsolete for the paired-interaction comparison. Rebuild after rubric FREEZE and retrain before including this arm. Unsafe fine-grained labels from `violated_categories` are training supervision only.

## Claude via the Braintrust AI gateway, and the constitutional-classifier reproduction

All Anthropic calls go through the Braintrust AI gateway (`https://api.braintrust.dev/v1/proxy/anthropic`) with the Braintrust key, gateway cache disabled per request. `claude_structured` ([adapters/claude.py](adapters/claude.py)) is a general-LLM arm: Claude Sonnet 5.5, low effort, structured output with an enum of option ids, no refusal fallbacks so the served model never changes inside the arm.

Anthropic's Constitutional Classifiers have no public weights or API, so `cc_repro` reproduces the *method*: [prompts/constitution.md](prompts/constitution.md) → `python -m training.generate_constitutional --launch` (Claude generates labelled prompts per class with near-miss, adversarial, and multilingual variants, deduped against every eval text and against Aegis train) → `python -m training.train_loops --launch --data training/data/cc-mini/instruction --name cc-repro-mini`. Because it never sees Aegis, XSTest, or WildGuard, it is out-of-distribution and may enter the accuracy contrasts, unlike `safety_tev`.

PolicyLM-1.7B is now a separate prompt-only custom-policy reference. It uses the
User message alone, a pinned release, native precision cutoff 0.335, and local
float32 inference in an isolated environment. Its authors exclude conversation
history and assistant moderation, so it does not enter the paired interaction
grid. See the [adapter setup and exposure caveats](adapters/README.md#policylm-prompt-only-reference).

## Historical pair-only completion status — 2026-10-07

The 5,000-call development reference and the 208-prediction PolicyLM prompt-only supplement are complete. The subsequent 2,000-call repeatability check completed with 1,984 valid classifications and 16 Claude refusals; there were no API errors. All 160 batched calls returned valid results (480 decisions). Compared with separate reference calls, batching changed 5/120 Clef labels and 9/120 Clef-flash labels; Jev and Decisions Luna changed 0/120. These are development diagnostics, not held-out performance.

Reproduce the local summary with `.venv/bin/python -m analysis.development_report`. Frozen stage artifacts and Braintrust experiment links are recorded under `data/processed/study-repeatability/477ac56d553a` and `data/processed/study-batching/4424eb16d43d`.

The blinded 200-pair audit dataset is uploaded as `human_policy_audit_blind__f0c7bbda4b65`; All 200 items are assigned to Izzy Hurley for native Braintrust annotation; actual judgments remain pending. These historical custom-policy assignments are optional supplemental work, not required for the source-label benchmark. Report these as single-reviewer policy judgments, with no inter-rater agreement or adjudication statistics. Use [native Braintrust review](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/datasets/human_policy_audit_blind__f0c7bbda4b65), with six configured human fields writing to expected. The local annotation packet remains a backup.

The user authorized inclusion of the Baseten models. `run/baseten_preflight.py` limits restarts to six explicit deployment IDs, checks hardware/replica bounds, waits for readiness, runs two-pair diagnostics, and deactivates deployments on completion or timeout. A cold-start attempt failed; its result is retained as failed, not usable evaluation evidence. The readiness-gated retry timed out waiting for Safeguard; no retry pilot calls were made. All six deployments were then verified inactive with zero replicas. This describes the historical pilot attempts. The October 9 source-label deployment builds completed; the four longer-context guard deployments use 8,192-token limits.

The held-out candidate at `data/processed/heldout-candidate/55a1f8b24835` retains full original conversations and quarantines seven development-exposed pairs, leaving 2,487 candidate primary pairs. It is not preregistered or launched. Its full proposed matrix contains 477,526 cell decisions, including native derived decisions and the local PolicyLM supplement; this is not a physical API-call count. Full serving-context coverage, physical-call costs, a frozen analysis plan, and human review remain unresolved. The 24 examples affected by the provisional length filter have not been silently dropped. Development planning reserves total $811.03; three bounded Baseten preflight reserves add $75. These reserves are assumptions, not confirmed charges. This historical robustness-grid candidate remains unlaunched. The October 9 binary source-reference run uses the separate additional $1,000 authorization. Recorded usage and public prices are estimates; reservations are not invoices. Decisions now has a verified $0.10/M-input rate; gateway fees remain unverified.

### Development analysis and charts

The [development report](analysis/output/development-2026-10-07/report.md) contains six verified PNG/SVG charts covering source-label agreement, harm tradeoffs, repeatability, batching, output reliability, and observed request latency. It includes matched-item bootstrap comparisons and explicitly separates the source-exposed PolicyLM supplement from the primary hosted-arm comparison. No independent human audit or held-out results are claimed.

The [full-source descriptive report](analysis/output/source-2026-10-12/report.md) covers all 124,773 verified physical predictions and 14,900 reused native judgments. Nine PNG/SVG charts compare hosted token cost and accuracy, matched safety accuracy, judgment types, latency, token use, model disagreement, unsafe recall/false alarms, a separately labeled dedicated-GPU cost scenario, and dataset/input-unit variation. CSV tables include every configured scorer's definition, denominator, missingness, mean, sample standard deviation and percentiles, plus physical-call cost/latency/token distributions. Regenerate with `.venv/bin/python -m analysis.source_report`; the local output also includes a compressed tidy table and `safety-analysis.zip`. Current public-rate estimates and the historical October 9 snapshot remain separate; dedicated GPU allocation is not invoice-reconciled spend.

The [detailed failure report](analysis/output/failures-2026-10-12/report.md) adds eight charts, class-conditional slice tables, paired Tev/Qwen wins and losses, cross-turn consistency, probability/operating-point tradeoffs, high-confidence mistakes, and archived execution failures. `.venv/bin/python -m analysis.failure_report` creates the self-contained local `failure-explorer.html` and `failure-analysis.zip` for 10,294 distinct source-item/task cases, with full original context and all available model decisions. The local explorer contains benchmark text and is generated rather than committed. Source tags and wording cues are exploratory slices, not newly adjudicated scenario labels; the rejected Topics backfill is not required or run. Validate the actual JavaScript filter predicates offline with `node analysis/test_failure_filters.cjs` after generation. Browser preview of local HTML was blocked by the browser tool's URL policy; counts, JavaScript syntax and filter behavior were checked offline.

Install plotting dependencies with `.venv/bin/python -m pip install -r analysis/requirements.txt`, then regenerate with `.venv/bin/python -m analysis.make_development_charts`. The output directory also contains tidy decision records, metric tables, paired differences, and a JSON summary with source hashes.

The [exploratory inferential report](analysis/output/development-2026-10-07/inference/report.md) adds paired t-tests with Holm correction, exact McNemar sensitivity tests, a frequentist logistic random-intercept model for strict source agreement, an arm-by-source-label extension, and a log-latency linear mixed model. Full tables, convergence checks, numerical verification, and coefficient charts are included. Regenerate with `.venv/bin/python -m analysis.inferential_models`; the binary likelihood is checked against independent adaptive integration in `analysis/test_inferential_models.py`.

### First-wave preparation

The checkpointed runner and freeze validator are implemented in `run/heldout.py` and `run/freeze_heldout.py`. The binary long-rubric first wave retains 2,487 pairs and schedules 39,792 physical calls; native guard interaction ORs add no inference calls. The separate local PolicyLM supplement has 2,487 User-only requests prepared. See the [launch runbook](run/HELDOUT_RUNBOOK.md) and [draft analysis plan](analysis/heldout-analysis-plan.md). No held-out predictions have been made.

The five-model Baseten preflight at `data/processed/baseten-preflight/20261007T200841Z` completed with no failed pilot outputs. All five were verified inactive with zero replicas; Safeguard remains unresolved and inactive. Matching Tev/Qwen configs now propose 8,192-token contexts, but those changes are not deployed or validated. Full serving-context coverage, completed human audit, deployment readiness and billing reconciliation remain launch requirements.
