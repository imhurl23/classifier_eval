# Draft: decision-interface safety classification results

This contribution adds 62 model/interface/task F2 results from the completed `source_aligned_v1_20261019` study. It covers harmful requests, harmful responses, whole exchanges, and refusal detection against published Aegis, WildGuardTest, and XSTest labels. The six cohorts remain separate.

**Draft status:** public source URL, release license, and unresolved registry mappings must be completed before opening the contribution. Adapter PR, registry PR, and datastore PR links are not yet available.

## Coverage and interpretation

62 source-table rows → 62 aggregate logs; zero dropped, failed, or excluded rows. The rows represent 89,251 decisions across 11 interfaces and eight underlying model identifiers. The separately reported development diagnostics, fixed-policy supplements, historical runs, and custom fine-tuning comparison are not part of this source table.

The revision followed inspection of test disagreements. It is exploratory, not a fresh held-out or preregistered result. XSTest reuses the original policy and results. Labels were not changed to reconcile model disagreements.

F2 uses unsafe as the positive class for harm and refusal for refusal detection. It includes missing positive verdicts as misses; missing negative verdicts do not count as false alarms. The stored confidence intervals are the original 95% bootstrap intervals. F2 is not strict accuracy or an end-to-end agent safety measure.

## Conversion decision log

| Decision | Choice and reason | Confidence |
|---|---|---|
| Record granularity | One task/configuration per log; retains different interfaces of the same model and avoids pooling task semantics | High |
| Record identity | Stable raw source model + interface + cohort + scheme + rubric + served version + study version; canonical alias changes do not change identity | High |
| Artifact type | `evaluation_run`; original per-item run outputs exist locally, though this export contains only aggregates | High |
| Source provenance | Actual dataset revisions and filtered/pool definitions; XSTest marked reused, Aegis not falsely labeled test-only | High |
| Policy | Rendered main-task policy and option definitions retained; provider request wrappers are not claimed identical | High |
| Uncertainty | Copy saved intervals/counts; no new inferential tests or invented standard errors | High |
| Missing verdicts | Retain explicit count and F2 treatment; do not infer a valid negative prediction from no verdict | High |
| Model identity | Pinned reviewed registry matches where exact; unresolved raw IDs explicitly unverified | Medium; review required |
| Task/metric/harness identity | Study variants retained; registry reconciliation pending | Pending |
| Export scope | Aggregate-only main F2 table, without raw conversations, private traces, or case reviews | High |
| Timestamp | Record creation time only; no invented exact per-configuration evaluation timestamp | High |
| Publication | Source commit/hash preserved; public URL and license pending owner decision | Pending |

No schema, shared converter, base adapter, or validator changes are requested.

## Validation

- Pinned EEE schema and semantic CLI checks: 62 pass, zero warnings, zero errors.
- Complete source-row coverage; F2 recomputed from saved confusion counts and checked against each source score.
- Tests check stable/different interface identities, source provenance, confidence interval preservation, corrupt rows, and duplicate rejection.
- Generated package is separate from the working repository and excludes private operational artifacts.

Re-run local validation and the datastore PR bot after final registry and publication metadata edits. Schema validity alone is not a substitute for reviewing the unresolved identities above.
