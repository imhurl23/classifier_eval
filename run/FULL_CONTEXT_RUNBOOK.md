# Original full-conversation study

The user selected original complete upstream conversations and original Assistant replies. `study_revision.json` is the active revision. Legacy pair-only results and annotations remain preserved, and the old runners reject their superseded plans.

## Data and review

Pinned HH-RLHF revision: `c72f5cee8eb7b4d2ea5617657f4430d5e333af07`, `harmless-base` train/test. The new builder selects one original preference branch per conversation group deterministically. Branch preference is not a safety label. No separately generated Aegis reply is spliced into history.

`data/processed/full-conversations/92f8046d8b14` contains 200 development conversations and 1,630 held-out candidates. Eleven test conversations with earlier development User-text exposure are quarantined. Original two-turn controls are separate. All 1,830 primary records were independently checked against source row, original transcript hash, all turns and original replies; development/test groups are disjoint. See `provenance-verification.json`.

Annotate [the new Braintrust dataset](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/datasets/human_full_conversation_audit__966835c91784) using its **Review** button. The sidebar Review page lists assigned experiment/log spans and is a different surface. The new dataset contains 200 dialogues with 4–32 turns and withholds source labels and model predictions.

Read every numbered turn. `human_interaction_harm` covers every observed turn. `human_prompt_harm` targets the marked final User turn; `human_response_harm` and `human_response_refusal` target the marked final Assistant turn, using all history. Record consequential ambiguity and explain unrateable cases. One reviewer means no inter-rater agreement or adjudication claim. Existing Aegis labels and HH preferences cannot supply these new safety judgments.

## Replacement execution

The completed seven-arm smoke is `data/processed/full-context-rerun/13f1d7b89069`: 112/112 valid classifications. Do not rerun completed paid calls.

The original known-price plans are:

| Stage | Plan directory | Calls |
| --- | --- | ---: |
| Reference | `bb9c7247bc35` | 4,200 |
| Repeatability | `dfd15281531e` | 1,680 |
| Final-turn companions | `581da8ff4e54` | 2,400 |

Directories are under `data/processed/full-context-rerun`. Inspect `execution.json` and `results.jsonl` for authoritative progress and failures. Input/wire/code hashes are frozen. Do not change the files named in a live plan's `file_sha256`. The runner checkpoints each completed result before logging, uses an exclusive lock, and refuses an ambiguous paid-call resume. Three consecutive failed outputs stop that arm. A partial stage stops the sequential chain for inspection. Invalid outputs and refusals remain failures; never discard them to improve reported agreement.

Cloudflare exhausted its daily free allocation during reference Clef execution. That arm stopped after three recorded API errors. Reference continues for the other arms. The two subsequent unexecuted plans are partitioned without changing job IDs or wire payloads: repeatability `748ad8638daf` (1,040 available calls) plus `6d0c98bfeb36` (640 pending Clef calls), and final-turn `53d540287282` (1,200 available calls) plus `7a70802265ed` (1,200 pending Clef calls). Available partitions are queued after reference stops, provided there are no ambiguous paid calls. Original parent plans remain unexecuted and must not also be launched. Partition reservations sum to their parent's reservation. See each parent's `quota-partitions.json`. Resume only uncompleted reference jobs after quota is restored; preserve its recorded failed outputs.

The full-history payload is used for every supported hosted arm and target. Validity is scored now; accuracy waits for human safety judgments on these exact inputs. No source preference or old pair-only gold is used.

## Outstanding study components

- Decisions completed its 16-call full-history smoke. Larger replacements await its endpoint billing reconciliation within the existing cap.
- Baseten remains inactive. Whole-conversation native guard contracts need validation; a final User/Assistant OR is insufficient for all-turn interaction harm. Tev/Qwen local 8,192-context configs are not deployed or verified. Safeguard readiness remains unresolved. Do not silently truncate history or auto-scale GPUs.
- PolicyLM remains a separate User-only supplement because its author contract excludes conversation history and Assistant moderation. It cannot become a full-conversation classifier by relabeling old results.
- Full-conversation batching and the outstanding arms need replacement runs; historical pair-only batching results do not answer the revised question.
- New human annotation retrieval and statistical analysis must bind the exact new dataset/policy hashes. The old t-tests, mixed models and charts are historical pair-only diagnostics.

## Budget and held-out gates

The authorized total remains $1,000. The $886.03392 historical figure is a planning reserve, not an invoice. Adding smoke and the three known-price stage allocations gives $991.647616617 of planning reservations. The latter use recorded provider prices, UTF-8 request bytes as a conservative token proxy, output caps and three attempts; unconfirmed endpoint/gateway fees and actual invoices still need reconciliation. Do not interpret either planning reserves or partial recorded token costs as verified total spend.

No new held-out predictions have been made. Before launch: finish and retrieve the development human audit; freeze the revised policy and analysis; validate complete-history serving contracts and context capacity for each included arm; reconcile actual spend and fees; prepare a new held-out call manifest and budget. The old pair-only freeze binds the old audit and is superseded. Independent accuracy on the new held-out dialogues requires human safety labels on those exact dialogues as well. Research-corpus full history does not establish production-traffic representativeness.
