# First-wave held-out launch

The first-wave plan is prepared locally, with 2,487 full conversations, 39,792 physical requests, 34,818 primary interaction decisions (including native derived judgments), and a separate 2,487-request local PolicyLM supplement. No held-out inference has launched. `summary.json` identifies the exact source, code, policy, deployment configs, tokenizer files and wire-request hashes.

The preflight at `data/processed/baseten-preflight/20261007T200841Z` passed Tev, Qwen letters/JSON, Granite, Qwen3Guard and Nemotron diagnostics. Those five deployments were verified inactive with zero replicas after completion. Safeguard's preceding readiness attempt timed out and it was also verified inactive. This is readiness evidence on small development examples, not full-context verification or held-out performance.

Tev and Qwen's local configs now request an 8,192-token context. The live deployments remain unchanged at 4,096: publish matched updates, update the explicit deployment bindings, and repeat bounded development preflight before using the new context. The longest local Qwen first-wave requests reserve 4,691 tokens for letters and 4,881 for JSON. Serialized-wire proxies also flag longer requests for other guards; their actual model templates and serving limits still need verification. Preserve every full conversation.

Finish these inputs before freezing:

1. Complete the audit in [Braintrust native review](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/datasets/human_policy_audit_blind__f0c7bbda4b65). All 200 rows are assigned to Izzy Hurley; the six human fields save into expected. Retrieve these actual judgments and their reviewer provenance into the validator packet after completion; the local HTML form is a backup. It needs the reviewer identity, all 200 unique ratings, and reasons for ambiguity or unrateable judgments. Review the findings and finalize the policy. The validator preserves the original ratings.
2. Verify each scheduled wire request fits the real serving template, context and output reserve. A tokenizer proxy is insufficient evidence. Fill the context template with the exact request hash and evidence.
3. Bind each arm to validated model/deployment versions. Resolve Safeguard startup before retaining it in an executable wave; revisions of the arm set require a new manifest and analysis plan before test inference.
4. Reconcile historical billed spend and unsettled charges, Decisions endpoint billing, gateway fees, and GPU rates/time. Allocate per-arm ceilings plus shared fees/contingency within the user's $1,000 total. The first-wave known-rate planning reserve is $172.39 for hosted arms, plus a $481.07 hypothetical Decisions contingency. These use three attempts and request bytes as an input proxy; the Decisions figure is not a verified rate. GPU compute is additional. Earlier $886.03 planning reserves are not actual charges and must not be mistaken for spend or available balance.
5. Freeze the analysis plan and evidence, then run the checkpointed executor only after deployment readiness and the compute lifecycle are established. This local receipt is not an externally timestamped preregistration.

Generate the plan and incomplete evidence templates without inference:

```bash
.venv/bin/python -m run.prepare_firstwave
.venv/bin/python -m run.prepare_launch_evidence --plan data/processed/heldout-firstwave/<version>
.venv/bin/python -m run.heldout --plan data/processed/heldout-firstwave/<version> --check
```

Copy/edit the templates rather than overwriting completed evidence. They deliberately start with false/missing values. Once complete:

```bash
.venv/bin/python -m run.freeze_heldout \
  --plan data/processed/heldout-firstwave/<version> \
  --audit <completed-audit.json> --context <verified-context.json> \
  --readiness <verified-readiness.json> --budget <reconciled-budget.json> \
  --policy-finalized
.venv/bin/python -m run.heldout \
  --plan data/processed/heldout-firstwave/<version> --wall-minutes <frozen-limit>
```

The executor uses an exclusive process lock, durable result checkpoints and an inflight journal. `--resume` permits only partial executions, skips recorded calls, and blocks unresolved inflight calls until billing/output reconciliation. Cumulative execution time cannot exceed the frozen envelope. Three consecutive failed outputs stop that arm. All missing outputs and failures remain visible. Finally, Baseten deactivation is requested and inactive/zero status checked; inspect the saved shutdown receipt. An external lifecycle supervisor is still needed to enforce GPU shutdown after an uncatchable process termination or host outage.

`native-derived.json` computes the interaction OR without new inference or duplicated usage; Nemotron's shared-call latency is recorded once. Qwen3Guard caution remains explicit. The PolicyLM supplement manifest is prepared, but the current executor handles the 39,792 remote requests only; its separate frozen local execution remains to be wired before claiming the supplement complete.

Expand additional schemes, rubrics and companions through separately frozen, costed waves. Use a fresh untouched test set for tuning after these held-out outcomes are examined.
