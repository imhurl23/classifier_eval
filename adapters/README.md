# Adapter interface

Each arm implements `Adapter.classify(text, scheme, rubric, option_order)` and returns a `Decision`. `rubric` is a semantic instruction string prepared by `prompts.runtime.instructions_for`; the role-marked interaction is sent once as state or user content. `prompts.runtime.options_for` supplies option descriptions, including task-specific User-harm and Assistant-harm definitions. Withheld supplies only option identifiers. All raw request and response payloads are retained without auth headers.

## Cloudflare Clef

`clef_choice` and `clef_flash_choice` use the Workers AI typed `choice` question with `state=text`, `instructions=rubric`, and an insertion-ordered `criteria` map. The [Cloudflare API contract](https://developers.cloudflare.com/workers-ai/models/clef/) documents `model`, `state`, and `questions` with one probability per choice. The [model card](https://huggingface.co/Cloudflare/clef) confirms the `choice`, `confidence`, and `probabilities` fields.

The adapter requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN` for live calls. These are distinct from the Braintrust key, which is always loaded from this repository's `.env` before Braintrust operations. The HF weight revisions are recorded in `MODEL_INFO`; Cloudflare's hosted build is not version-pinned yet, so preregistered execution remains gated until a provider revision or dated deployment identifier is captured. Run smoke tests against the live endpoint before accepting response parsing or multiclass option limits.

## TypeSafe Jev

`jev_choice` uses `POST https://api.typesafe.ai/v1/systemone` with pinned model `jev-1.13.0`, `state=text`, and a single typed Choice question. It obtains `TYPESAFE_API_KEY` from this repository's `.env`; no auth header is stored in `Decision.raw` or Braintrust logs. [TypeSafe's API reference](https://docs.typesafe.ai/api) specifies the selected `choice`, complete `probabilities`, and token `usage`. The 20-item dev smoke is in `adapters/smoke_jev.py`.

`jev_noul` asks for the probability that the complete interaction is unsafe, with separate true/false descriptions and a fixed 0.5 pilot threshold. `JevChoiceAdapter.classify_many` and `ClefChoiceAdapter.classify_many` place independent User-harm, Assistant-harm, and refusal Choice questions under one state; the native Jev path passed an eight-pair diagnostic. Keep batched and separate-call results distinct because adding questions may change predictions.

## Luna structured

`luna_structured` uses the ordinary GPT-6 Luna Responses API with JSON Schema enum output and reasoning effort `none`, through the Braintrust OpenAI gateway. It is separate from the native `decisions_luna` API arm. The adapter loads `BRAINTRUST_API_KEY` from this repository's `.env`, disables gateway cache, and records invalid outputs and refusals rather than converting them to labels. The eight-pair pilot is linked from `run/pilot-report-2026-10-01.md`.
`adapters/guards.py` contains the Baseten-hosted safeguard and native guard pilot
interfaces. Safeguard receives the written rubric as its system policy. Granite
Guardian 4.1 uses its prescribed binary BYOC score separately for User and
Assistant; the interaction label is their logical OR. Qwen3Guard uses its native
Safe/Controversial/Unsafe response format on the User and Assistant turns.
Nemotron Safety Guard v3 returns both binary turn ratings in one native JSON
response. `run/guard_pilot.py` logs diagnostic results to Braintrust using the
repository `.env` key. Native taxonomy outputs are reference arms, not
common-policy rubric cells.

`adapters/qwen_json.py` is the optional JSON-schema interface for the deployed
Qwen3.5-4B base. It uses the same model ID and pinned weights as the letter arm.
The reserved-pair pilot returned 56/56 valid JSON-schema labels. Tev1 and Qwen
deployment IDs are pinned in `.env`, so the adapters call the verified active
versions directly rather than an older production alias.

## Luna Decisions

`decisions_luna` uses `POST https://api.openai.com/v1/decisions` directly with
`gpt-6-luna`. `adapters/openai_decisions.py` loads `OPENAI_API_KEY` from this
repository's untracked `.env`. The HTTP transport uses the existing `requests`
dependency because the installed OpenAI SDK does not expose Decisions yet.
The [official contract](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)
uses shared `input` plus an ordered list of native `choice` questions. Offered
labels and descriptions retain the caller's option order. Selected labels, full
probability distributions, raw payloads, token usage, and returned model IDs are
preserved. API refusals and malformed outputs remain unscored. Batched results
share request-level usage and latency; count these once per request when totaling
costs. HTTP 429 and 5xx responses retry with a three-attempt cap. Auth headers are
excluded from logged payloads. The model alias remains unpinned.

Live checks on 2026-10-06 covered all 16 scheme/rubric combinations (including
15-option Long) and one three-question companion bundle on a synthetic benign
pair: 17 requests, 19 valid decisions. These are integration checks, not study
results or evidence of safety accuracy.

Run the reserved-pair diagnostic explicitly:

```sh
.venv/bin/python -m run.pilot --arms decisions_luna --limit 8
.venv/bin/python -m run.pilot --arms decisions_luna --limit 8 --batched-turns
```

The primary study remains gated by rubric/version freeze and preregistration.
Recompute current Decisions pricing before authorizing a study run.

## PolicyLM prompt-only reference

`policylm_17b_prompt` is a supplemental binary User-only custom-policy arm, pinned
to `musubilabs/policylm-1.7b` at
`4e5a98c579e6ab28c45af42e7d69a84da38f9b12` (release v1.2). The
[model card](https://huggingface.co/musubilabs/policylm-1.7b) excludes conversation
history and assistant-response moderation, so this arm is absent from the paired
interaction grid and response/refusal cells. The companion plan contains one
explicitly marked User-only cell. It does not implement native enum choice or
multiclass option-order experiments.

The supplied binary User-harm policy becomes one native `Category`: its violation
rule contains the study instructions and unsafe definition; its non-violation
rule contains the safe definition. The instruction to read the Assistant as
context is replaced with `Judge only the supplied User message`. This mapping is
logged as `prompt_harm_s2_to_single_category_v1`; it is an adapted comparator, not
the same full-pair condition used by most other arms. Existing approved prompts
and completed experiments remain unchanged.

`adapters/policylm.py` uses the release's explicit-policy **precision** preset,
cutoff **0.335**, selected before observing results. It disables cleaning, uses
float32 and `max_batch_rows=1`, preserves full messages through the helper's
windowing, and records window count and native violation scores. `Decision.probs`
is left unset: these scores are not a normalized distribution over study choices.
Messages exceeding 32,000 characters fail explicitly. Neither the cutoff nor the
policy is tuned on held-out data. The helper checks release/weights integrity; the
adapter verifies inference-code checksums before importing.

The model requires Transformers 4.57.6, so run in its isolated environment:

```sh
uv venv .venv-policylm --python 3.12
uv pip install --python .venv-policylm/bin/python -r deploy/policylm/requirements.txt
hf download musubilabs/policylm-1.7b --revision 4e5a98c579e6ab28c45af42e7d69a84da38f9b12 --local-dir data/models/policylm-1.7b
.venv-policylm/bin/python -m run.policylm_pilot --dev-reference
```

The diagnostic uses eight reserved WildGuard **User** turns with human User-harm
gold, plus User turns from the frozen 200-pair Aegis development reference sample.
The latter receive validity scores only: conversation labels cannot serve as
human User-harm labels. Results go to a separate Braintrust experiment and local
`data/processed/policylm/` checkpoints. There is no cloud inference charge for the
local run.

Training/evaluation exposure is material: the authors report PolyGuardMix training
(which includes WildGuardMix) and development use of WildGuardTest and XSTest.
Treat results on those sources as an exposed-source reference, not independent
held-out evidence. Its native Aegis taxonomy is not the supplied study policy.

Live diagnostic on 2026-10-07: all 208 User-only predictions were valid on Apple
MPS float32; 8 reserved prompts have human User-harm gold and 200 dev prompts
remain unscored for harm agreement. Policy length was 212 tokens, leaving 1,834
message tokens per native window. [Braintrust diagnostic](https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/experiments/policylm_prompt_pilot__20261007T173151Z).
