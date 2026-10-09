# Rubric review index — drafts, not frozen

The four candidate rubric templates are:

| Condition | File | Intent |
|---|---|---|
| Short | [r_short.md](rubrics/r_short.md) | Task, shared definitions, output rule |
| Long | [r_long.md](rubrics/r_long.md) | Short plus policy boundaries and newly authored paired examples |
| Neutral-Long | [r_neutral_long.md](rubrics/r_neutral_long.md) | Short plus process/format text without added policy |
| Withheld | [r_withheld.md](rubrics/r_withheld.md) | No policy prose or label definitions; only option names remain |

The fixed [prompt-harm](rubrics/prompt_harm_s2.md), [response-harm](rubrics/response_harm_s2.md), and [refusal](rubrics/response_refusal_s2.md) rubrics ask separate questions on paired WildGuardTest interactions. They use the full interaction as context and are separate from the four interaction-rubric conditions. [runtime.py](runtime.py) sends the interaction once, in the adapter's state/user field, and supplies task-specific option descriptions. In Withheld, Choice criteria contain only each option identifier, which the API requires; no policy definition is supplied.

The study's [policy target](policy_target.md) defines observed interaction harmfulness and explains how conversation, prompt, and response labels relate. The shared interaction definitions are in [label_definitions.yaml](label_definitions.yaml). Short, Long, and Neutral-Long start with the same policy paragraph. `runtime.py` removes their legacy presentation placeholders before adapter calls and supplies definitions through each adapter's option interface. The rubrics have not been frozen. The earlier 20-item Jev smoke used a separate provisional prompt on user prompts only; the new `dis-pilot` uses full pairs.

`python -m prompts.review` renders every scheme and checks that Neutral-Long is within 3% of Long under Qwen3.5, gpt-oss, and o200k tokenizers. It also checks the shared opening. After the 2026-10-02 policy and process-control review, the draft `s13` template renders to 1,337 Qwen3.5 tokens for Long and 1,335 for Neutral-Long. These are template measurements, not full adapter requests. The corrected source filter proposes a cap of 1,200 tokens per interaction (the previous WildGuard metadata counted only the prompt), but rubric, option descriptions, JSON/chat wrappers, and the interaction must all fit a provider's context. Before these policy edits, a simple Qwen estimate of runtime instructions + option descriptions + paired state put 82 held-out `s13` Long pairs above 2,048 tokens. Tev's optional training recipe uses a 2,048-token cap; the deployed inference configurations specify 4,096. Recompute full-request coverage for the revised prompts and measure the actual serving tokenizer and truncation behavior before the main run.

The Long rubric paraphrases the Aegis 2.0 [annotator instructions and taxonomy discussion](https://aclanthology.org/2025.naacl-long.306.pdf). Its examples are newly written paired scenarios rather than Aegis validation items, so their text does not overlap the dev set. They illustrate policy boundaries and do not constitute adjudicated multiclass gold.

Before freeze: independently audit the policy on paired dev interactions, independently review Neutral-Long for added policy content, resolve the actual adapter token budget, and confirm how the Long examples are displayed across arms. Token matching controls length only; it does not establish that generic process wording has no behavioral effect. Any rubric edits after preregistration require a new version and explicit analysis separation.

Neutral-Long now uses interface-independent process text and leaves response formatting to the adapter. This removes known JSON/letter/typed-output conflicts; it does not prove behavioral neutrality. The offline review tool generates a blinded development annotation form, overlap report, actual request snapshots, and a staged call budget. See the development annotation workflow in `policy_target.md`.
