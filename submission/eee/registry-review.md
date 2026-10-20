# Registry reconciliation draft

Read-only inspection: eval-card-registry commit `a3c24141aac369d0ed99cb8888f0b05b75383e48`. No entries were submitted or auto-created.

| Source model | Prepared identity | Status |
|---|---|---|
| `Qwen/Qwen3.5-4B` | same | Exact reviewed ID |
| `claude-sonnet-5-5` | `anthropic/claude-sonnet-5.5` | Exact alias in reviewed entry |
| `gpt-6-luna` | `openai/gpt-6-luna` | Exact alias in reviewed entry |
| `openai/gpt-oss-safeguard-120b` | same | Exact reviewed ID |
| `jev-1.13.0` | same raw ID | No exact reviewed match found; do not silently map to the draft `jev-1-13` entry |
| `clef`, `clef-flash` | same raw IDs | No exact reviewed match found; hosted build was unpinned, so do not claim a specific HF weight revision |
| `togethercomputer/Tev1-4B-experimental` | same raw HF ID | No exact reviewed match found |

The source metadata records the file supplying each confirmed match. Unknowns remain marked unverified; they are not proposed new canonical IDs.

Six task variants need an alias/new-task decision based on their definitions in `source_metadata.json`. Aegis pools filtered validation/test records; WildGuard request classification sees the answer; XSTest measures prompt harm rather than a generated model answer. The prepared `decision_interface_safety.*` names deliberately retain these distinctions while registry review is pending.

The metric is binary F-beta with beta=2, range [0,1], larger is better. The prepared `f2` ID and `classifier_eval` harness ID have not been confirmed in the registry. Use a matching existing entity where possible; propose a new entity only after the owner and registry maintainers agree on the definition.
