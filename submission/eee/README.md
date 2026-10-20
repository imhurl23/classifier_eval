# Every Eval Ever submission preparation

**Prepared locally; not submitted.** The offline adapter exports the complete main F2 table from `source_aligned_v1_20261019`: 62 task/configuration records, 11 interfaces, eight underlying model identifiers, six cohorts, and 89,251 scored decisions. No source rows are dropped.

This is an aggregate-only contribution. It contains no raw benchmark conversations, per-example predictions, credentials, internal experiment links, browser captures, or private checkpoint artifacts. It does include the rendered policy and option definitions needed to interpret the scores.

## Rebuild and validate

Use Python 3.12+ and `uv`. Run from the repository root, choosing a fresh output directory for each export:

```sh
uv venv tmp/eee-env --python 3.12
uv pip install --python tmp/eee-env/bin/python -r submission/eee/requirements.txt
tmp/eee-env/bin/python -m pytest -q analysis/test_eee_export.py
tmp/eee-env/bin/python -m submission.eee.adapter --output-dir output/eee-fresh/data
```

Run the validator **from the package root** so its path checks see `data/<collection>/<developer>/<model>/<uuid>.json`:

```sh
cd output/eee-fresh
../../tmp/eee-env/bin/python -m every_eval_ever validate 'data/decision-interface-safety/*/*/*.json'
../../tmp/eee-env/bin/python -m every_eval_ever.check_duplicate_entries data/decision-interface-safety/*/*/*.json
```

Dependencies are pinned to EEE commit `0791adc3c2488cc8c91d109b3c541da492238e49` (schema 0.3.0). The adapter imports `SCHEMA_VERSION` from the installed package. It checks the source CSV's SHA-256, verifies F2 against confusion counts, preserves confidence intervals, rejects duplicate identities, and writes an explicit coverage report outside `data/`. It refuses a nonempty output directory. Logical evaluation IDs are stable; fresh exports receive new UUID filenames.

The local prepared package is `output/eee-submission/`; the review bundle is `output/eee-submission-bundle.zip`. Generated exports stay out of Git. EEE accepts data through its [Hugging Face datastore](https://huggingface.co/datasets/evaleval/EEE_datastore), with adapter code contributed separately under its [contribution process](https://github.com/evaleval/every_eval_ever/blob/0791adc3c2488cc8c91d109b3c541da492238e49/CONTRIBUTING.md).

## What is preserved

- One log per model/interface/task configuration. Jev Choice/Noul, Luna Decisions/structured, and Qwen letters/JSON retain separate identities.
- Raw model names and served revisions alongside registry matches where confirmed.
- Aegis's filtered validation-plus-test pooling; WildGuard's target and full-pair context; refusal's different positive class.
- XSTest's reused original-policy provenance. Its score measures prompt harm classification, not the standard generated-answer refusal/appropriateness evaluation.
- F2 on a 0–1 scale, 95% bootstrap intervals, counts, bootstrap seed/replicates, and missing-verdict treatment.
- The exploratory policy revision and other study limitations.

No new scores, inference, human labels, or significance tests are created. Other historical runs, fixed-policy supplements, small development diagnostics, and the separate custom fine-tuning study are outside the selected main table. This is not an export of every experiment ever run in the repository. API costs and pairwise significance remain in the study's companion reports rather than being mislabeled as F2 fields.

## Registry review still needed

[`source_metadata.json`](source_metadata.json) pins registry commit `a3c24141aac369d0ed99cb8888f0b05b75383e48`. Offline exact-ID/alias inspection found reviewed matches for Qwen, Claude, Luna, and Safeguard. Jev 1.13.0, Clef, Clef Flash, and Tev1 retain their raw source identifiers, explicitly marked unverified. Jev's weight availability is left unknown.

The six study task variants, F2 metric ID, and custom harness also need registry reconciliation. In particular, these task definitions must not silently inherit scores for a different standard XSTest or WildGuard evaluation. See [registry review](registry-review.md). No registry entries were created or changed. The resolver can auto-create draft entries, so preparation used a read-only pinned snapshot instead.

A successful schema validation does not resolve these semantic identity questions.

## Before public submission

1. Select an approved public source URL and confirm attribution. The working repository is private; the export currently identifies its exact source commit and CSV hash without pretending they are publicly accessible.
2. Confirm a release license for the contribution. No code/data license has been assigned on the owner's behalf. Source datasets retain their own terms; see [provenance](../../docs/data-provenance.md).
3. Reconcile unresolved registry IDs and regenerate if mappings change. Use the [submission draft](submission-draft.md) and record those decisions.
4. Open the adapter, registry (if needed), and datastore PRs and cross-link them. For a datastore PR, upload only the `data/decision-interface-safety/` tree, not this repository or the full review bundle. Run the datastore bot's validation on the PR as well as the local validator.

No repository visibility change, public upload, PR, or message to maintainers has been made.
