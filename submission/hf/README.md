# Local HF release preparation

Published dataset: [BraintrustDataDev/decision-interface-safety-eval](https://huggingface.co/datasets/BraintrustDataDev/decision-interface-safety-eval).
The publication history and verification results are recorded in the
[release review](../../docs/hf-release-review.md).

This is a new distribution layer; it does not rewrite frozen source datasets,
runner files, study manifests, scores, or the existing EEE bundle. It never uploads.

```sh
.venv/bin/python -m submission.hf.export --output output/hf-reviewed-2026-10-21
.venv/bin/python -m pytest -q submission/hf/test_export.py
```

Use a fresh output directory for every build. The complete allowlisted output is
the only candidate for HF publication; never upload the working repository or
the contents of `data/processed/` wholesale.

`sources.json` records the provider terms and source bytes checked on 2026-10-21.
Public source files were compared byte-for-byte against downloads at the pinned
revisions. The WildGuard file was compared to the authenticated HF snapshot at
its pinned revision. The builder checks those hashes on every run, resolves each
main-study row back to its original row and source ID, and checks its actual text
and source labels. Topic examples are reconstructed from the pinned dialogue and
intervention indices. Unknown sources and altered data stop the build.

`reviews.jsonl` is the text-free review ledger. To add a review, inspect the full
candidate returned by `candidates()` (including policy and history), record the
exact `review_sha256`, the current policy version, reviewer identity/type, date,
decision, reason and excluded topics. Do not turn a source `safe` label or a clean
keyword scan into a publication decision. Changed attribution, labels, source
text, transformations or policy invalidate prior reviews. Do not use provider
secrets or upload source text to perform a review.

All main and topic study rows remain in the public reference manifest, even when
their text is withheld. Exclusion counts overlap; a row can have multiple reasons.
The text subset is selected for distribution and cannot reproduce the full-cohort
metrics without obtaining the omitted sources separately.

The public dataset's new arrangement and annotation contributions use CC BY 4.0;
upstream licenses remain source-specific. This does not apply a repository-wide
code license. Study-authored controlled-probe text remains out of scope pending
its own license assignment. HarmBench and HH-RLHF artifacts are not exported;
their presence in the working project cannot implicitly enable redistribution.
