# Repository cleanup — 2026-10-21

The GitHub working repository remains private. The public release is the separate
[HF companion dataset](https://huggingface.co/datasets/BraintrustDataDev/decision-interface-safety-eval).
This cleanup does not clear the working repository or its history for publication.

## Removed from version control

93 files (27,390,239 bytes) were removed from the current Git tree. Every file
remains locally at its original path, with its SHA-256 verified after removal.
The local inventory is `tmp/privacy-audit-2026-10-21/local-only-files.json`.

- Benchmark conversation tables and reports quoting unreviewed source text.
- Per-example predictions, diagnostic decisions, retry details and review tables.
- Browser captures, provider-output illustrations, an internal review PDF,
  experiment reports and account/deployment audit artifacts.
- Superseded charts and duplicate exports.
- Optional historical-notebook logos whose redistribution rights were not documented.

42 saved outputs were cleared from the historical notebook. Its logo loader now
treats local artwork as optional, so names still render when the logos are absent.
The analysis cells are otherwise unchanged; an original copy remains in the
ignored audit directory. A hardcoded WildGuard quotation in the historical report
generator was replaced with a general description.

227 aggregate output files remain tracked, including the inputs and five figures
for the current article. Frozen study datasets, policies, adapters, runner code,
labels and scores were not changed. Some historical analyses still require the
local artifacts or source downloads under their original terms.

## Keeping generated material local

New files under `analysis/output/` are now ignored by default. A deliberate
addition requires content review, an entry in
`analysis/tracked_aggregate_outputs.json`, and an explicit `git add -f`.
Do not add source conversations or per-example predictions to that list.

After staging changes, run:

```sh
.venv/bin/python -m analysis.check_repository_privacy
```

The check reads staged bytes, rejects known local-only paths and saved notebook
outputs, and checks generated CSV/JSON fields for source text, example identifiers
and operational identifiers. It also rejects new generated paths that have not
been listed. It is a commit check, not comprehensive PII detection or a substitute
for reviewing text, images and new export schemas.

## Public dataset and licenses

All 13 published files at HF revision
`444cc355cd00306c2f6abfccabd2934005261392` match a fresh offline export byte-for-byte.
The release contains 5,643 source references and 301 reviewed text rows: 250
XSTest prompts, 33 Aegis records and 18 topic-control records. Each text row retains
its creator credit, source locator and revision, license, transformation notes and
hash-bound review. The reviews are agent reviews, not independent human review.

The [component license notice](../submission/hf/LICENSE.md) retains CC BY 4.0
for the reviewed text and ODC-BY attribution for WildGuard database references.
WildGuard text and labels remain absent. The exporter continues to withhold
unreviewed text and the categories specified in its
[release policy](../submission/hf/release_policy.json), including source-tagged
minor sexual content, privacy/confidentiality material and copyright violations.
Provider redactions remain intact. HarmBench, HH-RLHF dialogue artifacts, training
files, weights and provider outputs remain outside this release.

No additional HF change was needed. The checks found no mismatch with the
documented release rules; they do not establish rights that an upstream provider
does not hold. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) requires
attribution and change notices and does not grant every privacy or publicity
right. [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) covers database rights
separately from rights in individual contents. Source research-use guidance has
not been rewritten as a noncommercial license.

## Remaining private material

Removing files from the current tree does not remove them from old commits.
Private Git history still contains benchmark text, browser captures, predictions
and internal operational details. Frozen operational code and dated runbooks also
retain internal project references. Keep this repository private; distribute only
the separately reviewed exports. No history rewrite or visibility change was made.

The reachable-history Gitleaks scan covered 13 commits and reported one finding:
the public Qwen tokenizer revision in `data/build_datasets.py`, misclassified as
an API key. It is not a credential. Scan reports and local source-matching results
remain in the ignored audit directory.

## Validation

- Main test suite: 248 passed, one skipped for an optional dependency.
- HF release tests: 18 passed; staged-boundary tests: four passed.
- The five current article figures rebuilt from an isolated copy containing only
  staged files, with no raw source data, local predictions or credentials.
- The current October 19 source-aligned run's frozen code, data and request
  manifest checks pass. No frozen run files were modified.
- The staged-tree credential scan found only the same public tokenizer revision
  false positive. The downloaded HF tree had zero findings.
- Exact matching against known local credential values found zero matches in the
  staged files or 600 historical blobs, including three decompressed gzip blobs.
  Values were not printed or written to reports.

These scans cannot establish the absence of every unknown secret or private detail.
