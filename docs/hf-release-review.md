# HF companion review — 2026-10-21

Reviewed locally, then published on 2026-10-21 at the user's instruction to
[BraintrustDataDev/decision-interface-safety-eval](https://huggingface.co/datasets/BraintrustDataDev/decision-interface-safety-eval)
at commit `586d263a549d7ed08ef1e46b9ff476b8ae210fe8`. All 13 uploaded files were
downloaded without authentication and verified byte-for-byte against the reviewed
bundle. The working repository's visibility was not changed. This review covers
the main study and separate topic-control diagnostic, not every working artifact.

The dataset card wording was simplified in HF commit
`444cc355cd00306c2f6abfccabd2934005261392`. Only `README.md` and its entry in
`checksums.json` changed. Both published files were downloaded and verified;
data files and license terms were unchanged.

## Coverage

| Item | Count |
| --- | ---: |
| Main-study source references | 5,252 |
| Topic-control source references | 391 |
| Total attributed rows | 5,643 |
| Individually agent-reviewed rows | 308 |
| Text included | 301 |
| Reviewed text withheld | 7 |
| Unreviewed text withheld | 5,335 |

Included text consists of 250 XSTest safe prompts, 33 Aegis validation examples
and 18 human-test topic-control first turns with their policies. This is an initial
curated companion, not full clearance of the remaining source text. Selection
strongly favors benign material and is not a representative benchmark sample.
Reviews were performed by the assistant, not an independent human reviewer.

## Provenance

Public source bytes were downloaded at the pinned revisions and compared with
local Aegis test/validation, XSTest and both topic-control files. WildGuard was
compared with an authenticated HF download at the pinned revision. Exact hashes,
creator lists, citations, license links and source-card evidence hashes are in
`submission/hf/sources.json`. The exporter verifies those bytes every time.

Every main-study ID resolves to the exact upstream row, original provider ID
(where available), source text and original labels. Topic examples are rebuilt
from the pinned source dialogue and turn/distractor index. Row indices are
explicitly zero-based; they are not invented upstream IDs. Frozen membership
hashes prevent substituting a different cohort of the same size.

Each reviewed payload hash covers attribution, all released text, source labels,
derived labels where applicable and transformation notes. Stale, duplicate or
unknown reviews fail the build. Immediate provider attribution does not invent
an original individual author for in-the-wild material.

## Distribution rules and licensing

The complete rules are in `submission/hf/release_policy.json`. Source-tagged
sexual content involving minors, privacy/confidentiality material and copyright
violations are excluded from text release. Reviews additionally withhold harmful
instructions, exploitation and targeted hate/harassment. Benign homonyms,
fictional contexts and educational discussion are distinguished from prohibited
content. Source `safe` labels are never publication approval: seven reviewed
Aegis rows were withheld for discrimination or unresolved context.

WildGuard remains references only. Provider-redacted text is never reconstructed.
An identifier/secret-pattern check applies to every content field, including
assistant responses and histories, but is not represented as comprehensive PII
detection. Unreviewed text never enters the public text export.

Original CC BY 4.0 attribution and notices remain attached to reviewed Aegis,
XSTest and topic-control material. XSTest model completions are not included.
WildGuard's database attribution remains ODC-BY and its access conditions are
linked. The new export's arrangement and annotation contributions use CC BY 4.0;
this does not relicense upstream material or the entire working repository.
The 12 study-authored control probes are referenced but their text remains out
pending a separate license assignment.

HarmBench, HH-RLHF full-dialogue artifacts, training files, checkpoints, provider
outputs, predictions, internal logs and Git history are outside this export.

## Validation

All 18 release tests pass. They cover source identity/text mismatches, stale reviews, mandatory
topic/access exclusions, safe-label non-approval, nested content scans, metadata
leak prevention, complete reference coverage and output checksums. Both JSONL
files were also parsed with PyArrow to check HF-compatible schemas.

EEE source metadata now includes creator, license, citation and provider terms
for every aggregate task. All nine EEE export tests pass, including a new check
that every record carries provider attribution. The refreshed 62-record export
in `output/eee-attributed-2026-10-21/` passes the upstream schema and semantic
validator with zero errors or warnings. Its previous aggregate bundle remains
historical. None of the full-cohort scores describes the 301-row text subset.

The October 19 source-aligned run's frozen code and request manifest hashes match.
The earlier `000bf1bcb6dc` run has pre-existing code-manifest drift in
`adapters/README.md`, `adapters/luna.py` and `prompts/README.md`: their current bytes
match Git HEAD, but not that older manifest. Its frozen data and request artifacts
still match. This release work changes none of those files. Do not describe that
older code snapshot as fully reproducible from current HEAD without reconciling
the recorded versions.

The local text-and-reference bundle is `output/hf-reviewed-2026-10-21/`; the
corresponding zip contains only that generated directory's allowlisted files.
The final 13-file publication tree passed a redacted Gitleaks directory scan
with zero findings. Archive membership and CRCs were verified. This is an
automated credential check, not a guarantee of complete privacy detection.
