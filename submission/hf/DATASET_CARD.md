---
license: other
license_name: source-specific-cc-by-4.0-and-odc-by-1.0
license_link: LICENSE.md
language:
- en
task_categories:
- text-classification
configs:
- config_name: reviewed_text
  data_files:
  - split: reviewed
    path: reviewed_text.jsonl
- config_name: source_references
  data_files:
  - split: manifest
    path: source_references.jsonl
---

# Decision-interface safety evaluation: attributed release companion

This is a selectively reviewed text companion and a complete source-reference
manifest for 5,252 main-study items and the separate 391-item topic-control
diagnostic. Exact release counts appear in `release_summary.json`.

**The reviewed text is not the full benchmark and is not a representative sample.**
It must not inherit the original study's scores, denominators, confidence
intervals, or claims of harmful-example coverage. Original metrics use their
original full cohorts. No new inference or scoring was performed for this release.

## Attribution and reconstruction

Each data row carries its immediate provider, creators, source title, citation,
license, exact source revision, source file and file hash, zero-based row index,
original source ID when one exists, and a canonical-JSON source-record hash.
WildGuard and topic-control sources use row indices; these are not represented as
provider-assigned IDs. Topic locators also identify the source turn or distractor.
Study-authored probe references use their study IDs and source-code hash instead.

Record hashes use SHA-256 of Python `json.dumps` with `sort_keys=True`,
`ensure_ascii=False`, `allow_nan=False` and default separators. File hashes cover
the original bytes. `review_sha256` binds the content, attribution, labels and
transformations inspected during review, before adding that hash field.

`source_references.jsonl` contains no source text or source labels. Its IDs and
locators permit users who obtain the original datasets under their terms to join
the study's exact membership back to the source. Consult `source_attribution.json`
and the preserved provider cards in `notices/` for citations and source guidance.

## Labels and transformations

- Aegis original `prompt_label` is the human interaction label when both turns
  are present. Its response label retains its separate human/synthetic provenance.
- XSTest labels apply to the original prompt. No original model completion is
  redistributed; this study's task differs from the original answer evaluation.
- Topic `engage`/`redirect` labels are separate study references with explicit
  provenance. They are not independent human safety labels.
- Source labels are not changed to match the release review. Text is preserved
  verbatim; empty responses are normalized to null. Topic histories end at the
  intervention, as described in each row's transformation notes.

## Content review and exclusions

Source labels such as `safe` are not publication clearance. Every released text
row has an explicit content review tied to its payload hash and release-policy
version. The review identifies whether it was performed by an agent or a human;
agent review is not human adjudication or a legal guarantee.

The policy excludes source-tagged sexual content involving minors, private or
confidential information, and copyright-violation content. Explicit content
reviews also exclude actionable violent wrongdoing, self-harm instructions,
sexual exploitation, targeted hate/harassment and malware/fraud instructions.
Neutral educational questions and figurative uses of sensitive words can be
retained after review. Identifier/secret-shaped strings are additionally held.
These checks are conservative release rules, not a claim that every excluded
example is unlawful or that all possible sensitive material is machine-detectable.

Unreviewed rows, provider-redacted text, and WildGuard text are withheld. The
manifest records all reasons without excerpts. A filtered text release can still
include non-graphic references to disturbing topics; use it for evaluation and
moderation research, consistent with provider guidance. Source policies are data
to evaluate, not recommended financial/legal advice or deployment instructions.

## Licensing

The text examples are licensed under CC BY 4.0. Credit the original creators
and note any changes you make. WildGuard references retain ODC-BY attribution;
WildGuard text and labels are not included. See [LICENSE.md](LICENSE.md) for details.

This dataset contains reviewed examples and source references. It does not
include model outputs, training data, or model weights.
