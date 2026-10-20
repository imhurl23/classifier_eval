# Release audit — 2026-10-20

**Scope:** preparation of a limited Every Eval Ever aggregate submission, not clearance to make this entire working repository public. The repository remains private. No public upload, external PR, history rewrite, or visibility change occurred.

## Credential checks

Gitleaks 8.30.1 was downloaded from its official release and its archive SHA-256 checked against the release checksum list. Redacted scans covered all 11 reachable Git commits and the 521-file committed HEAD snapshot at `1cd8da8`.

Both scans reported the same single finding in `data/build_datasets.py`: the public Qwen tokenizer revision (`SOURCE_REVISIONS['qwen35_tokenizer']`), misidentified as a generic API key. Inspection confirmed it is a source revision, not a credential. No actual credential was found by these scans.

A separate exact-match check compared locally configured API keys/tokens/secrets against all 521 tracked files and all 569 historical blobs, decompressing three historical gzip blobs. It found no matches. Secret values were not printed or placed in reports. This check only covers known local values; pattern scans cannot guarantee the absence of every possible secret.

Raw redacted scan reports remain in ignored `tmp/public-release/`, rather than being published with operational details.

## Content review and changes

- Rewrote the root README around the completed October 19 comparison, its six tasks, scoring, limitations, and offline chart reproduction. Removed the accumulated operational diary and internal experiment links from that entry point. Original notes remain recoverable in private Git history and dated runbooks.
- Updated adapter and policy README status so historical preflight notes do not imply the latest run is still waiting to launch.
- Removed a local user/workspace path from saved notebook output. Historical Git versions still contain that path.
- Inventoried text matches for workspace URLs, emails, deployment/account identifiers, and local paths. The pre-edit inventory found workspace URLs in 14 files and email-shaped strings in four files. These include historical experiment links, example benchmark text, and code patterns; presence is not itself proof of a secret.
- Reviewed two original article screenshots visually; neither showed an obvious account name or secret. This was a spot check, not an exhaustive image/PDF privacy audit. **All screenshots and PDFs are excluded from the EEE package.**

The full repository still contains internal experiment/project references, operational reports, and historical benchmark case text. Those artifacts are not needed for this aggregate submission and were not represented as public-release-ready. Frozen runner/adapter/policy files were preserved; both completed run manifests still verify.

## Submission boundary

The prepared export is an allowlisted package: 62 aggregate JSON records; reviewed model/task metadata and policies; the exact aggregate source CSV; export code/dependencies; coverage, validation, and submission notes. No `.env`, raw datasets, per-example traces, case-review tables, notebooks, browser captures, private fine-tuning checkpoints, or Git history enter it.

Exported model/task IDs marked unverified remain publication work, not schema errors. No repository release license is selected, and the source repository is private. Public source access, attribution/license, and registry reconciliation remain necessary before submission.

## Validation performed

- Main repository suite: **244 passed, 1 skipped**. The skipped module requires the optional EEE dependency.
- Isolated EEE export suite: **8 passed**.
- EEE schema and semantic CLI: **62 passed; 0 errors; 0 warnings**.
- EEE duplicate check: **no duplicates**.
- Conversion coverage: **62 of 62 source rows**, zero failures/exclusions; counts sum to 89,251 decisions.
- New export code/tests: Ruff import/unused-code checks and formatting pass.
- Current article charts rebuild offline in an isolated environment; both frozen run manifests verify unchanged.

For repeatable EEE validation, follow [`submission/eee/README.md`](../submission/eee/README.md). For a future repository-wide release, scan the final publication tree and history again and separately review media, source-data redistribution, and internal operational material. This report is a record of performed checks, not a guarantee that unreviewed content is publishable.
