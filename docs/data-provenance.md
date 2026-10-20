# Data provenance and release scope

The study evaluates against existing published labels. It does not create an independently adjudicated safety dataset. Original source text can include harmful, offensive, or personal-looking material; benchmark provenance is not a guarantee that every string is appropriate to redistribute.

| Source | Pinned revision | Current use |
|---|---|---|
| [NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` | Filtered validation and test, pooled by original single-turn/paired input unit |
| [AllenAI WildGuardMix / WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` | Filtered test pairs, separate request harm, response harm, and refusal labels |
| [XSTest](https://github.com/paul-rottger/xstest) | `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` | All 450 original prompts, binary prompt-harm labels, reused original run |

The source revisions are recorded in `data/build_datasets.py`; fetching uses `data/fetch_sources.py`. The pinned Aegis and XSTest source documentation records CC BY 4.0; WildGuard records ODC-BY and requires accepting its dataset access terms. Consult the original source cards and terms before redistribution. This repository does not relicense upstream data or grant rights to provider outputs.

Aegis human `prompt_label` on paired records represents an interaction-level vote; separate response labels are not used as independent human gold. Single-turn Aegis records are original single-turn examples, not conversations with the answer removed. Its complete human annotation instructions are not available, so published taxonomy alignment does not establish identical policies.

WildGuard classifiers receive both turns even when judging only the request. The main study excludes reserved pilot examples and keeps the three human targets distinct. One missing refusal label is excluded. The derived whole-exchange OR target from historical runs is absent from the revised main comparison.

EEE preparation includes aggregate metrics and study-authored policy text only. It excludes raw datasets, example-level predictions, case-review tables, screenshots, and private fine-tuning artifacts. Separate release of those materials requires its own review.

The policy revision followed examination of test disagreements. These data cannot support a claim that the revised policy was chosen without test exposure. The small policy/choice probes and fine-tuning follow-up are separate from the exported main F2 table.
