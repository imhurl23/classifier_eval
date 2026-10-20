# Article charts

Regenerate the current article figures with:

```sh
.venv/bin/python -m analysis.plot_article_human_labels
```

The current analysis uses F2 for five harm-screening cohorts and accuracy for
refusal detection. `score_metrics.csv` and `score_pairwise_significance.csv`
contain the current six-cohort results; `score_pairwise_all_345.csv` retains the
full correction family. Older `f2_*.csv` files are historical F2-only exports.

Current figures:

- `01_clef_evidence.png`: Clef's significant comparisons, with the metric named per row.
- `02_clef_vs_best_alternative.png` and `.svg`: Actual scores for Clef and each task's strongest alternative, with confidence intervals and significance notes.
- `04_refusal_detection_accuracy.png` and `.svg`: All ten refusal models ranked by accuracy, with Wilson intervals.

Failed decisions count as incorrect in accuracy. Refusal tests use exact paired
McNemar tests. Harm-screening F2 scores and unadjusted tests are unchanged. Holm
correction is recomputed over the original 345 comparison slots, including the
retired derived WildGuard cohort. These exploratory tests are conditional on
saved runs; unresolved differences do not establish equivalence.
