"""Check export semantics separately from upstream schema validation."""

import csv
import json
from copy import deepcopy

import pytest

pytest.importorskip("every_eval_ever")
from submission.eee.adapter import (
    DEFAULT_CSV,
    DEFAULT_METADATA,
    convert_rows,
    make_log,
)


@pytest.fixture
def source():
    with DEFAULT_CSV.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return rows, json.loads(DEFAULT_METADATA.read_text())


def test_complete_export_preserves_scores_and_cohorts(source, tmp_path):
    rows, metadata = source
    converted = convert_rows(rows, metadata, tmp_path / "data", "1")
    assert len(converted.records) == len(rows) == 62
    assert not converted.failures
    assert len({r.eval_log.evaluation_id for r in converted.records}) == 62
    assert sum(int(r["total_n"]) for r in rows) == 89251
    for row, output in zip(rows, converted.records):
        result = output.eval_log.evaluation_results[0]
        assert result.score_details.score == float(row["f2"])
        assert result.score_details.uncertainty.confidence_interval.lower == float(
            row["ci_low"]
        )
        assert result.score_details.details["positive_label"] == row["source_positive"]


def test_interfaces_and_reruns_keep_stable_distinct_ids(source):
    rows, metadata = source
    selected = [
        r
        for r in rows
        if r["model"] == "gpt-6-luna" and r["cohort"] == "wildguard_response"
    ]
    first = [make_log(r, metadata, "1") for r in selected]
    assert len(first) == 2
    assert len({r.evaluation_id for r in first}) == 2
    changed = deepcopy(metadata)
    changed["models"]["gpt-6-luna"]["id"] = "openai/some-future-canonical-alias"
    assert make_log(selected[0], changed, "2").evaluation_id == first[0].evaluation_id


@pytest.mark.parametrize(
    "field,value",
    [("f2", "0.001"), ("ci_low", "nan"), ("positive_n", "0"), ("fp", "-1")],
)
def test_corrupt_scores_or_counts_are_accounted_for(source, tmp_path, field, value):
    rows, metadata = source
    bad = dict(rows[0], **{field: value})
    result = convert_rows([bad, rows[1]], metadata, tmp_path / "data", "1")
    assert len(result.records) == len(result.failures) == 1
    with pytest.raises(ValueError):
        result.raise_if_incomplete()


def test_duplicate_rows_fail(source, tmp_path):
    rows, metadata = source
    result = convert_rows([rows[0], rows[0]], metadata, tmp_path / "data", "1")
    assert len(result.failures) == 1


def test_different_tasks_retain_their_provenance(source):
    rows, metadata = source
    xstest = next(r for r in rows if r["cohort"] == "xstest_prompt")
    result = make_log(xstest, metadata, "1").evaluation_results[0]
    assert (
        result.generation_config.additional_details["provenance"]
        == "reused_original_run"
    )
    assert "appropriateness" in result.source_data.additional_details["subset"]
    aegis = metadata["cohorts"]["aegis_conversation"]["source_data"]
    assert "hf_split" not in aegis
    assert "validation + test" in aegis["additional_details"]["splits"]


def test_each_aggregate_record_retains_provider_attribution(source):
    rows, metadata = source
    for row in rows:
        details = (
            make_log(row, metadata, "1")
            .evaluation_results[0]
            .source_data.additional_details
        )
        assert details["source_creator"]
        assert details["source_license_url"].startswith("https://")
        assert details["source_citation_url"].startswith("https://")
        assert "do not change these scores" in details["redistribution_scope"]
        if row["cohort"].startswith("wildguard"):
            assert details["source_license"] == "ODC-By-1.0"
            assert (
                details["source_access_terms_url"]
                == "https://allenai.org/responsible-use"
            )
        else:
            assert details["source_license"] == "CC-BY-4.0"
