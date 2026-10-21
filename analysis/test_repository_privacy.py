import json

from analysis.check_repository_privacy import violations


def test_generated_prediction_cannot_enter_as_aggregate():
    path = "analysis/output/new/metrics.csv"
    assert violations(path, b"item_id,label\nexample,safe\n", {path})
    assert not violations(path, b"model,total,correct\nmodel,100,90\n", {path})


def test_nested_source_text_and_operational_metadata_are_rejected():
    path = "analysis/output/new/summary.json"
    for field in ("source_record", "request", "deployment_id"):
        content = json.dumps({"nested": [{field: "synthetic fixture"}]}).encode()
        assert violations(path, content, {path})


def test_unknown_outputs_and_private_paths_fail_closed():
    for path in (
        "analysis/output/new/report.html",
        "data/raw/source.json",
        "tmp/review.md",
        ".env",
    ):
        assert violations(path, b"", set())


def test_notebook_output_is_rejected_without_rejecting_clean_code():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["print(1)"],
                "outputs": [],
                "execution_count": None,
            }
        ]
    }
    assert not violations(
        "analysis/example.ipynb", json.dumps(notebook).encode(), set()
    )
    notebook["cells"][0]["outputs"] = [
        {"output_type": "stream", "text": "private fixture"}
    ]
    assert violations("analysis/example.ipynb", json.dumps(notebook).encode(), set())
