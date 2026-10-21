"""Check the staged Git tree for known private artifacts; not a secret/PII audit."""

import csv
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = "analysis/tracked_aggregate_outputs.json"
ROW_FIELDS = {
    "item_id",
    "case_id",
    "job_id",
    "pair_id",
    "dialogue_index",
    "prompt",
    "response",
    "user_text",
    "assistant_text",
    "request",
    "assistant_request",
    "assistant_raw",
    "source_record",
    "processed_record",
    "deployment_id",
    "account_id",
    "organization_id",
    "experiment_url",
}
LOCAL_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "data/models/",
    "training/data/",
    "training/logs/",
    "output/",
    "tmp/",
    "analysis/assets/company-logos/",
)
LOCAL_FILES = {
    "REVIEW_CHECKLIST.md",
    "REVIEW_CHECKLIST.pdf",
    "REVIEW_CHECKLIST.print.html",
    "run/pilot-report-2026-10-01.md",
}


def nested_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from nested_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_fields(child)


def violations(name, content, aggregate_outputs):
    """Return reasons without echoing any potentially private file contents."""
    path = Path(name)
    if name.startswith(LOCAL_PREFIXES) or name in LOCAL_FILES or path.name == ".env":
        return ["local-only artifact"]
    if path.suffix == ".ipynb":
        notebook = json.loads(content)
        if any(
            c.get("outputs") or c.get("execution_count") is not None
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        ):
            return ["saved notebook output or execution state"]
    if not name.startswith("analysis/output/"):
        return []
    if name not in aggregate_outputs:
        return ["generated output has not been added to the reviewed aggregate list"]
    if path.suffix == ".csv":
        fields = set(next(csv.reader(io.StringIO(content.decode())), []))
    elif path.suffix == ".json":
        fields = set(nested_fields(json.loads(content)))
    elif path.suffix in {".png", ".svg", ".html", ".md"}:
        # These need content review before adding their path to the allowlist.
        fields = set()
    else:
        return ["unreviewed generated file type"]
    return [
        f"private or example-level field: {field}"
        for field in sorted(fields & ROW_FIELDS)
    ]


def staged_bytes(name):
    return subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)


def main():
    paths = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")[:-1]
    )
    allowed = set(json.loads(staged_bytes(ALLOWLIST)))
    failures = []
    for name in paths:
        failures.extend(
            f"{name}: {reason}"
            for reason in violations(name, staged_bytes(name), allowed)
        )
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"Repository privacy boundary passed for {len(paths)} staged files.")


if __name__ == "__main__":
    main()
