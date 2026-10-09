"""Read newline-delimited JSON without splitting Unicode separators in strings."""
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
