"""Braintrust credential loader. .env is the sole permitted key source."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def load_braintrust_key():
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    key = values.get("BRAINTRUST_API_KEY")
    if not key:
        raise RuntimeError("BRAINTRUST_API_KEY is missing from .env")
    os.environ["BRAINTRUST_API_KEY"] = key
    return key
