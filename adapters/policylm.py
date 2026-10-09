"""PolicyLM custom-policy prompt-only reference; native scores are not choices."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from adapters.base import Decision

ROOT = Path(__file__).resolve().parents[1]
MODEL = "musubilabs/policylm-1.7b"
REVISION = "4e5a98c579e6ab28c45af42e7d69a84da38f9b12"
MODEL_DIR = ROOT / "data/models/policylm-1.7b"
PRESET = "precision"
CUTOFF = 0.335  # Release v1.2 explicit-mode precision preset; never fitted here.


def make_prompt_request(prompt: str, instructions: str, definitions: dict[str, str]) -> dict:
    if set(definitions) != {"safe", "unsafe"}:
        raise ValueError("PolicyLM prompt reference is binary only")
    # Explicitly adapt the companion's context directive to the model's supported
    # single-message input. Preserve all remaining policy clauses and log the mapping.
    policy_text = instructions.replace(
        "Read the Assistant turn as context, but judge what the User requested or expressed.",
        "Judge only the supplied User message.")
    category = {"title": "Unsafe User content",
                "violation_rule": " ".join((policy_text + " " + definitions['unsafe']).split()),
                "not_violation_rule": " ".join(definitions['safe'].split())}
    return {"model": MODEL, "revision": REVISION, "target": "prompt_harm",
            "input_context": "user_only", "message": prompt,
            "policy": {"categories": [category], "calls": "together"},
            "policy_mapping": "prompt_harm_s2_to_single_category_v1",
            "cutoff_preset": PRESET, "cutoff": CUTOFF, "clean": False,
            "dtype": "float32", "max_chars": 32000, "max_batch_rows": 1}


def load_model(directory: Path = MODEL_DIR):
    # Review/downloaded release code is pinned by its distribution checksums;
    # verify the inference files before importing them. The helper additionally
    # binds tokenizer/config/thresholds to the weight sha256 at load time.
    manifest = json.loads((directory / "distribution_manifest.json").read_text())
    for entry in manifest['files']:
        if entry['path'].startswith('inference/'):
            file = directory / entry['path']
            if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
                raise RuntimeError(f"PolicyLM release code checksum mismatch: {entry['path']}")
    sys.path.insert(0, str(directory / "inference"))
    helper = importlib.import_module("policylm_infer")
    if Path(helper.__file__).resolve() != (directory / "inference/policylm_infer.py").resolve():
        raise RuntimeError("Unexpected PolicyLM helper already imported")
    return helper.PolicyLM.from_pretrained(directory, local_files_only=True,
        device="auto", dtype="float32", clean=False, cutoff=PRESET,
        max_chars=32000, max_batch_rows=1)


def parse_result(result: Any, request: dict, latency_ms: float) -> Decision:
    response = result.to_dict()
    raw = {"request": request, "response": response}
    try:
        score, cutoff = float(response['score']), float(response['cutoff'])
        flagged = response['flagged']
        if (not math.isfinite(score) or not 0 <= score <= 1
                or not math.isclose(cutoff, CUTOFF, abs_tol=1e-7)
                or not isinstance(flagged, bool) or flagged != (score >= cutoff)):
            raise ValueError("Invalid native score/cutoff")
    except (KeyError, TypeError, ValueError):
        return Decision(None, None, raw, "invalid_output", latency_ms)
    # Do not normalize category scores or manufacture an enum distribution.
    return Decision("unsafe" if flagged else "safe", None, raw, "ok", latency_ms,
                    {"input_characters": len(request['message']),
                     "windows": response.get('windows', 1)})


class PolicyLMPromptAdapter:
    name = "policylm_17b_prompt"
    model = MODEL
    model_version = f"hf:{REVISION}"

    def __init__(self, definitions: dict[str, str], model: Any = None):
        self.definitions = dict(definitions)
        self.native_model = model if model is not None else load_model()

    def classify_prompt(self, prompt: str, instructions: str) -> Decision:
        request = make_prompt_request(prompt, instructions, self.definitions)
        start = time.perf_counter()
        try:
            response = self.native_model.classify(prompt, request['policy'], cutoff=PRESET, clean=False)
        except Exception as exc:
            return Decision(None, None, {"request":request, "error":f"{type(exc).__name__}: {exc}"},
                            "api_error", (time.perf_counter()-start)*1000)
        return parse_result(response, request, (time.perf_counter()-start)*1000)
