"""Native GPT-6 Luna Choices via the direct OpenAI Decisions HTTP API."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math
import time
from typing import Any

import requests

from adapters.base import Decision

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"


def load_openai_key() -> str:
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    key = values.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing from this repository's .env")
    return key


def make_multi_choice_request(text: str, questions: Mapping[str, Mapping[str, Any]]) -> dict:
    if not questions:
        raise ValueError("At least one question is required")
    rendered = []
    for name, spec in questions.items():
        criteria = spec.get("criteria")
        instructions = spec.get("instructions")
        if (not isinstance(name, str) or not isinstance(instructions, str)
                or not isinstance(criteria, Mapping) or len(criteria) < 2
                or any(not isinstance(k, str) or not isinstance(v, str)
                       for k, v in criteria.items())):
            raise ValueError(f"Invalid Choice question: {name}")
        rendered.append({"type": "choice", "name": name, "instructions": instructions,
                         "choices": [{"value": label, "description": description}
                                     for label, description in criteria.items()]})
    return {"model": MODEL, "input": text, "questions": rendered}


def make_choice_request(text: str, rubric: str, option_order: list[str],
                        definitions: Mapping[str, str]) -> dict:
    if len(option_order) != len(set(option_order)) or set(option_order) != set(definitions):
        raise ValueError("Definitions must cover exactly the distinct offered options")
    return make_multi_choice_request(text, {"safety": {
        "instructions": rubric, "criteria": {label: definitions[label] for label in option_order}}})


def parse_multi_choice(payload: Any, request: dict, latency_ms: float) -> dict[str, Decision]:
    answers = payload.get("answers") if isinstance(payload, Mapping) else None
    usage = payload.get("usage", {}) if isinstance(payload, Mapping) else {}
    usage = usage if isinstance(usage, dict) else {}
    raw = {"request": request, "response": payload}
    results = {}
    for index, question in enumerate(request["questions"]):
        options = [c["value"] for c in question["choices"]]
        label, probs, status = None, None, "invalid_output"
        if isinstance(answers, list) and len(answers) == len(request["questions"]):
            answer = answers[index]
            if isinstance(answer, Mapping) and answer.get("name") == question["name"]:
                if answer.get("type") == "refusal":
                    status = "refusal"
                elif answer.get("type") == "choice":
                    try:
                        entries = answer["probabilities"]
                        if not isinstance(entries, list) or len(entries) != len(options):
                            raise ValueError("Incomplete probabilities")
                        probs = {p["value"]: float(p["probability"]) for p in entries}
                        if any(not math.isfinite(p) for p in probs.values()):
                            raise ValueError("Non-finite probability")
                        label = answer["choice"]
                        Decision(label, probs, raw, "ok", latency_ms, usage).validate(options)
                        status = "ok"
                    except (KeyError, TypeError, ValueError):
                        label, probs = None, None
        results[question["name"]] = Decision(label, probs, raw, status, latency_ms, usage)
    return results


class OpenAIDecisionsAdapter:
    name = "decisions_luna"
    model = MODEL

    def __init__(self, definitions: Mapping[str, str], session: requests.Session | None = None):
        self.definitions = dict(definitions)
        self.model_version = MODEL
        self.session = session or requests.Session()

    def _call(self, request: dict) -> dict[str, Decision]:
        key = load_openai_key()
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.session.post(API_URL, headers={"Authorization": f"Bearer {key}"},
                                             json=request, timeout=60)
                if response.status_code in {429} | set(range(500, 600)) and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"non_json_response": response.text[:2000].replace(key, "[REDACTED]")}
                error = not response.ok
                http_status = response.status_code
            except requests.RequestException as exc:
                payload = {"error": str(exc).replace(key, "[REDACTED]")}
                error, http_status = True, None
            latency_ms = (time.perf_counter() - started) * 1000
            if error:
                return {q["name"]: Decision(None, None, {"request": request, "response": payload,
                                                         "http_status": http_status},
                                              "api_error", latency_ms) for q in request["questions"]}
            if isinstance(payload, Mapping) and isinstance(payload.get("model"), str):
                self.model_version = payload["model"]
            return parse_multi_choice(payload, request, latency_ms)
        raise AssertionError("Retry loop exhausted")

    def classify(self, text: str, scheme: str, rubric: str, option_order: list[str]) -> Decision:
        del scheme
        return self._call(make_choice_request(text, rubric, option_order, self.definitions))["safety"]

    def classify_many(self, text: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Decision]:
        return self._call(make_multi_choice_request(text, questions))
