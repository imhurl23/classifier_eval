"""Cloudflare Workers AI Clef choice adapter.

Contract: https://developers.cloudflare.com/workers-ai/models/clef/
The hosted model revision must be pinned or dated in experiment metadata before a
preregistered run. HF weight revisions are recorded separately from hosted builds.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
import time
from typing import Any

import requests

from adapters.base import Decision

MODEL_INFO = {
    "clef_choice": ("clef", "@cf/cloudflare/clef",
                    "2f3de3dd85f379784083b0814d997ab627200f0c"),
    "clef_flash_choice": ("clef-flash", "@cf/cloudflare/clef-flash",
                          "17f0b0ad64efb65d273590632833508766b2aae6"),
}


def make_choice_request(text: str, instructions: str,
                        option_order: list[str], definitions: Mapping[str, str],
                        model: str) -> dict[str, Any]:
    if len(option_order) < 2 or len(option_order) != len(set(option_order)):
        raise ValueError("Choice requires distinct options")
    if set(option_order) != set(definitions):
        raise ValueError("Definitions must cover exactly the offered options")
    return {
        "model": model,
        "state": text,
        "questions": {"safety": {
            "type": "choice", "instructions": instructions,
            "criteria": {label: definitions[label] for label in option_order},
        }},
    }


def make_multi_choice_request(text: str, questions: Mapping[str, Mapping[str, Any]],
                              model: str) -> dict[str, Any]:
    """One state, several independently worded Choice decisions."""
    if not questions:
        raise ValueError("At least one question is required")
    rendered = {}
    for question_id, spec in questions.items():
        criteria = spec.get("criteria")
        instructions = spec.get("instructions")
        if not isinstance(criteria, Mapping) or len(criteria) < 2 or not isinstance(instructions, str):
            raise ValueError(f"Invalid Choice question: {question_id}")
        rendered[question_id] = {"type": "choice", "instructions": instructions,
                                 "criteria": dict(criteria)}
    return {"model": model, "state": text, "questions": rendered}


def parse_multi_choice(payload: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]],
                       request: dict[str, Any], latency_ms: float) -> dict[str, Decision]:
    result = {}
    response = payload.get("result", payload)
    answers = response.get("answers", {}) if isinstance(response, Mapping) else {}
    for question_id, spec in questions.items():
        options = list(spec["criteria"])
        single_response = {"answers": {"safety": answers.get(question_id)},
                           "usage": response.get("usage", {}) if isinstance(response, Mapping) else {}}
        result[question_id] = parse_choice(single_response, options, request, latency_ms)
    return result


def parse_choice(payload: Mapping[str, Any], options: list[str],
                 request: dict[str, Any], latency_ms: float) -> Decision:
    response = payload.get("result", payload)
    answer = response.get("answers", {}).get("safety") if isinstance(response, Mapping) else None
    status = "ok"
    label = None
    probs = None
    if isinstance(answer, Mapping):
        label = answer.get("choice")
        raw_probs = answer.get("probabilities")
        if isinstance(raw_probs, Mapping):
            try:
                probs = {option: float(raw_probs[option]) for option in options}
            except (KeyError, TypeError, ValueError):
                probs = None
    if label not in options or probs is None:
        status, label, probs = "invalid_output", None, None
    usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
    decision = Decision(label=label, probs=probs,
                        raw={"request": request, "response": payload},
                        status=status, latency_ms=latency_ms,
                        usage=usage if isinstance(usage, dict) else {})
    try:
        decision.validate(options)
    except ValueError:
        return Decision(label=None, probs=None, raw=decision.raw,
                        status="invalid_output", latency_ms=latency_ms,
                        usage=decision.usage)
    return decision


class ClefChoiceAdapter:
    def __init__(self, arm: str, definitions: Mapping[str, str],
                 account_id: str | None = None, auth_token: str | None = None,
                 session: requests.Session | None = None):
        if arm not in MODEL_INFO:
            raise ValueError(f"Unknown Clef arm {arm!r}")
        self.name = arm
        self.model, self.endpoint_model, self.weights_revision = MODEL_INFO[arm]
        self.model_version = "workers-ai-unpinned"
        self.definitions = dict(definitions)
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.auth_token = auth_token or os.getenv("CLOUDFLARE_AUTH_TOKEN")
        self.session = session or requests.Session()

    def classify(self, text: str, scheme: str, rubric: str,
                 option_order: list[str]) -> Decision:
        del scheme  # Allowed labels and definitions are supplied by the caller.
        if not self.account_id or not self.auth_token:
            raise RuntimeError("Cloudflare account ID and auth token are required for Clef")
        request = make_choice_request(text, rubric, option_order,
                                      self.definitions, self.model)
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{self.account_id}/ai/run/{self.endpoint_model}")
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.session.post(
                    url, headers={"Authorization": f"Bearer {self.auth_token}"},
                    json=request, timeout=60)
            except requests.RequestException as exc:
                return Decision(None, None, {"request": request, "error": str(exc)},
                                "api_error", (time.perf_counter() - started) * 1000)
            if response.status_code in {429} | set(range(500, 600)) and attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = {"non_json_response": response.text[:2000]}
            latency = (time.perf_counter() - started) * 1000
            if not response.ok or payload.get("success") is False:
                return Decision(None, None,
                                {"request": request, "response": payload,
                                 "http_status": response.status_code},
                                "api_error", latency)
            return parse_choice(payload, option_order, request, latency)
        raise AssertionError("Retry loop exhausted")

    def classify_many(self, text: str, questions: Mapping[str, Mapping]) -> dict[str, Decision]:
        """Ask several typed Choice questions in one Workers AI request."""
        if not self.account_id or not self.auth_token:
            raise RuntimeError("Cloudflare account ID and auth token are required for Clef")
        request = make_multi_choice_request(text, questions, self.model)
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{self.account_id}/ai/run/{self.endpoint_model}")
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.session.post(url, headers={"Authorization": f"Bearer {self.auth_token}"},
                                             json=request, timeout=60)
            except requests.RequestException as exc:
                status, payload = "api_error", {"error": str(exc)}
            else:
                if response.status_code in {429} | set(range(500, 600)) and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"non_json_response": response.text[:2000]}
                status = "ok" if response.ok and payload.get("success") is not False else "api_error"
            latency = (time.perf_counter() - started) * 1000
            if status != "ok":
                return {name: Decision(None, None, {"request": request, "response": payload},
                                       "api_error", latency) for name in questions}
            return parse_multi_choice(payload, questions, request, latency)
        raise AssertionError("Retry loop exhausted")
