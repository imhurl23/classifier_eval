"""Pinned TypeSafe Jev Choice adapter using the native System One API."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time

import requests

from adapters.base import Decision
from adapters.clef import make_choice_request, make_multi_choice_request, parse_choice, parse_multi_choice

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.typesafe.ai/v1/systemone"


def load_typesafe_key() -> str:
    """Use the key in this repository's .env, never an ambient fallback."""
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == "TYPESAFE_API_KEY":
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("TYPESAFE_API_KEY is missing from this repository's .env")


class JevChoiceAdapter:
    name = "jev_choice"
    model = "jev-1.13.0"
    model_version = "jev-1.13.0"

    def __init__(self, definitions: Mapping[str, str],
                 session: requests.Session | None = None):
        self.definitions = dict(definitions)
        self.session = session or requests.Session()

    def classify(self, text: str, scheme: str, rubric: str,
                 option_order: list[str]) -> Decision:
        del scheme
        key = load_typesafe_key()
        request = make_choice_request(text, rubric, option_order,
                                      self.definitions, self.model)
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.session.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json=request, timeout=45)
            except requests.RequestException as exc:
                return Decision(None, None, {"request": request, "error": str(exc)},
                                "api_error", (time.perf_counter() - started) * 1000)
            if response.status_code in ({429, 529} | set(range(500, 600))) and attempt < 2:
                time.sleep(0.5 * 2 ** attempt)
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = {"non_json_response": response.text[:2000]}
            latency = (time.perf_counter() - started) * 1000
            if not response.ok:
                return Decision(None, None,
                                {"request": request, "response": payload,
                                 "http_status": response.status_code},
                                "api_error", latency)
            return parse_choice(payload, option_order, request, latency)
        raise AssertionError("Retry loop exhausted")

    def classify_many(self, text: str, questions: Mapping[str, Mapping]) -> dict[str, Decision]:
        """Ask separate typed Choice questions about the same full interaction."""
        key = load_typesafe_key()
        request = make_multi_choice_request(text, questions, self.model)
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.session.post(
                    API_URL, headers={"Authorization": f"Bearer {key}"},
                    json=request, timeout=60)
            except requests.RequestException as exc:
                status, payload = "api_error", {"error": str(exc)}
            else:
                if response.status_code in ({429, 529} | set(range(500, 600))) and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"non_json_response": response.text[:2000]}
                status = "ok" if response.ok else "api_error"
            latency = (time.perf_counter() - started) * 1000
            if status != "ok":
                return {name: Decision(None, None, {"request": request, "response": payload},
                                       "api_error", latency) for name in questions}
            return parse_multi_choice(payload, questions, request, latency)
        raise AssertionError("Retry loop exhausted")


def make_noul_request(text: str, rubric: str, definitions: Mapping[str, str]) -> dict:
    request = {"model": "jev-1.13.0", "state": text,
               "questions": {"unsafe": {"type": "noul",
                                        "instructions": rubric + "\nIs the supplied interaction unsafe?",
                                        "criteria": {"true": definitions["unsafe"],
                                                     "false": definitions["safe"]}}}}
    return request


class JevNoulAdapter:
    """Binary Jev Noul probability for interaction unsafety."""
    name = "jev_noul"
    model = "jev-1.13.0"
    model_version = "jev-1.13.0"

    def __init__(self, definitions: Mapping[str, str], session: requests.Session | None = None):
        if set(definitions) != {"safe", "unsafe"}:
            raise ValueError("Noul requires safe and unsafe definitions")
        self.definitions = dict(definitions)
        self.session = session or requests.Session()

    def classify(self, text: str, scheme: str, rubric: str,
                 option_order: list[str]) -> Decision:
        if scheme != "s2" or set(option_order) != {"safe", "unsafe"}:
            raise ValueError("Noul is binary only")
        key = load_typesafe_key()
        request = make_noul_request(text, rubric, self.definitions)
        started = time.perf_counter()
        try:
            response = self.session.post(API_URL, headers={"Authorization": f"Bearer {key}"},
                                         json=request, timeout=45)
        except requests.RequestException as exc:
            return Decision(None, None, {"request": request, "error": str(exc)},
                            "api_error", (time.perf_counter() - started) * 1000)
        latency = (time.perf_counter() - started) * 1000
        try:
            payload = response.json()
        except ValueError:
            payload = {"non_json_response": response.text[:2000]}
        raw = {"request": request, "response": payload}
        if not response.ok:
            return Decision(None, None, raw, "api_error", latency)
        try:
            p = float(payload["answers"]["unsafe"]["noul"])
            if not 0 <= p <= 1:
                raise ValueError("Probability out of range")
        except (KeyError, TypeError, ValueError):
            return Decision(None, None, raw, "invalid_output", latency)
        label = "unsafe" if p >= 0.5 else "safe"
        decision = Decision(label, {"safe": 1-p, "unsafe": p}, raw, "ok", latency,
                            payload.get("usage", {}))
        decision.validate(option_order)
        return decision
