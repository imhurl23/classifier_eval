"""Structured JSON interface on the same Qwen3.5-4B Baseten deployment as letters."""
from __future__ import annotations

import json
import os
import time

import requests

from adapters.base import Decision


def make_request(text: str, rubric: str, option_order: list[str],
                 definitions: dict[str, str]) -> dict:
    policy = (f"{rubric}\n\nOptions:\n" +
              "\n".join(f"{label}: {definitions[label]}" for label in option_order) +
              "\nReturn a JSON object with one key, label, whose value is exactly one offered identifier.")
    request = {
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "system", "content": policy},
                     {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "decision", "strict": True,
            "schema": {"type": "object", "properties": {
                "label": {"type": "string", "enum": option_order}},
                "required": ["label"], "additionalProperties": False}}},
    }
    return request


class QwenJsonAdapter:
    name = "qwen35_4b_json"
    model = "Qwen/Qwen3.5-4B"
    model_version = "hf:851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

    def __init__(self, definitions: dict[str, str], session: requests.Session | None = None):
        self.definitions = definitions
        self.model_id = os.getenv("BASETEN_QWEN35_MODEL_ID")
        self.deployment_id = os.getenv("BASETEN_QWEN35_DEPLOYMENT_ID")
        self.api_key = os.getenv("BASETEN_API_KEY")
        self.session = session or requests.Session()

    def classify(self, text: str, scheme: str, rubric: str,
                 option_order: list[str]) -> Decision:
        del scheme
        if not self.api_key or not self.model_id:
            raise RuntimeError("Baseten API key and Qwen model ID required")
        request = make_request(text, rubric, option_order, self.definitions)
        started = time.perf_counter()
        try:
            url = (f"https://model-{self.model_id}.api.baseten.co/deployment/{self.deployment_id}/predict"
                   if self.deployment_id else
                   f"https://model-{self.model_id}.api.baseten.co/environments/production/sync/v1/chat/completions")
            response = self.session.post(
                url,
                headers={"Authorization": f"Api-Key {self.api_key}"}, json=request, timeout=120)
            try:
                payload = response.json()
            except ValueError:
                payload = {"non_json_response": response.text[:1000]}
        except requests.RequestException as exc:
            return Decision(None, None, {"request": request, "error": str(exc)},
                            "api_error", (time.perf_counter()-started)*1000)
        latency = (time.perf_counter()-started)*1000
        raw = {"request": request, "response": payload, "http_status": response.status_code}
        if not response.ok:
            return Decision(None, None, raw, "api_error", latency)
        try:
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                return Decision(None, None, raw, "truncated", latency, payload.get("usage") or {})
            obj = json.loads(choice["message"]["content"])
            label = obj["label"]
        except (KeyError, IndexError, TypeError, ValueError):
            return Decision(None, None, raw, "invalid_output", latency, payload.get("usage") or {})
        if label not in option_order:
            return Decision(None, None, raw, "invalid_output", latency, payload.get("usage") or {})
        return Decision(label, None, raw, "ok", latency, payload.get("usage") or {})
