"""Shared decision contract for provider adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Status = Literal["ok", "invalid_output", "refusal", "api_error", "truncated", "scoring_miss"]


@dataclass(frozen=True)
class Decision:
    label: str | None
    probs: dict[str, float] | None
    raw: dict[str, Any]
    status: Status
    latency_ms: float
    usage: dict[str, int | float | None] = field(default_factory=dict)

    def validate(self, options: list[str]) -> None:
        if self.status == "ok" and self.label not in options:
            raise ValueError("An ok decision must select an allowed option")
        if self.status != "ok" and self.label is not None:
            raise ValueError("A failed decision must have label=None")
        if self.probs is not None:
            if set(self.probs) != set(options):
                raise ValueError("Probability keys must match all options")
            if any(not 0 <= p <= 1 for p in self.probs.values()):
                raise ValueError("Probabilities must lie in [0, 1]")
            if abs(sum(self.probs.values()) - 1) > 1e-3:
                raise ValueError("Probabilities must sum to one")


class Adapter(Protocol):
    name: str
    model: str
    model_version: str

    def classify(self, text: str, scheme: str, rubric: str,
                 option_order: list[str]) -> Decision: ...
