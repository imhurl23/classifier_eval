"""Shared model labels and cost–quality logic for the revised analysis."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/output/source-policy-revision-2026-10-19"
NAMES = {
    "jev_choice": "Jev · Choice",
    "jev_noul": "Jev · Noul",
    "clef_choice": "Clef",
    "clef_flash_choice": "Clef Flash",
    "tev1_native": "Tev1",
    "decisions_luna": "Luna · decision API",
    "luna_structured": "Luna · structured output",
    "claude_structured": "Claude",
    "qwen35_4b_letters": "Qwen · letters",
    "qwen35_4b_json": "Qwen · JSON",
    "safeguard_120b": "Safeguard 120B",
}
TYPES = {
    a: (
        "Decision model"
        if a
        in {"jev_choice", "jev_noul", "clef_choice", "clef_flash_choice", "tev1_native"}
        else "Safety model"
        if a == "safeguard_120b"
        else "General LLM"
    )
    for a in NAMES
}
COLORS = {
    "Decision model": "#2C1FEB",
    "General LLM": "#956BDD",
    "Safety model": "#A15B96",
}


def pareto_mask(cost, quality):
    cost, quality = np.asarray(cost, float), np.asarray(quality, float)
    assert (
        len(cost) == len(quality)
        and np.isfinite(cost).all()
        and np.isfinite(quality).all()
    )
    return np.array(
        [
            not np.any((cost <= x) & (quality >= y) & ((cost < x) | (quality > y)))
            for x, y in zip(cost, quality)
        ]
    )
