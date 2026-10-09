"""Enumerate planned experiment cells without initiating paid model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ("r_short", "r_long", "r_neutral_long", "r_withheld")
ARMS = {
    "jev_choice": ("s2", "s3", "s6", "s13"),
    "jev_noul": ("s2",),
    "clef_choice": ("s2", "s3", "s6", "s13"),
    "clef_flash_choice": ("s2", "s3", "s6", "s13"),
    "luna_structured": ("s2", "s3", "s6", "s13"),
    "decisions_luna": ("s2", "s3", "s6", "s13"),
    "claude_structured": ("s2", "s3", "s6", "s13"),
    "tev1_native": ("s2", "s3", "s6", "s13"),
    "qwen35_4b_letters": ("s2", "s3", "s6", "s13"),
    "safeguard_120b": ("s2", "s3", "s6", "s13"),
}
OPTIONAL = {
    "qwen35_4b_json": ("s2", "s3", "s6", "s13"),
    "safety_tev": ("s2", "s3", "s6", "s13"),
    "cc_repro": ("s2", "s3", "s6", "s13"),
}
NATIVE = {
    "qwen3guard_8b": ("s3",),
    "nemotron_guard": ("s2",),
}
BYOC = {"granite_41_8b": ("s2",)}
RESPONSE_COMPANION_RUBRIC = "response_harm_s2"
COMPANION_RUBRICS = {
    "prompt_harm": "prompt_harm_s2",
    "response_harm": "response_harm_s2",
    "response_refusal": "response_refusal_s2",
}


def cells(include_optional=False):
    arms = ARMS | (OPTIONAL if include_optional else {})
    for arm, schemes in arms.items():
        for scheme in schemes:
            for rubric in RUBRICS:
                yield {"arm": arm, "scheme": scheme, "rubric": rubric,
                       "cell_id": f"{arm}__{scheme}__{rubric}"}
    for arm, schemes in (BYOC | NATIVE).items():
        for scheme in schemes:
            rubric = "byoc" if arm in BYOC else "native"
            yield {"arm": arm, "scheme": scheme, "rubric": rubric,
                   "decision_mode": "derived_or",
                   "cell_id": f"{arm}__{scheme}__{rubric}"}


def response_companion_cells(include_optional=False):
    """Supported response-harm tasks, including explicitly marked native arms."""
    yield from (cell for cell in companion_cells(include_optional)
                if cell["target"] == "response_harm")


def companion_cells(include_optional=False):
    """Only implemented targets; native user-only context is explicitly marked."""
    arms = ARMS | (OPTIONAL if include_optional else {}) | BYOC | NATIVE
    for arm, schemes in arms.items():
        if arm == "jev_noul":
            continue  # Current adapter implements interaction classification only.
        for target, rubric in COMPANION_RUBRICS.items():
            if arm in BYOC | NATIVE and target == "response_refusal":
                continue
            mode = "byoc" if arm in BYOC else "native" if arm in NATIVE else "common_policy"
            scheme = schemes[0] if arm in NATIVE else "s2"
            rendered_rubric = mode if mode != "common_policy" else rubric
            yield {"arm": arm, "scheme": scheme, "rubric": rendered_rubric,
                   "unit": "conversation", "source": "wildguardtest", "target": target,
                   "policy_mode": mode,
                   "input_context": "user_only" if arm == "qwen3guard_8b" and target == "prompt_harm" else "full_pair",
                   "cell_id": f"{arm}__{scheme}__{rendered_rubric}__{target}"}
    # PolicyLM's authors exclude conversation history and Assistant moderation.
    # Its explicit-policy binary score is a supplemental User-only reference.
    yield {"arm": "policylm_17b_prompt", "scheme": "s2", "rubric": "prompt_harm_s2",
           "unit": "conversation", "source": "wildguardtest", "target": "prompt_harm",
           "policy_mode": "custom_policy_prompt_only", "input_context": "user_only",
           "cell_id": "policylm_17b_prompt__s2__prompt_harm_s2__prompt_harm"}


def matched_option_order(item_id: str, target: str, scheme: str,
                         options: list[str]) -> list[str]:
    """Match order across arms, rubrics, and batching; rotations are separate treatments."""
    return option_order(item_id, f"matched_v1__{target}__{scheme}", sorted(options))


def option_order(item_id: str, cell_id: str, options: list[str]) -> list[str]:
    seed = int.from_bytes(hashlib.sha256(f"{item_id}\0{cell_id}".encode()).digest()[:8], "big")
    order = list(options)
    random.Random(seed).shuffle(order)
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Print the pilot plan; execution requires verified adapters and rubrics")
    parser.add_argument("--primary", action="store_true", help="Print the primary plan")
    parser.add_argument("--controls", action="store_true", help="Print the controls plan")
    parser.add_argument("--list", action="store_true", help="List planned cells without running them")
    parser.add_argument("--response-companion", action="store_true", help="List fixed WildGuardTest response-harm cells")
    parser.add_argument("--companions", action="store_true", help="List all three WildGuardTest turn/refusal decision cells")
    args = parser.parse_args()
    if args.response_companion and args.companions:
        parser.error("Choose one companion selector")
    plan = list(companion_cells(args.include_optional) if args.companions else
                response_companion_cells(args.include_optional) if args.response_companion else
                cells(args.include_optional))
    manifest_path = ROOT / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if args.primary and not manifest["datasets"]["dis-test"]["braintrust_version"]:
            raise SystemExit("Test dataset has no pinned Braintrust version")
    if args.pilot or args.primary or args.controls:
        raise SystemExit("Execution is gated: provider adapters, rubric versions, and preregistration are not yet frozen. Use --list to inspect cells.")
    print(json.dumps({"cell_count": len(plan), "cells": plan}, indent=2))


if __name__ == "__main__":
    main()
