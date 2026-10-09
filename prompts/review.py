"""Render each draft rubric for every scheme and check length controls."""
from __future__ import annotations

from pathlib import Path
import json

import tiktoken
from tokenizers import Tokenizer
import yaml

from prompts.runtime import instructions_for, options_for

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ("r_short", "r_long", "r_neutral_long", "r_withheld")


def review():
    ontology = yaml.safe_load((ROOT / "data/label_map.yaml").read_text())
    definitions = yaml.safe_load((ROOT / "prompts/label_definitions.yaml").read_text())
    long_template = (ROOT / "prompts/rubrics/r_long.md").read_text()
    shared_openings = {
        rubric: (ROOT / f"prompts/rubrics/{rubric}.md").read_text().split("\n\n", 1)[0]
        for rubric in ("r_short", "r_long", "r_neutral_long")
    }
    if len(set(shared_openings.values())) != 1:
        raise ValueError("Short, Long, and Neutral-Long must share an identical policy opening")
    if any(marker in long_template.lower() for marker in
           ("aegis-validation-", "dev item", "dataset's binary label", "aggregate category field")):
        raise ValueError("Long rubric must contain examples without dataset identifiers or annotation metadata")
    qwen = Tokenizer.from_file(str(ROOT / "data/raw/qwen-tokenizer/tokenizer.json"))
    gptoss = Tokenizer.from_file(str(ROOT / "data/raw/gpt-oss-tokenizer/tokenizer.json"))
    o200k = tiktoken.get_encoding("o200k_base")
    result = {}
    for scheme, labels in ontology["schemes"].items():
        if missing := set(labels) - set(definitions):
            raise ValueError(f"Missing definitions for {scheme}: {sorted(missing)}")
        shared = "\n".join(f"{label}: {definitions[label]}" for label in labels)
        counts = {}
        for rubric in RUBRICS:
            template = (ROOT / f"prompts/rubrics/{rubric}.md").read_text()
            rendered = template.format(allowed_labels=", ".join(labels),
                                       label_definitions=shared, text="[ITEM_TEXT]")
            counts[rubric] = {
                "qwen35": len(qwen.encode(rendered, add_special_tokens=False).ids),
                "gptoss": len(gptoss.encode(rendered, add_special_tokens=False).ids),
                "o200k": len(o200k.encode(rendered)),
            }
        for tokenizer in ("qwen35", "gptoss", "o200k"):
            long = counts["r_long"][tokenizer]
            neutral = counts["r_neutral_long"][tokenizer]
            if abs(neutral - long) / long > 0.03:
                raise ValueError(f"{scheme} {tokenizer}: Neutral-Long differs from Long by more than 3%")
        # Typed decision models receive the interaction once in `state` and
        # definitions separately in criteria. Check the real semantic payload.
        for tokenizer_name, encoder in (("qwen35", lambda s: len(qwen.encode(s, add_special_tokens=False).ids)),
                                        ("gptoss", lambda s: len(gptoss.encode(s, add_special_tokens=False).ids)),
                                        ("o200k", lambda s: len(o200k.encode(s)))):
            runtime_counts = {}
            for rubric in ("r_long", "r_neutral_long"):
                _, criteria = options_for(scheme, rubric)
                runtime_counts[rubric] = encoder(instructions_for(rubric) + "\n" +
                                                 "\n".join(criteria.values()))
            long = runtime_counts["r_long"]
            neutral = runtime_counts["r_neutral_long"]
            if abs(neutral - long) / long > 0.03:
                raise ValueError(f"{scheme} {tokenizer_name}: Runtime Neutral-Long differs by over 3%")
        result[scheme] = counts
    return result


if __name__ == "__main__":
    print(json.dumps(review(), indent=2))
