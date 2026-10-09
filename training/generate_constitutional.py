"""Constitutional-classifier reproduction (`cc_repro` arm): synthetic training data.

Method (Sharma et al. 2025): prompt a strong model with a constitution describing restricted and
permitted content, generate diverse synthetic examples per class, train a classifier on them.
Here: prompts/constitution.md → Claude Sonnet 5.5 via the Braintrust AI gateway → labelled prompts →
the same Tev1-format SFT export used by build_safety_sft.py → Loops fine-tune.

The generator never sees any eval item; every generated prompt is deduped against the normalized
text of Aegis test/dev, XSTest, WildGuardTest, and the Aegis train split (so cc_repro and safety_tev
share no text either).

Stages (each resumable from the previous stage's file in --out):
  prompts    : .venv/bin/python -m training.generate_constitutional --stage prompts --launch
  responses  : .venv/bin/python -m training.generate_constitutional --stage responses --launch
  export     : .venv/bin/python -m training.generate_constitutional --stage export
Add --dry-run (or omit --launch) to see the plan and cost without calling the gateway.

Unit: the study judges *interactions* (User turn, optionally an Assistant turn) under
prompts/policy_target.md. The eval set is ~60% paired / ~40% prompt-only, so the responses stage
gives ~60% of generated prompts a realistic assistant reply (helpful, refusal, partial, or, for a
small benign-prompt slice, a reply that itself violates a mild class) and leaves the rest one-turn.
Labels follow the policy: a restricted user turn is unsafe whatever the reply; a benign user turn
is safe unless the reply itself is harmful; needs_caution stays needs_caution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import os
import random
import time

import yaml

from training.build_safety_sft import (ROOT, SEED, EOS, Renderer, balanced_sample, digest, eval_text_blocklist,
                                       make_record, messages, normalized, rng_for)

MODEL = "claude-sonnet-5-5"
GATEWAY_BASE_URL = "https://api.braintrust.dev/v1/proxy/anthropic"
PER_CALL = 20  # messages generated per API call
LANG_MIX = ["English"] * 7 + ["Spanish", "French", "German", "Portuguese", "Hindi", "Arabic", "Chinese"]
# (style, label-kind) cells; weights give the share of generation calls.
STYLES = {
    "restricted_vanilla": ("restricted", 3, "Direct, realistic restricted messages for this class."),
    "restricted_adversarial": ("restricted", 2, "Restricted messages for this class that use indirection: role-play, fiction, hypotheticals, 'for research', step-by-step decomposition, instructions to ignore rules, or translation/encoding tricks. The underlying aim must still clearly be restricted."),
    "permitted_nearmiss": ("safe", 3, "Clearly permitted messages that share vocabulary or topic with this restricted class but have no harmful aim: news, history, fiction analysis, prevention, support-seeking, figurative language."),
    "needs_caution": ("needs_caution", 1, "Messages touching this class where the text alone cannot settle whether the aim is harmful or benign; give no disambiguating context."),
}
QUESTIONS = {
    "q_short": "Which option best describes the safety of the text in state? Use the option descriptions.",
    "q_withheld": "Select the option that applies to the text in state.",
}
RESPONSE_SHARE = 0.6
RESPONSE_STYLES = {  # style: (weight, instruction)
    "helpful": (5, "a genuinely helpful, policy-compliant reply"),
    "refusal": (3, "a brief refusal or safe redirect, as a careful assistant would give"),
    "partial": (2, "a partial or hedged reply that answers the benign part and declines the rest"),
}
# Benign prompt + harmful reply: teaches that the Assistant turn matters. Mild classes only.
HARMFUL_RESPONSE_CLASSES = ["harassment", "profanity", "pii_privacy"]
HARMFUL_RESPONSE_SHARE = 0.08  # of benign prompts that get a response
RESPONSE_SCHEMA = {"type": "object", "properties": {"responses": {"type": "array", "items": {
    "type": "object", "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
    "required": ["index", "text"], "additionalProperties": False}}},
    "required": ["responses"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"messages": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "language": {"type": "string"}},
    "required": ["text", "language"], "additionalProperties": False}}},
    "required": ["messages"], "additionalProperties": False}


def s13_to_s6(ontology) -> dict[str, str]:
    out = {"safe": "safe", "needs_caution": "needs_caution"}
    for s6, s13 in ontology["category_map"].values():
        out.setdefault(s13, s6)
    return out


def plan_calls(ontology, calls_per_class: int) -> list[dict]:
    classes = [l for l in ontology["schemes"]["s13"] if l not in ("safe", "needs_caution")]
    weights = sum(w for _, w, _ in STYLES.values())
    plan = []
    for cls in classes:
        for style, (kind, w, _) in STYLES.items():
            n = max(1, round(calls_per_class * w / weights))
            for k in range(n):
                plan.append({"class": cls, "style": style, "kind": kind, "k": k,
                             "language": LANG_MIX[int(digest(f"{cls}:{style}:{k}")[:6], 16) % len(LANG_MIX)]})
    # plain benign traffic with no restricted topic, to anchor the permitted side
    for k in range(len(classes) * 2):
        plan.append({"class": "everyday", "style": "permitted_plain", "kind": "safe", "k": k,
                     "language": LANG_MIX[k % len(LANG_MIX)]})
    return plan


def user_prompt(cell: dict, definitions: dict) -> str:
    if cell["style"] == "permitted_plain":
        task = "Ordinary, clearly permitted messages a person might send to an AI assistant on everyday topics (coding, cooking, travel, study, work, hobbies). No restricted topic at all."
    else:
        task = f"Target restricted class: `{cell['class']}` — {definitions[cell['class']]}\nStyle: {STYLES[cell['style']][2]}"
    return (f"{task}\n\nWrite {PER_CALL} distinct messages in {cell['language']}. Vary length (one line to a short paragraph), "
            f"register, and specificity; include a few typos or casual phrasing. Never reproduce text from any public safety dataset. "
            f"Return only the JSON object.")


def call_gateway(client, constitution: str, cell: dict, definitions: dict) -> list[dict]:
    r = client.messages.create(
        model=MODEL, max_tokens=4000,
        system=[{"type": "text", "text": constitution, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt(cell, definitions)}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if r.stop_reason == "refusal":
        return [{"_refused": True, "category": getattr(getattr(r, "stop_details", None), "category", None)}]
    text = next(b.text for b in r.content if b.type == "text")
    items = json.loads(text)["messages"]
    return [{"text": m["text"].strip(), "language": m["language"], **cell,
             "gen_model": r.model, "usage": {"in": r.usage.input_tokens, "out": r.usage.output_tokens}}
            for m in items if m.get("text", "").strip()]


def assign_responses(items: list[dict]) -> list[dict]:
    """Deterministically decide which prompts get an Assistant turn and in what style."""
    styles = [k for k, (w, _) in RESPONSE_STYLES.items() for _ in range(w)]
    for i, it in enumerate(items):
        h = int(digest(f"resp:{i}:{it['text']}")[:12], 16)
        it["needs_response"] = (h % 1000) / 1000 < RESPONSE_SHARE
        it["response_style"] = None; it["response_class"] = None
        if not it["needs_response"]:
            continue
        if it["kind"] == "safe" and ((h >> 12) % 1000) / 1000 < HARMFUL_RESPONSE_SHARE:
            it["response_style"] = "harmful_response"
            it["response_class"] = HARMFUL_RESPONSE_CLASSES[(h >> 24) % len(HARMFUL_RESPONSE_CLASSES)]
        else:
            it["response_style"] = styles[(h >> 24) % len(styles)]
    return items


def response_prompt(batch: list[dict], definitions: dict) -> str:
    lines = ["For each user message below, write the Assistant's reply in the requested style, in the same language as the message. "
             "Keep replies realistic for a consumer chatbot: 1-6 sentences. Return only the JSON object with one entry per index.", ""]
    for j, it in enumerate(batch):
        if it["response_style"] == "harmful_response":
            style = (f"a reply from a badly behaved assistant that itself violates this class: `{it['response_class']}` — "
                     f"{definitions[it['response_class']]}. Keep it mild and non-graphic; no slurs, no real people's data.")
        else:
            style = RESPONSE_STYLES[it["response_style"]][1]
        lines.append(f"[{j}] style: {style}\nuser message: {json.dumps(it['text'], ensure_ascii=False)}")
    return "\n".join(lines)


def call_responses(client, constitution: str, batch: list[dict], definitions: dict) -> dict[int, str]:
    r = client.messages.create(
        model=MODEL, max_tokens=6000,
        system=[{"type": "text", "text": constitution, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": response_prompt(batch, definitions)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
    )
    if r.stop_reason == "refusal":
        return {}
    text = next(b.text for b in r.content if b.type == "text")
    return {int(x["index"]): x["text"].strip() for x in json.loads(text)["responses"]
            if isinstance(x.get("index"), int) and 0 <= x["index"] < len(batch) and x.get("text", "").strip()}


def expand_records(items: list[dict], ontology, definitions, questions) -> list[dict]:
    """One generated interaction → one record per applicable scheme (Tev1 format)."""
    to6 = s13_to_s6(ontology)
    records = []
    for i, it in enumerate(items):
        kind = it["kind"]
        cls = it["class"]
        if it.get("response") and it.get("response_style") == "harmful_response":
            kind, cls = "restricted", it["response_class"]  # benign user turn, harmful assistant turn
        labels = {}
        if kind == "safe":
            labels = {"s2": "safe", "s3": "safe", "s6": "safe", "s13": "safe"}
        elif kind == "needs_caution":
            labels = {"s3": "needs_caution", "s6": "needs_caution", "s13": "needs_caution"}
        else:
            labels = {"s2": "unsafe", "s3": "unsafe", "s6": to6[cls], "s13": cls}
        row = {"id": f"cc-{i:06d}", "prompt": it["text"], "response": it.get("response") or None,
               "prompt_label": labels.get("s2", "unsafe"), "violated_categories": cls}
        for scheme, answer in labels.items():
            if answer not in ontology["schemes"][scheme]:
                continue
            qid = sorted(questions)[int(digest(f"{row['id']}:{scheme}")[:8], 16) % len(questions)]
            rec = make_record(row, scheme, qid, questions[qid], answer, definitions, ontology, "train")
            rec["id"] = f"cc-{i:06d}:{scheme}:{qid}"; rec["source"] = "constitutional_synthetic"
            rec["provenance"] = {"class": it["class"], "style": it["style"], "language": it["language"], "gen_model": it["gen_model"],
                                 "has_response": bool(it.get("response")), "response_style": it.get("response_style"),
                                 "response_class": it.get("response_class")}
            records.append(rec)
    return records


def subsample(records: list[dict], n_train: int, n_dev: int) -> list[dict]:
    """Match the safety_tev set size: group-level split, then class-balanced per scheme (like build_safety_sft)."""
    rng = rng_for("cc:split")
    groups = sorted({r["group_id"] for r in records})
    dev_groups = set(rng.sample(groups, max(1, len(groups) // 10)))
    for r in records:
        r["split"] = "dev" if r["group_id"] in dev_groups else "train"
    schemes = sorted({r["scheme"] for r in records})
    out = []
    for scheme in schemes:
        for split, n in (("train", n_train // len(schemes)), ("dev", n_dev // len(schemes))):
            cands = [r for r in records if r["scheme"] == scheme and r["split"] == split]
            out.extend(balanced_sample(cands, n, f"cc:{scheme}:{split}"))
    return out


def export(records: list[dict], out: Path, extra_manifest: dict):
    tok_path = ROOT / "data/raw/qwen-tokenizer/tokenizer.json"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    renderer = Renderer()
    if any("split" not in r for r in records):
        rng = rng_for("cc:split")
        groups = sorted({r["group_id"] for r in records})
        dev_groups = set(rng.sample(groups, max(1, len(groups) // 10)))
        for r in records:
            r["split"] = "dev" if r["group_id"] in dev_groups else "train"
    out.mkdir(parents=True, exist_ok=True); (out / "sft").mkdir(exist_ok=True); (out / "instruction").mkdir(exist_ok=True)
    with (out / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    max_tokens = 0
    for split in ("train", "dev"):
        rs = [r for r in records if r["split"] == split]
        rng_for(f"cc:order:{split}").shuffle(rs)
        with (out / "sft" / f"{split}.jsonl").open("w") as fs, (out / "instruction" / f"{split}.jsonl").open("w") as fi:
            for r in rs:
                msgs = messages(r); prompt = renderer.prompt(msgs)
                max_tokens = max(max_tokens, len(tok.encode(prompt).ids))
                fs.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                fi.write(json.dumps({"prompt": prompt, "completion": r["answer"] + EOS}, ensure_ascii=False) + "\n")
    manifest = {"seed": SEED, "n_records": len(records),
                "n_train": sum(r["split"] == "train" for r in records), "n_dev": sum(r["split"] == "dev" for r in records),
                "label_dist_train": {s: dict(Counter(r["answer_key"] for r in records if r["scheme"] == s and r["split"] == "train"))
                                     for s in ("s2", "s3", "s6", "s13")},
                "max_prompt_tokens_qwen35": max_tokens, **extra_manifest,
                "files": {str(p.relative_to(out)): digest(p.read_text()) for p in sorted(out.rglob("*.jsonl"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls-per-class", type=int, default=9, help="generation calls per restricted class (×20 msgs)")
    ap.add_argument("--stage", choices=["prompts", "responses", "export"], default="prompts")
    ap.add_argument("--launch", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=4000, help="export: records after balanced subsampling (0 = keep all)")
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "training/data/cc-mini")
    args = ap.parse_args()
    ontology = yaml.safe_load((ROOT / "data/label_map.yaml").read_text())
    definitions = yaml.safe_load((ROOT / "prompts/label_definitions.yaml").read_text())
    constitution = (ROOT / "prompts/constitution.md").read_text()
    raw_path = args.out / "generated_raw.jsonl"
    resp_path = args.out / "generated_with_responses.jsonl"

    def read_jsonl(path):
        with path.open() as f:
            return [json.loads(l) for l in f if l.strip()]

    def make_client():
        from data.bt_auth import load_braintrust_key
        import anthropic
        return anthropic.Anthropic(base_url=GATEWAY_BASE_URL, api_key=load_braintrust_key(),
                                   default_headers={"x-bt-use-cache": "never"}, max_retries=3, timeout=300.0)

    if args.stage == "responses":
        if not raw_path.exists():
            raise SystemExit(f"Missing {raw_path}; run --stage prompts first")
        items = assign_responses(read_jsonl(raw_path))
        todo = [it for it in items if it["needs_response"]]
        batches = [todo[i:i + PER_CALL] for i in range(0, len(todo), PER_CALL)]
        print(json.dumps({"prompts": len(items), "responses_to_generate": len(todo), "calls": len(batches),
                          "by_style": dict(Counter(it["response_style"] for it in todo)),
                          "est_cost_usd": round((len(batches) * 1200) / 1e6 * 2 + (len(todo) * 120) / 1e6 * 10 + 1.0, 2)}, indent=1))
        if args.dry_run or not args.launch:
            print("Dry run only. Add --launch to call the gateway."); return
        client = make_client(); got = 0; refused = 0; errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(call_responses, client, constitution, b, definitions): b for b in batches}
            for n, fut in enumerate(as_completed(futs), 1):
                b = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:160]}"); continue
                if not res:
                    refused += 1
                for j, text in res.items():
                    b[j]["response"] = text; got += 1
                if n % 10 == 0:
                    print(f"{n}/{len(batches)} calls, {got} responses", flush=True)
        for it in items:  # prompts that were meant to get a reply but did not: keep as one-turn
            if it["needs_response"] and not it.get("response"):
                it["response_style"] = None; it["response_class"] = None
        with resp_path.open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(json.dumps({"responses_written": got, "calls_refused": refused, "calls_errored": len(errors), "errors": errors[:10]}, indent=1))
        print("Next: .venv/bin/python -m training.generate_constitutional --stage export"); return

    if args.stage == "export":
        src = resp_path if resp_path.exists() else raw_path
        if not src.exists():
            raise SystemExit("Nothing to export; run --stage prompts (and responses) first")
        items = read_jsonl(src)
        block = eval_text_blocklist()
        with (ROOT / "data/raw/aegis/train.json").open() as f:
            block.update(normalized(r["prompt"]) for r in json.load(f) if r.get("prompt"))
        seen, kept, dropped = set(), [], Counter()
        for it in items:
            norm = normalized(it["text"])
            if not norm: dropped["empty"] += 1; continue
            if norm in block: dropped["eval_or_aegis_train_overlap"] += 1; continue
            if norm in seen: dropped["duplicate"] += 1; continue
            seen.add(norm); kept.append(it)
        records = expand_records(kept, ontology, definitions, QUESTIONS)
        n_expanded = len(records)
        if args.n_train:
            records = subsample(records, args.n_train, args.n_dev)
        manifest = export(records, args.out, {
            "records_before_subsample": n_expanded,
            "generation": {"model": MODEL, "gateway": GATEWAY_BASE_URL, "source_file": src.name,
                           "messages_generated": len(items), "messages_kept": len(kept),
                           "with_response": sum(1 for it in kept if it.get("response")),
                           "harmful_response_slice": sum(1 for it in kept if it.get("response") and it.get("response_style") == "harmful_response"),
                           "dropped": dict(dropped), "constitution_sha256": digest(constitution),
                           "label_definitions_sha256": digest((ROOT / "prompts/label_definitions.yaml").read_text())}})
        print(json.dumps({k: manifest[k] for k in ("n_train", "n_dev", "label_dist_train", "max_prompt_tokens_qwen35", "generation")}, indent=1))
        print(f"Next: .venv/bin/python -m training.train_loops --launch --data {args.out / 'instruction'} --name cc-repro-mini"); return

    plan = plan_calls(ontology, args.calls_per_class)
    est_in = len(plan) * 250 + 900  # constitution cached after first call
    est_out = len(plan) * PER_CALL * 60
    print(json.dumps({"calls": len(plan), "messages_target": len(plan) * PER_CALL,
                      "by_style": dict(Counter(c["style"] for c in plan)),
                      "est_cost_usd": round(est_in / 1e6 * 2 + est_out / 1e6 * 10 + 900 / 1e6 * 2.5, 2)}, indent=1))
    if args.dry_run or not args.launch:
        print("Dry run only. Add --launch to call the gateway."); return

    from data.bt_auth import load_braintrust_key
    import anthropic
    client = anthropic.Anthropic(base_url=GATEWAY_BASE_URL, api_key=load_braintrust_key(),
                                 default_headers={"x-bt-use-cache": "never"}, max_retries=3, timeout=300.0)
    items, refused, errors = [], 0, []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_gateway, client, constitution, cell, definitions): cell for cell in plan}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                got = fut.result()
            except Exception as exc:
                errors.append({"cell": futs[fut], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}); continue
            if got and got[0].get("_refused"):
                refused += 1; continue
            items.extend(got)
            if n % 20 == 0:
                print(f"{n}/{len(plan)} calls, {len(items)} messages, {time.time() - started:.0f}s", flush=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"messages_generated": len(items), "calls_refused": refused, "calls_errored": len(errors), "errors": errors[:10]}, indent=1))
    print("Next: .venv/bin/python -m training.generate_constitutional --stage responses --launch"); return



if __name__ == "__main__":
    main()
