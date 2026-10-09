"""Mini safety fine-tune (optional `safety_tev` arm) with Baseten Loops.

Tev1 recipe: LoRA rank 8, lr 5e-5, 1 epoch, batch 8, cosine schedule with 3% warmup,
grad clip 1.0, completion-only loss (prompt positions get weight 0), seq len 2048.
Data: training/data/safety-mini/instruction/{train,dev}.jsonl from build_safety_sft.py
(prompt already rendered with the pinned non-thinking chat template; completion = letter + EOS).

Dry run (no GPUs, no cost):   .venv/bin/python -m training.train_loops --dry-run
Launch (billable GPUs):       .venv/bin/python -m training.train_loops --launch
Loads BASETEN_API_KEY from the environment (source .env first).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/safety-mini/instruction"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # pinned in adapters/tev1.py; Loops serves its own copy
RECIPE = dict(lora_rank=8, lora_alpha=16, learning_rate=5e-5, epochs=1, batch_size=8,
              warmup_ratio=0.03, max_grad_norm=1.0, weight_decay=0.0, max_seq_len=2048,
              seed=42, loss="cross_entropy", completion_only=True)


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def encode_example(tokenizer, prompt: str, completion: str, max_len: int):
    """Tinker/Loops convention: input = tokens[:-1], target = tokens[1:], weights mask prompt."""
    p = tokenizer.encode(prompt, add_special_tokens=False)
    a = tokenizer.encode(completion, add_special_tokens=False)
    tokens = p + a
    if len(tokens) > max_len:
        return None
    inputs = tokens[:-1]
    targets = tokens[1:]
    weights = [0.0] * (len(p) - 1) + [1.0] * len(a)
    assert len(inputs) == len(targets) == len(weights)
    return inputs, targets, weights


def to_datum(encoded):
    from baseten.loops import Datum, ModelInput, TensorData
    inputs, targets, weights = encoded
    return Datum(model_input=ModelInput.from_ints(inputs),
                 loss_fn_inputs={"target_tokens": TensorData(data=targets, dtype="int64", shape=[len(targets)]),
                                 "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)])})


def lr_at(step: int, total: int, peak: float, warmup_ratio: float) -> float:
    warm = max(1, int(round(total * warmup_ratio)))
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))  # cosine to 0, 0.5 cycles


def eval_dev(sampler, tokenizer, rows: list[dict], n: int = 100) -> dict:
    from baseten.loops import ModelInput, SamplingParams
    rng = random.Random(RECIPE["seed"])
    sample = rng.sample(rows, min(n, len(rows)))
    correct = valid = 0
    for r in sample:
        ids = tokenizer.encode(r["prompt"], add_special_tokens=False)
        out = sampler.sample(prompt=ModelInput.from_ints(ids), num_samples=1,
                             sampling_params=SamplingParams(max_tokens=8, temperature=0))
        text = tokenizer.decode(out.sequences[0].tokens, skip_special_tokens=True).strip()
        gold = r["completion"].replace("<|im_end|>", "").strip()
        valid += len(text) == 1 and text.isalpha()
        correct += text == gold
    return {"n": len(sample), "accuracy": correct / len(sample), "valid_rate": valid / len(sample)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--launch", action="store_true", help="Create billable trainer/sampler and train")
    ap.add_argument("--dry-run", action="store_true", help="Tokenize locally and print plan; no GPUs")
    ap.add_argument("--name", default="safety-tev-mini")
    ap.add_argument("--max-steps", type=int, default=None, help="Cap steps (smoke test)")
    args = ap.parse_args()
    train_rows, dev_rows = load_rows(args.data / "train.jsonl"), load_rows(args.data / "dev.jsonl")
    steps = math.ceil(len(train_rows) / RECIPE["batch_size"]) * RECIPE["epochs"]
    if args.max_steps:
        steps = min(steps, args.max_steps)
    plan = {"base_model": BASE_MODEL, "n_train": len(train_rows), "n_dev": len(dev_rows),
            "steps": steps, **RECIPE}
    print(json.dumps(plan, indent=1))

    if args.dry_run or not args.launch:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ROOT / "data/raw/qwen-tokenizer/tokenizer.json"))
        lens, dropped, ans_lens = [], 0, set()
        for r in train_rows:
            p = tok.encode(r["prompt"], add_special_tokens=False).ids
            a = tok.encode(r["completion"], add_special_tokens=False).ids
            ans_lens.add(len(a))
            if len(p) + len(a) > RECIPE["max_seq_len"]:
                dropped += 1
            lens.append(len(p) + len(a))
        print(json.dumps({"max_tokens": max(lens), "mean_tokens": sum(lens) / len(lens),
                          "over_limit": dropped, "completion_token_lengths": sorted(ans_lens),
                          "lr_schedule_sample": [round(lr_at(s, steps, RECIPE["learning_rate"], RECIPE["warmup_ratio"]), 7)
                                                 for s in (0, 5, 15, steps // 2, steps - 1)]}, indent=1))
        if not args.launch:
            print("Dry run only. Add --launch to start billable training.")
        return

    from baseten.loops import AdamParams, ServiceClient
    service = ServiceClient()  # BASETEN_API_KEY
    trainer = service.create_lora_training_client(
        base_model=BASE_MODEL, rank=RECIPE["lora_rank"], seed=RECIPE["seed"],
        max_seq_len=RECIPE["max_seq_len"], ready_timeout=1800, sampler_timeout=600, name=args.name)
    print("run_id", trainer.run_id)
    tokenizer = trainer.get_tokenizer()
    encoded = [e for e in (encode_example(tokenizer, r["prompt"], r["completion"], RECIPE["max_seq_len"])
                           for r in train_rows) if e is not None]
    rng = random.Random(RECIPE["seed"])
    rng.shuffle(encoded)
    log = []
    started = time.time()
    for step in range(steps):
        batch = [to_datum(e) for e in encoded[step * RECIPE["batch_size"]:(step + 1) * RECIPE["batch_size"]]]
        if not batch:
            break
        fb = trainer.forward_backward(data=batch).result(timeout=900)
        lr = lr_at(step, steps, RECIPE["learning_rate"], RECIPE["warmup_ratio"])
        trainer.optim_step(AdamParams(learning_rate=lr, weight_decay=RECIPE["weight_decay"],
                                      grad_clip_norm=RECIPE["max_grad_norm"])).result(timeout=900)
        loss = getattr(fb, "loss", None) or (fb.metrics.get("loss") if hasattr(fb, "metrics") else None)
        log.append({"step": step, "lr": lr, "loss": loss})
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step}/{steps} lr={lr:.2e} loss={loss} elapsed={time.time() - started:.0f}s", flush=True)
    state = trainer.save_state(name="final").result(timeout=900)
    weights = trainer.save_weights_for_sampler(name="final").result(timeout=900)
    sampler = trainer.create_sampling_client(model_path=weights.path)
    dev = eval_dev(sampler, tokenizer, dev_rows)
    out = {"run_id": trainer.run_id, "state_path": state.path, "sampler_weights_path": weights.path,
           "dev_eval": dev, "plan": plan, "log": log,
           "data_manifest": json.loads((args.data.parent / "manifest.json").read_text())}
    (ROOT / "training/data" / f"{args.name}-run.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"run_id": trainer.run_id, "dev_eval": dev, "weights": weights.path}, indent=1))
    print("Deploy with: baseten loops checkpoint deploy --run-id", trainer.run_id, "--checkpoint final")
    trainer.close()


if __name__ == "__main__":
    main()
