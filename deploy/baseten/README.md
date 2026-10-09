# Baseten dedicated deployments

Two byte-identical vLLM configs (only `model_name`, `weights.source`, `--served-model-name` differ):

| Dir | Model | HF revision |
|---|---|---|
| `tev1/` | togethercomputer/Tev1-4B-experimental | `0b7becf017daa0e5eb222f8ce7483c8c8259c52f` |
| `qwen35-4b/` | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

Deploy (billable; scale-to-zero is the workspace default, so idle cost is zero):

```bash
set -a; source .env; set +a
.venv/bin/truss push deploy/baseten/tev1 --publish
.venv/bin/truss push deploy/baseten/qwen35-4b --publish
```

Record the printed model ids as `BASETEN_TEV1_MODEL_ID` / `BASETEN_QWEN35_MODEL_ID` in `.env`, then run the smoke test on 20 dev items (`adapters/test_smoke.py`, pending). Verify: single-letter output, `top_logprobs` present with ≥ 15 candidates, thinking disabled (no `<think>` in content), latency.

Notes
- L4 is enough for a 4B bf16 model; H100 is not needed. Keep the accelerator identical across both configs.
- Qwen3.5-4B ships vision weights; vLLM loads them but we send text only. If memory is tight, add a `--limit-mm-per-prompt` flag to both configs identically.
- The Tev1 weights license is pending on the HF card. Private inference is fine; do not publish self-hosted Tev1 numbers until it is posted.
