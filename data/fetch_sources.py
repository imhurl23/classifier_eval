"""Download the pinned inputs used by build_datasets.py."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from data.build_datasets import ROOT, SOURCE_REVISIONS

FILES = {
    "data/raw/aegis/test.json": f"https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/{SOURCE_REVISIONS['aegis']}/test.json",
    "data/raw/aegis/validation.json": f"https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/{SOURCE_REVISIONS['aegis']}/validation.json",
    "data/raw/aegis/README.md": f"https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/{SOURCE_REVISIONS['aegis']}/README.md",
    "data/raw/xstest_prompts.csv": f"https://raw.githubusercontent.com/paul-rottger/xstest/{SOURCE_REVISIONS['xstest']}/xstest_prompts.csv",
    "data/raw/gpt-oss-tokenizer/tokenizer.json": "https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/tokenizer.json",
    "data/raw/qwen-tokenizer/tokenizer.json": f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{SOURCE_REVISIONS['qwen35_tokenizer']}/tokenizer.json",
}


GATED_HF = {
    # Requires accepting the AI2 terms on the DATASET repo and a token in ~/.cache/huggingface/token
    "data/raw/wildguard/test/wildguard_test.parquet": ("allenai/wildguardmix", "test/wildguard_test.parquet", SOURCE_REVISIONS["wildguardmix"]),
}


def fetch_gated():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    for rel, (repo, filename, revision) in GATED_HF.items():
        path = ROOT / rel
        if path.exists():
            print(f"present {rel}")
            continue
        try:
            hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                            local_dir=str(path.parents[1]))
            print(f"downloaded {rel}")
        except GatedRepoError:
            print(f"SKIPPED {rel}: accept the gate at https://huggingface.co/datasets/{repo} with the logged-in HF account, then re-run")


def main():
    fetch_gated()
    for rel, url in FILES.items():
        path = ROOT / rel
        if path.exists():
            print(f"present {rel}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        print(f"downloaded {rel} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
