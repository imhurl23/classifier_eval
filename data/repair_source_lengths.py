"""Version corrected length metadata without changing inputs, labels or membership."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer
from data.bt_auth import ROOT
from data.build_datasets import count_tokens, interaction_text, push
from data.jsonl import read_jsonl


def repair(row, tokenizer):
    if row['metadata']['source'] != 'wildguardtest':
        return row
    row = {**row, 'metadata': dict(row['metadata'])}
    row['metadata']['token_len_qwen35'] = count_tokens(
        tokenizer, interaction_text(row['input']['prompt'], row['input']['response']))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    if args.push and not args.apply:
        parser.error('--push requires --apply')
    manifest_path = ROOT/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tokenizer = Tokenizer.from_file(str(ROOT/'data/raw/qwen-tokenizer/tokenizer.json'))
    datasets = {}; report = {}
    for name, entry in manifest['datasets'].items():
        path = ROOT/'data/processed'/f'{name}.jsonl'
        original = read_jsonl(path)
        repaired = [repair(row, tokenizer) for row in original]
        assert [(r['id'], r['input'], r['expected']) for r in original] == [(r['id'], r['input'], r['expected']) for r in repaired]
        changed = sum(a['metadata'] != b['metadata'] for a,b in zip(original,repaired))
        datasets[name] = repaired
        report[name] = {'items': len(repaired), 'metadata_changed': changed}
        if args.apply and changed:
            old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = ROOT/'data/processed/metadata-history'/old_sha
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive/path.name)
            shutil.copy2(manifest_path, archive/'run_manifest.json')
            with path.open('w') as handle:
                for row in repaired:
                    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            entry.update(count=len(repaired),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),braintrust_version=None)
            entry.pop('braintrust_name',None)
    if args.apply:
        manifest['length_metadata_repair'] = {'date':'2026-10-09','scope':'WildGuard original full prompt/response with wrapper; inputs, gold and membership preserved','datasets':report}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.push:
        push(datasets,manifest)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
