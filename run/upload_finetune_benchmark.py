"""Upload completed custom-checkpoint results; never makes model calls."""
from collections import defaultdict
import json
from urllib.parse import quote

from data.bt_auth import load_braintrust_key
from run.finetune_benchmark import OUT, MODELS, records, verify
from run.heldout import atomic, sha
from run.offline_review import digest
from run.source_labels import usage_metrics


def main():
    import braintrust as bt
    load_braintrust_key()
    verify()
    for arm in MODELS:
        folder = OUT / arm
        receipt = json.loads((folder / 'verification.json').read_text())
        assert receipt['complete'] and receipt['results_sha256'] == sha(folder / 'results.jsonl')
        uploaded = folder / 'braintrust-upload.json'
        if uploaded.exists() and json.loads(uploaded.read_text())['results_sha256'] == receipt['results_sha256']:
            continue
        jobs = {j['job_id']: j for j in records(folder / 'manifest.jsonl')}
        groups = defaultdict(list)
        results = list(records(folder / 'results.jsonl'))
        assert len(results) == len(jobs) == receipt['completed']
        assert {r['job_id'] for r in results} == set(jobs)
        for row in results:
            job = jobs[row['job_id']]
            assert row['model_version'] == job['checkpoint']
            assert row['source_gold'] == job['source_gold']
            assert row['raw']['request_sha256'] == digest(job['request'])
            assert row['label'] in job['option_order'] or row['status'] != 'ok'
            groups[(row['target'], row['rubric'])].append(row)
        urls = {}
        for (target, rubric), rows in groups.items():
            name = f'custom_finetune_full_20261019__{arm}__{target}__s2__{rubric}'
            experiment = bt.init(project='decision-interface-safety', experiment=name,
                update=True, set_current=False, metadata={
                    'arm': arm, 'model': f'Qwen3.5-4B custom LoRA ({arm})',
                    'model_version': rows[0]['model_version'], 'target': target,
                    'scheme': 's2', 'rubric': rubric, 'evaluation_role': 'existing_custom_checkpoint_transfer',
                    'policy_mode': 'source_aligned' if rubric == 'aligned_long' else 'original_xstest',
                    'preregistered': False, 'training_and_validation_overlap_excluded': True,
                    'provider': 'baseten_loops', 'study_status': 'completed'})
            for r in rows:
                j = jobs[r['job_id']]
                experiment.log(id=r['job_id'], input=j['request'],
                    output={'label': r['label'], 'status': r['status'], 'text': r['raw']['output']},
                    expected={'label': r['source_gold']}, scores=r['scores'],
                    metadata={k: r[k] for k in ('arm', 'item_id', 'source', 'unit', 'target',
                                               'scheme', 'rubric', 'policy_version', 'model_version')},
                    metrics={'latency_ms': r['latency_ms'], **usage_metrics(r['usage'])})
            experiment.flush()
            urls[name] = 'https://www.braintrust.dev/app/eval_engineering/p/decision-interface-safety/experiments/' + quote(name)
        atomic(uploaded, {'completed': receipt['completed'], 'results_sha256': receipt['results_sha256'],
                          'experiment_urls': urls})
        print(json.dumps({'arm': arm, 'uploaded': receipt['completed'], 'experiments': len(urls)}), flush=True)


if __name__ == '__main__':
    main()
