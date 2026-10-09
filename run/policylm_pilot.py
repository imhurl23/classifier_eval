"""Supplemental PolicyLM User-only diagnostic; run with .venv-policylm/bin/python."""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from adapters.policylm import PolicyLMPromptAdapter, make_prompt_request, REVISION, MODEL_DIR
from data.bt_auth import load_braintrust_key
from prompts.runtime import instructions_for, options_for
from scorers.scorers import score_item

ROOT=Path(__file__).resolve().parents[1]


def run(include_dev=False):
    reserved=[json.loads(l) for l in (ROOT/'data/processed/dis-pilot.jsonl').read_text().splitlines()]
    if len(reserved)!=8: raise ValueError('Expected eight reserved pilot pairs')
    rows=[('dis-pilot',r) for r in reserved]
    reference=ROOT/'data/processed/study-reference/00de7797f0a6'
    if include_dev:
        sample=[json.loads(l) for l in (reference/'sample.jsonl').read_text().splitlines()]
        expected_sha=json.loads((reference/'summary.json').read_text())['sample_sha256']
        if hashlib.sha256((reference/'sample.jsonl').read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('Frozen dev reference sample changed')
        if len(sample)!=200 or any(r['metadata']['split']!='validation' for r in sample):
            raise ValueError('Expected 200 frozen dev pairs')
        rows += [('development_reference',r) for r in sample]
    labels,definitions=options_for('s2','prompt_harm_s2')
    instructions=instructions_for('prompt_harm_s2')
    adapter=PolicyLMPromptAdapter(definitions)
    request=make_prompt_request('example',instructions,definitions)
    policy_report=asdict(adapter.native_model.check_policy(request['policy']))
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'data/processed/policylm'/timestamp
    out.mkdir(parents=True)
    import braintrust as bt
    load_braintrust_key()
    experiment=init_experiments(project='decision-interface-safety',experiment=f'policylm_prompt_pilot__{timestamp}',
                       metadata={**experiment_model_metadata(['policylm_17b_prompt']),'purpose':'supplemental_prompt_only_reference','preregistered':False,
                                 'hf_revision':REVISION,'model_info':adapter.native_model.info,
                                 'policy_report':policy_report,'input_context':'user_only',
                                 'training_overlap_caveat':'PolyGuardMix includes WildGuardMix; model card used WildGuardTest/XSTest during development',
                                 'primary_cutoff':.335,'clean':False,'policy_mapping':request['policy_mapping']})
    state={'status':'running','planned_calls':len(rows),'completed_calls':0,'counts':{},
           'experiment_url':experiment.summarize().experiment_url,'experiment_urls':experiment.experiment_urls,'model_info':adapter.native_model.info,
           'policy_report':policy_report,'hf_revision':REVISION,'source_model_card':'https://huggingface.co/musubilabs/policylm-1.7b',
           'cloud_inference_cost_usd':0,'scoring':'Human User-harm gold on dis-pilot only; 200 Aegis dev User-only judgments have no independent prompt gold',
           'started_at_utc':datetime.now(timezone.utc).isoformat()}
    def checkpoint():
        (out/'summary.tmp').write_text(json.dumps(state,indent=2)+'\n')
        (out/'summary.tmp').replace(out/'summary.json')
    checkpoint();start=time.monotonic()
    for dataset,row in rows:
        prompt=row['input']['prompt']
        with experiment.start_span(name='policylm_17b_prompt__s2',type='llm',set_current=False,
                                   input={'prompt':prompt,'input_context':'user_only'},metadata={**model_metadata('policylm_17b_prompt'),'target':'prompt_harm','scheme':'s2','rubric':'custom_policy'}) as span:
            decision=adapter.classify_prompt(prompt,instructions)
            decision.validate(labels)
            # Aegis's conversation vote is not human User-turn gold.
            gold=row['expected'].get('prompt_s2') if dataset=='dis-pilot' else None
            scores=({k:v for k,v in score_item({'status':decision.status,'label':decision.label,'probs':None},
                       {'prompt_s2':gold},row['metadata'],'s2','prompt_harm').items() if v is not None}
                    if gold else {'valid_output':float(decision.status=='ok')})
            response=decision.raw.get('response',{})
            result={**model_metadata('policylm_17b_prompt'),'source_dataset':dataset,'item_id':row['id'],'input_context':'user_only',
                    'target':'prompt_harm','human_prompt_gold':gold,'scores':scores,**asdict(decision)}
            with (out/'results.jsonl').open('a') as handle:handle.write(json.dumps(result)+'\n')
            key=f'{dataset}|{decision.status}';state['counts'][key]=state['counts'].get(key,0)+1
            state['completed_calls']+=1;checkpoint()
            span.log(output={'label':decision.label,'status':decision.status,'violation_score':response.get('score')},
                     expected={'label':gold} if gold else None,scores=scores,
                     metadata=result,metrics={'latency_ms':decision.latency_ms})
        if decision.status != 'ok':
            state['status']='partial';state['error']='Stopped on a native inference failure; inspect logged request';break
        if state['completed_calls']%20==0:print(json.dumps({'completed':state['completed_calls'],'planned':len(rows)}),flush=True)
    experiment.flush()
    if state['completed_calls']==len(rows):state['status']='completed'
    state['elapsed_seconds']=time.monotonic()-start
    checkpoint();print(json.dumps({'output':str(out),**state},indent=2),flush=True)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-reference',action='store_true',help='Also classify User turns of the frozen 200 dev pairs; no User-harm accuracy scores')
    run(parser.parse_args().dev_reference)
