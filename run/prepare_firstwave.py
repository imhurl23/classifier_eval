"""Prepare full first-wave wire requests, coverage evidence and a provisional cost inventory."""
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from adapters.policylm import make_prompt_request
from prompts.runtime import options_for,instructions_for
from run import offline_review
from run.offline_review import digest,request_for,assert_context
from run.heldout import ROOT,sha,load_plan

COMMON=('jev_choice','jev_noul','clef_choice','clef_flash_choice','luna_structured','decisions_luna','claude_structured',
        'tev1_native','qwen35_4b_letters','qwen35_4b_json','safeguard_120b')


def jobs_for(rows):
    for row in rows:
        for arm in COMMON:yield arm,row,'interaction_harm','r_long','s2'
        for arm in ('granite_41_8b','qwen3guard_8b'):
            for target in ('prompt_harm','response_harm'):
                yield arm,row,target,target+'_s2','s2'
        yield 'nemotron_guard',row,'native_pair','r_long','s2'


def prepare():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    offline_review.options_for=lru_cache(None)(offline_review.options_for)
    offline_review.instructions_for=lru_cache(None)(offline_review.instructions_for)
    source=ROOT/'data/processed/heldout-candidate/55a1f8b24835'
    rows=list(map(json.loads,(source/'sample.jsonl').read_text().splitlines()))
    paths=[p for folder in ('adapters','prompts') for p in sorted((ROOT/folder).rglob('*')) if p.suffix in {'.py','.md','.yaml'}]
    paths += [ROOT/p for p in ('data/label_map.yaml','run/heldout.py','run/model_metadata.py','run/experiment_groups.py','run/source_labels.py','run/prepare_firstwave.py','run/freeze_heldout.py','run/offline_review.py','run/matrix.py','run/pilot.py','run/baseten_preflight.py','scorers/scorers.py','analysis/heldout-analysis-plan.md')]
    paths += sorted((ROOT/'deploy/baseten').glob('*/config.yaml'))
    paths += [p for folder in ('qwen-tokenizer','gpt-oss-tokenizer') for p in sorted((ROOT/'data/raw'/folder).glob('*')) if p.is_file()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    version=digest({'files':hashes,'sample':sha(source/'sample.jsonl'),'wave':'binary_long_native_separate_v1'})[:12]
    out=ROOT/'data/processed/heldout-firstwave'/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists():raise ValueError('This plan has already started')
    (out/'sample.jsonl').write_bytes((source/'sample.jsonl').read_bytes())
    qwen=Tokenizer.from_file(str(ROOT/'data/raw/qwen-tokenizer/tokenizer.json'))
    gpt=Tokenizer.from_file(str(ROOT/'data/raw/gpt-oss-tokenizer/tokenizer.json'))
    template=AutoTokenizer.from_pretrained(str(ROOT/'data/raw/qwen-tokenizer'),local_files_only=True)
    costs=json.loads((ROOT/'data/processed/study-reference/00de7797f0a6/pricing.json').read_text())
    cover=defaultdict(lambda:{'requests':0,'max_wire_proxy_tokens_plus_output':0,'max_local_qwen_chat_tokens_plus_output':0,'local_4096_overflow_ids':[],'wire_proxy_over_4096_ids':[]})
    reserves=defaultdict(float);unknown=[];count=0;counts=Counter()
    with (out/'manifest.jsonl').open('w') as manifest,(out/'context-checks.jsonl').open('w') as context:
        for arm,row,target,rubric,scheme in jobs_for(rows):
            req=request_for(arm,row,target,rubric,scheme);assert_context(arm,row,target,req)
            job=dict(arm=arm,item_id=row['id'],target=target,rubric=rubric,scheme=scheme,replicate=0,
                     request=req,request_sha256=digest(req),policy_mode='native' if arm in {'qwen3guard_8b','nemotron_guard'} else 'byoc' if arm=='granite_41_8b' else 'common_policy')
            job['job_id']=digest({k:job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})
            manifest.write(json.dumps(job,ensure_ascii=False)+'\n');count+=1;counts[arm]+=1
            wire=json.dumps(req,ensure_ascii=False);output=req.get('max_tokens',req.get('max_output_tokens',0))
            encoder=gpt if arm=='safeguard_120b' else qwen
            proxy=len(encoder.encode(wire,add_special_tokens=False).ids)+output
            local=None
            if arm in {'tev1_native','qwen35_4b_letters','qwen35_4b_json'}:
                rendered=template.apply_chat_template(req['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                local=len(qwen.encode(rendered,add_special_tokens=False).ids)+output
            evidence={'job_id':job['job_id'],'arm':arm,'item_id':row['id'],'wire_proxy_tokens_plus_output':proxy,
                      'qwen_chat_tokens_plus_output':local,'serving_template_verified':False,
                      'method':'Actual local Qwen base template for Qwen letters/JSON; proxy for Tev. Serialized-wire tokenizer proxy for other arms.',
                      'local_4096_overflow':local>4096 if local is not None else None}
            context.write(json.dumps(evidence)+'\n');c=cover[arm];c['requests']+=1;c['max_wire_proxy_tokens_plus_output']=max(c['max_wire_proxy_tokens_plus_output'],proxy)
            if proxy>4096:c['wire_proxy_over_4096_ids'].append(row['id']+'|'+target)
            if local is not None:
                c['max_local_qwen_chat_tokens_plus_output']=max(c['max_local_qwen_chat_tokens_plus_output'],local)
                if local>4096:c['local_4096_overflow_ids'].append(row['id'])
            if arm in costs and costs[arm].get('input') is not None:
                rate=costs[arm];input_rate=max(rate['input'],rate.get('cache_write',0));reserves[arm]+=3*(len(wire.encode())*input_rate+output*rate['output'])/1e6
            elif arm=='decisions_luna':
                # Explicit hypothetical contingency; never presented as endpoint pricing.
                reserves['decisions_unverified_10_usd_per_million_input_scenario']+=3*len(wire.encode())*10/1e6
    labels,definitions=options_for('s2','prompt_harm_s2')
    with (out/'policylm-supplement.jsonl').open('w') as f:
        for row in rows:
            req=make_prompt_request(row['input']['prompt'],instructions_for('prompt_harm_s2'),definitions)
            f.write(json.dumps({'item_id':row['id'],'request':req,'request_sha256':digest(req),'human_prompt_gold_available':row['metadata']['source']=='wildguardtest'},ensure_ascii=False)+'\n')
    estimate={'status':'provisional_not_billing_verified','saved_rates_checked':costs['checked_date'],
              'known_hosted_rate_reserves_usd':dict(reserves),'known_hosted_total_usd':sum(v for k,v in reserves.items() if not k.startswith('decisions_')),
              'decisions_scenario_usd':reserves['decisions_unverified_10_usd_per_million_input_scenario'],
              'basis':'Three attempts; UTF-8 request bytes as conservative input proxy; capped output. Decisions $10/M is an unverified scenario, not a provider rate.',
              'baseten_compute':'Requires measured throughput, selected hardware rates and wall-time envelope; token counts do not price GPU deployments.',
              'unresolved':['Decisions endpoint billing','gateway fees','past billed spend and unused reserves','Baseten compute/time envelope']}
    (out/'cost-inventory.json').write_text(json.dumps(estimate,indent=2)+'\n')
    (out/'context-summary.json').write_text(json.dumps({'by_arm':dict(cover),'full_pairs_retained':len(rows),'serving_verified':False,
                                                   'length_policy':'No silent dropping or truncation; overflows require verified larger contexts or explicit separate coverage reporting.'},indent=2)+'\n')
    summary={'version':version,'stage':'heldout_firstwave','status':'prepared_not_frozen_not_launched','items':len(rows),'physical_requests':count,
             'requests_by_arm':dict(counts),'policy_mode':'Common-policy arms and native/BYOC guards analyzed separately',
             'primary_interaction_decisions':len(rows)*14,'policylm_local_user_supplement':len(rows),
             'prior_source':str(source.relative_to(ROOT)),'file_sha256':hashes,
             'artifact_sha256':{name:sha(out/name) for name in ('sample.jsonl','manifest.jsonl','context-checks.jsonl','policylm-supplement.jsonl','context-summary.json','cost-inventory.json')},
             'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'audit_status':'One-reviewer audit pending',
             'launch_gates':['Completed single-reviewer audit and finalized policy','Verified full context coverage for every scheduled request','Successful development preflight and deployment binding','Verified billing and total study envelope <= $1,000','Frozen analysis plan and evidence']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');load_plan(out)
    print(json.dumps({'plan':str(out.relative_to(ROOT)),'items':len(rows),'physical_requests':count,'known_hosted_reserve':estimate['known_hosted_total_usd'],'unverified_decisions_scenario':estimate['decisions_scenario_usd'],'context_overflows':{a:len(c['local_4096_overflow_ids']) for a,c in cover.items()}},indent=2))
if __name__=='__main__':prepare()
