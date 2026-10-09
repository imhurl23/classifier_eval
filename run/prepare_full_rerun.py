"""Prepare full-dialogue replacement stages; no inference or fabricated gold."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from run.full_context import request
from run.offline_review import digest
from data.full_conversations import ROOT
HOSTED=('jev_choice','jev_noul','clef_choice','clef_flash_choice','luna_structured','decisions_luna','claude_structured')


def jobs(rows,stage):
    if stage=='full_context_smoke':
        by_length=sorted(rows,key=lambda r:len(r['input']['text']));rows=[by_length[int(i*(len(rows)-1)/7)] for i in range(8)]
        for row in rows:
            for arm in HOSTED:
                for target,rubric in [('interaction_harm','r_long'),('response_harm','response_harm_s2')]:
                    yield arm,row,target,rubric,'s2',0
    elif stage=='full_context_reference':
        for row in rows:
            for arm in HOSTED:
                for scheme in ('s2',) if arm=='jev_noul' else ('s2','s3','s6','s13'):
                    yield arm,row,'interaction_harm','r_long',scheme,0
    elif stage=='full_context_repeatability':
        rows=sorted(rows,key=lambda r:r['id'])[::5][:40]
        for row in rows:
            for arm in HOSTED:
                for scheme in ('s2',) if arm=='jev_noul' else ('s2','s3','s6','s13'):
                    for replicate in (1,2):yield arm,row,'interaction_harm','r_long',scheme,replicate
    elif stage=='full_context_final_turns':
        for row in rows:
            for arm in HOSTED:
                if arm=='jev_noul':continue
                for target in ('prompt_harm','response_harm','response_refusal'):
                    yield arm,row,target,target+'_s2','s2',0
    else:raise ValueError(stage)


def prepare(source,known_rate_stages=False):
    source=Path(source).resolve();all_rows=list(map(json.loads,(source/'development.jsonl').read_text().splitlines()))
    rates=json.loads((ROOT/'data/processed/study-reference/00de7797f0a6/pricing.json').read_text())
    paths=[ROOT/'run/full_context.py',ROOT/'run/prepare_full_rerun.py',ROOT/'run/full_development.py',ROOT/'run/model_metadata.py',ROOT/'run/experiment_groups.py',ROOT/'prompts/full_conversation_policy.md']
    paths += [ROOT/p for p in ('run/heldout.py','run/pilot.py','run/matrix.py','run/offline_review.py','data/full_conversations.py','data/label_map.yaml')]
    paths += [p for folder in ('adapters','prompts') for p in (ROOT/folder).rglob('*') if p.suffix in {'.py','.yaml','.md'}]
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    summaries=[]
    stages=('full_context_reference','full_context_repeatability','full_context_final_turns') if known_rate_stages else ('full_context_smoke','full_context_reference','full_context_repeatability','full_context_final_turns')
    for base_stage in stages:
        stage=base_stage+'_known_rates' if known_rate_stages else base_stage
        selected=list(jobs(all_rows,base_stage))
        if known_rate_stages:
            omitted={'decisions_luna','claude_structured'} if base_stage=='full_context_final_turns' else {'decisions_luna'}
            selected=[j for j in selected if j[0] not in omitted]
        rows={row['id']:row for _,row,*_ in selected}
        version=digest({'stage':stage,'files':files,'source':hashlib.sha256((source/'development.jsonl').read_bytes()).hexdigest()})[:12]
        out=ROOT/'data/processed/full-context-rerun'/version;out.mkdir(parents=True,exist_ok=True)
        if (out/'execution.json').exists():raise ValueError('Do not rewrite an executed plan')
        (out/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows.values()))
        for name in files:
            snapshot=out/'frozen-code'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes((ROOT/name).read_bytes())
        (out/'pricing.json').write_text(json.dumps(rates,indent=2)+'\n')
        counts=Counter();reserve=0;known=defaultdict(float);decision_scenario=0
        with (out/'manifest.jsonl').open('w') as f:
            for arm,row,target,rubric,scheme,replicate in selected:
                wire=request(arm,row,target,rubric,scheme)
                job={'arm':arm,'item_id':row['id'],'target':target,'rubric':rubric,'scheme':scheme,'replicate':replicate,
                     'request':wire,'request_sha256':digest(wire)}
                job['job_id']=digest({k:job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})
                f.write(json.dumps(job,ensure_ascii=False)+'\n');counts[arm]+=1
                nbytes=len(json.dumps(wire).encode());output=wire.get('max_tokens',wire.get('max_output_tokens',0));reserve+=3*(nbytes+output)*10/1e6
                if arm=='decisions_luna':decision_scenario+=3*(nbytes+output)*10/1e6
                else:
                    rate=rates[arm];known[arm]+=3*(nbytes*max(rate['input'],rate.get('cache_write',0))+output*rate['output'])/1e6
        summary={'version':version,'stage':stage,'context_policy':'original_full_conversation_v2','status':'prepared_not_executed','items':len(rows),'requests':len(selected),
            'source':str(source.relative_to(ROOT)),'file_sha256':files,'sample_sha256':hashlib.sha256((out/'sample.jsonl').read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256((out/'manifest.jsonl').read_bytes()).hexdigest(),'requests_by_arm':dict(counts),
            'uniform_three_attempt_planning_reserve_usd':reserve,'known_rate_three_attempt_reserve_usd':sum(known.values()),
            'allocated_planning_reserve_usd':sum(known.values()) if known_rate_stages else reserve,
            'rate_basis':'Saved provider rates, three attempts, UTF-8 request bytes as conservative input proxy; output caps. Other arms remain pending, not removed from study.' if known_rate_stages else 'Uniform $10/M input and output contingency; three attempts; not established Decisions endpoint billing.',
            'decisions_unverified_10_usd_per_million_scenario_usd':decision_scenario,
            'billing_note':'Planning reserves, not confirmed endpoint billing. GPU compute and provider fees additional; historical billed charges unresolved.',
            'gold':'No Aegis or HH preference labels transferred. Human full-dialogue audit pending. Log validity and decisions; accuracy awaits actual human labels.'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append({'plan':str(out.relative_to(ROOT)),**{k:summary[k] for k in ('stage','requests','items','uniform_three_attempt_planning_reserve_usd','known_rate_three_attempt_reserve_usd','decisions_unverified_10_usd_per_million_scenario_usd')}})
    print(json.dumps(summaries,indent=2));return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--known-rate-stages',action='store_true');a=p.parse_args();prepare(a.source,a.known_rate_stages)
