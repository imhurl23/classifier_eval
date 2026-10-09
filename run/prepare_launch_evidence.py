"""Create incomplete, editable launch-evidence templates without running inference."""
import argparse,json
from functools import lru_cache
from pathlib import Path
from run.heldout import load_plan
from run.baseten_preflight import ALLOWLIST


def prepare(directory):
    from run import offline_review
    offline_review.options_for=lru_cache(None)(offline_review.options_for)
    offline_review.instructions_for=lru_cache(None)(offline_review.instructions_for)
    directory=Path(directory);summary,rows,jobs=load_plan(directory)
    folder=directory/'launch-evidence-templates';folder.mkdir(exist_ok=True)
    payloads={
      'context.json':{'jobs':[{'job_id':j['job_id'],'request_sha256':j['request_sha256'],
        'fits_full_input_and_output':False,'evidence':''} for j in jobs]},
      'readiness.json':{'arms':{a:{'validated':False,'model_version':'','evidence':'',
          **({'model_id':ALLOWLIST['qwen35_4b_letters' if a=='qwen35_4b_json' else a][0],
              'deployment_id':ALLOWLIST['qwen35_4b_letters' if a=='qwen35_4b_json' else a][1]}
             if a in ALLOWLIST or a=='qwen35_4b_json' else {})} for a in summary['requests_by_arm']}},
      'budget.json':{'manifest_sha256':summary['artifact_sha256']['manifest.jsonl'],'billing_verified':False,
         'total_study_planning_ceiling_usd':1000,'max_wall_minutes':None,
         'prior_spend_and_unsettled_ceiling_usd':None,'prior_spend_evidence':'',
         'shared_fees_and_contingency_usd':None,
         'arms':{a:{'rate_basis':'','evidence':'','ceiling_usd':None} for a in summary['requests_by_arm']}}}
    for name,payload in payloads.items():
        destination=folder/name
        if destination.exists():raise ValueError('Do not overwrite edited evidence: '+str(destination))
        destination.write_text(json.dumps(payload,indent=2)+'\n')
    print(folder)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    prepare(p.parse_args().plan)
