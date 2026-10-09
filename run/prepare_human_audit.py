"""Upload a blinded review dataset; never copy source gold or model predictions."""
import json
from datetime import datetime,timezone
from pathlib import Path
from data.bt_auth import load_braintrust_key
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    import braintrust as bt
    directory=ROOT/'data/processed/offline-review/f0c7bbda4b65'
    receipt=directory/'private/braintrust-audit-dataset.json'
    if receipt.exists():
        print(receipt.read_text()); return
    summary=json.loads((directory/'summary.json').read_text())
    private=[json.loads(x) for x in (directory/'private/sample.jsonl').read_text().splitlines()]
    rows=[{'audit_id':r['audit_id'],'user':r['record']['input']['prompt'],'assistant':r['record']['input']['response']} for r in private]
    load_braintrust_key()
    dataset=bt.init_dataset(project='decision-interface-safety',name='human_policy_audit_blind__f0c7bbda4b65',
                           description='200 development pairs. Independent judgments only; source labels and model predictions withheld.',
                           metadata={'policy_hash':summary['policy_hash'],'version':summary['version'],'human_annotations_complete':False})
    for row in rows:
        dataset.insert(id=row['audit_id'],input=row,expected={},metadata={'policy_hash':summary['policy_hash']})
    dataset.flush()
    result={'dataset_id':dataset.id,'name':dataset.name,'items':len(rows),'uploaded_at_utc':datetime.now(timezone.utc).isoformat(),
            'reviewer_count':1,'reviewer_assignments':'pending single reviewer identity','agreement_scope':'single-reviewer judgments; inter-rater agreement unmeasured','annotations':'none; no synthetic labels',
            'blind_local_packet':str((directory/'annotate.html').relative_to(ROOT))}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': prepare()
