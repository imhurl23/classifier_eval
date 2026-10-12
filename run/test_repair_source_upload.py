import pytest
from run.repair_source_upload import stale_ids,event


def test_upload_repair_only_removes_invalid_derived_placeholders():
    stale={'id':'partial','metadata':{'physical_calls':0},'output':{'status':'invalid_output'}}
    assert stale_ids([stale],{})==['partial']
    assert stale_ids([stale],{'partial':{}})==[]
    with pytest.raises(ValueError):stale_ids([{**stale,'metadata':{'physical_calls':1}}],{})
    with pytest.raises(ValueError):stale_ids([{**stale,'output':{'status':'ok'}}],{})


def test_replayed_derivation_preserves_gold_and_does_not_duplicate_usage():
    row={'id':'item','expected':{'s2':'unsafe'},'metadata':{'source':'wildguardtest','label_source':'human','gold_provenance':'human'}}
    result={'job_id':'derived','arm':'granite_41_8b','target':'interaction_harm','scheme':'s2','rubric':'native_or',
            'physical_calls':0,'parent_job_ids':['prompt','response'],'label':'unsafe','probs':None,'status':'ok',
            'scores':{'source_accuracy':1},'model_version':'pinned','latency_ms':3,'usage':{'input_tokens':100}}
    replay=event(result,row)
    assert replay['input']=={'derived_from':['prompt','response']}
    assert replay['expected']=={'label':'unsafe'} and replay['scores']==result['scores']
    assert replay['metadata']['physical_calls']==0 and replay['metadata']['model_version']=='pinned'
    assert replay['metrics']=={'latency_ms':3}
    assert replay['span_id']==replay['root_span_id']
