import pytest
from run import source_supplement as supplement
from run.test_source_full import sample
from run.heldout import derive_native
from run.source_labels import source_scores


def test_supplement_coverage_has_explicit_native_and_prompt_only_contracts():
    row=sample();jobs=list(supplement.supplement_jobs(row))
    assert len(jobs)==22
    assert {j['arm'] for j in jobs}==set(supplement.COMMON+supplement.NATIVE+(supplement.POLICY,))
    qwen=[j for j in jobs if j['arm']=='qwen3guard_8b']
    assert len(qwen)==2 and all(j['scheme']=='s3' for j in qwen)
    policy=next(j for j in jobs if j['arm']==supplement.POLICY)
    assert policy['request']['message']==row['input']['prompt'] and policy['target']=='prompt_harm'
    aegis=sample('aegis_v2_test','prompt_only')
    assert {j['arm'] for j in supplement.supplement_jobs(aegis)}==set(supplement.COMMON)


def test_native_turn_gold_does_not_transfer_onto_aegis_but_interaction_is_scored():
    row=sample('aegis_v2_test')
    results=[{'job_id':t,'arm':'qwen3guard_8b','item_id':row['id'],'target':t,'label':label,'status':'ok','latency_ms':1}
             for t,label in [('prompt_harm','unsafe'),('response_harm','safe')]]
    for r in results:assert 'source_accuracy' not in source_scores(r,row,r['target'],'s3')
    derived=derive_native(results,{row['id']:row})[0]
    assert derived['scores']['source_accuracy']==1 and derived['physical_calls']==0


def test_nemotron_derived_turns_and_interaction_have_distinct_ids_and_shared_provenance():
    from run.source_native import native_results
    row=sample()
    parent={'job_id':'parent','arm':'nemotron_guard','item_id':row['id'],'target':'native_pair',
            'model_version':'hf:pinned','latency_ms':5,
            'batch_results':{t:{'label':label,'status':'ok','latency_ms':5,'usage':{'input_tokens':100}}
                             for t,label in [('prompt_harm','unsafe'),('response_harm','safe')]}}
    derived=native_results([parent],{row['id']:row})
    assert len(derived)==len({r['job_id'] for r in derived})==3
    assert {r['target'] for r in derived}=={'prompt_harm','response_harm','interaction_harm'}
    assert all(r['model_version']=='hf:pinned' and r['physical_calls']==0 and 'usage' not in r for r in derived)
    assert all('source_accuracy' in r['scores'] for r in derived)
