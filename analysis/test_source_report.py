import pandas as pd
from analysis.source_report import normalize_usage,describe,normalize,disagreement,quality,KEYS


def test_usage_cache_subsets_and_separate_anthropic_counts():
    assert normalize_usage({'input_tokens':100,'output_tokens':2,'input_tokens_details':{'cached_tokens':40}})['total_tokens']==102
    u=normalize_usage({'input_tokens':100,'output_tokens':2,'cache_read_input_tokens':20,'cache_creation_input_tokens':5})
    assert u['input_tokens']==125 and u['total_tokens']==127 and u['cached_input_tokens']==20
    u=normalize_usage({'input_characters':40,'windows':1})
    assert u['input_tokens'] is None and u['total_tokens'] is None


def test_missing_observations_and_zero_resources_are_not_invented_rates():
    missing=describe(pd.Series([None,None]),2)
    assert missing['n']==0 and missing['missing']==2 and missing['mean'] is None
    zero=describe(pd.Series([0,0,0]),3,binary=False)
    assert zero['mean_ci_low']==zero['mean_ci_high']==0
    binary=describe(pd.Series([1,1,1]),3)
    assert binary['mean_ci_low']<1 and binary['mean_ci_high']<=1


def test_derived_views_do_not_duplicate_physical_resources_or_assume_free_gpu():
    row={'id':'x','input':{'unit':'conversation'},'metadata':{'source':'aegis_v2_test','label_source':'human'},'expected':{'s2':'unsafe'}}
    r={'job_id':'j','item_id':'x','arm':'granite_41_8b','target':'interaction_harm','scheme':'s2','rubric':'native_or',
       'model_version':'pinned','physical_calls':0,'parent_job_ids':['a','b'],'status':'ok','label':'unsafe',
       'latency_ms':200,'usage':{'prompt_tokens':100},'scores':{'source_accuracy':1}}
    n=normalize(r,row,'baseten',{}, {},'plan')
    assert n['latency_s'] is None and n['parent_inference_time_s']==.2
    assert n['input_tokens'] is None and n['token_cost_usd'] is None and n['gpu_allocated_proxy_usd'] is None
    assert n['human_gold_available'] and n['physical_calls']==0


def test_disagreement_uses_matched_jointly_valid_denominator():
    common={'source_dataset':'wildguardtest','input_unit':'conversation','target':'interaction_harm','human_gold_available':True,'policy_mode':'common_policy'}
    rows=[{**common,'arm':a,'item_id':str(i),'status':status,'label':label} for a,i,status,label in
          [('a',0,'ok','safe'),('a',1,'refusal',None),('b',0,'ok','unsafe'),('b',1,'ok','safe')]]
    result=disagreement(pd.DataFrame(rows)).iloc[0]
    assert result.paired_n==2 and result.joint_valid_n==1 and result.disagreement==1


def test_strict_class_recall_counts_failed_unsafe_predictions_as_misses():
    common={k:'test' for k in KEYS};common.update(human_gold_available=True,source_gold='unsafe')
    rows=[{**common,'status':'ok','label':'unsafe','score__source_accuracy':1,'score__unsafe_recall':1},
          {**common,'status':'invalid_output','label':None,'score__source_accuracy':0,'score__unsafe_recall':None}]
    frame=pd.DataFrame(rows);q=quality(frame).iloc[0]
    assert q.unsafe_recall==.5 and q.unsafe_recall_n==2
    assert frame['score__unsafe_recall'].mean()==1 # exported recorded scorer uses another denominator


def test_refusal_false_positive_diagnostic_does_not_reuse_safe_refusal_harm_scorer_name():
    common={k:'test' for k in KEYS};common.update(human_gold_available=True,source_gold='not_refusal')
    rows=[{**common,'status':'ok','label':'refusal','score__source_accuracy':0},
          {**common,'status':'ok','label':'not_refusal','score__source_accuracy':1}]
    q=quality(pd.DataFrame(rows)).iloc[0]
    assert q.refusal_false_positive_rate==.5 and q.refusal_false_positive_rate_n==2
    assert 'safe_refusal_fp' not in q.index
