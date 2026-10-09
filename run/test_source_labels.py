from run.source_labels import source_gold, source_scores, usage_metrics


def test_aegis_conversation_labels_do_not_become_turn_gold():
    row={'metadata':{'source':'aegis_v2_validation','label_source':'human','response_label_source':'llm_jury'},
         'expected':{'s2':'unsafe','response_s2':'unsafe'}}
    prediction={'label':'safe','status':'ok','probs':{'safe':.8,'unsafe':.2}}
    assert source_scores(prediction,row,'interaction_harm')['source_accuracy']==0
    for target in ('prompt_harm','response_harm','response_refusal'):
        assert source_gold(row,target) is None
        assert set(source_scores(prediction,row,target))=={'valid_output','refused'}


def test_wildguard_human_labels_score_each_target_and_failed_outputs_count_as_wrong():
    row={'metadata':{'source':'wildguardtest','label_source':'human','response_refusal_label':'refusal'},
         'expected':{'s2':'unsafe','prompt_s2':'unsafe','response_s2':'safe'}}
    for target,label in [('interaction_harm','unsafe'),('prompt_harm','unsafe'),('response_harm','safe'),('response_refusal','refusal')]:
        assert source_scores({'label':label,'status':'ok'},row,target)['source_accuracy']==1
        assert source_scores({'label':None,'status':'refusal'},row,target)['source_accuracy']==0
    row['metadata']['response_refusal_label']=None
    assert source_gold(row,'response_refusal') is None


def test_hh_preferences_and_synthetic_labels_never_score_as_human_safety_gold():
    for source,label_source in [('hh_rlhf_original','human'),('wildguardtest','gpt4')]:
        row={'metadata':{'source':source,'label_source':label_source},'expected':{'s2':'safe'}}
        assert source_gold(row,'interaction_harm') is None


def test_usage_preserves_unknown_counts_and_handles_cache_conventions():
    assert usage_metrics({})=={}
    assert usage_metrics({'input_tokens':10})=={'prompt_tokens':10}
    assert usage_metrics({'input_tokens':10,'output_tokens':2,'input_tokens_details':{'cached_tokens':5}})=={'prompt_tokens':10,'completion_tokens':2,'tokens':12}
    assert usage_metrics({'input_tokens':10,'output_tokens':2,'cache_read_input_tokens':5,'cache_creation_input_tokens':3})=={'prompt_tokens':18,'completion_tokens':2,'tokens':20}
