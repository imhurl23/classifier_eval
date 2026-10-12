import pandas as pd
from analysis.failure_report import outcome,metrics,paired_gains,roc


def test_failure_types_distinguish_classifier_refusal_from_observed_refusal():
    assert outcome('refusal',None,'refusal')=='classifier_refusal'
    assert outcome('ok','not_refusal','refusal')=='missed_refusal'
    assert outcome('ok','refusal','not_refusal')=='false_refusal'
    assert outcome('ok','needs_caution','unsafe')=='caution_abstention'
    assert outcome('truncated',None,'unsafe')=='truncated'
    assert outcome('ok','safe',None)=='unscored'


def test_strict_miss_denominator_and_disjoint_error_accounting():
    rows=[{'error':True,'outcome':'false_safe','source_gold':'unsafe','status':'ok','predicted_label_confidence':.99,'high_confidence_error':True},
          {'error':True,'outcome':'classifier_refusal','source_gold':'unsafe','status':'refusal','predicted_label_confidence':None,'high_confidence_error':False},
          {'error':False,'outcome':'correct','source_gold':'unsafe','status':'ok','predicted_label_confidence':.8,'high_confidence_error':False},
          {'error':True,'outcome':'false_unsafe','source_gold':'safe','status':'ok','predicted_label_confidence':.7,'high_confidence_error':False}]
    m=metrics(pd.DataFrame(rows));assert m['errors']==3 and m['false_safe']==1 and m['classifier_refusal']==1
    assert m['unsafe_miss_n']==3 and m['unsafe_miss_k']==2
    assert m['safe_false_unsafe_n']==1 and m['safe_false_unsafe_k']==1


def test_paired_finetune_gains_use_same_items_not_marginal_row_counts():
    base={'source_dataset':'test','input_unit':'conversation','target':'interaction_harm','human_gold_available':True}
    rows=[{**base,'item_id':str(i),'arm':a,'error':bad} for a,i,bad in
          [('tev1_native',0,False),('qwen35_4b_letters',0,True),('tev1_native',1,True),('qwen35_4b_letters',1,True),('tev1_native',2,False)]]
    m=paired_gains(pd.DataFrame(rows)).iloc[0]
    assert m.n==2 and m.a_wins==1 and m.a_losses==0 and m.both_wrong==1 and m.both_correct==0
    assert m.accuracy_delta_pp==50


def test_probability_roc_groups_ties_and_has_correct_direction():
    assert roc([.9,.8,.2,.1],[True,True,False,False])[2]==1
    assert roc([.1,.2,.8,.9],[True,True,False,False])[2]==0
    assert roc([.5,.5,.5,.5],[True,True,False,False])[2]==.5
