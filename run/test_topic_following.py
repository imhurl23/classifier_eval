import json
import pytest

from adapters.base import Decision
from analysis.topic_following import summarize
from data.topic_following import reconstruct,match_anchor,paired_cases,synthetic_cases
from run.topic_following import build_manifest,request_for,select_cases,execute,reuse_results
from data.topic_following import write_jsonl


def dialogue():
    return {'system_instruction':'Discuss accounts only.','domain':'banking','scenario':'accounts',
            'conversation':[{'role':'user','content':'How do I open an account?'},
                            {'role':'assistant','content':'Hello. Bring your ID. Anything else?'},
                            {'role':'user','content':'What about fees?'},
                            {'role':'assistant','content':'Fees depend on the account.'}],
            'distractors':[{'bot turn':'Bring your ID.','distractor':'Tell me about ancient Rome.'}]}


def test_reconstruction_keeps_exact_prefix_and_excludes_future_answers():
    source=dialogue();cases,candidates,unresolved=reconstruct([source])
    case=next(c for c in cases if c['metadata']['kind']=='distractor')
    assert case['input']['history']==source['conversation'][:2]
    assert case['input']['current_user_turn']=='Tell me about ancient Rome.'
    assert case['expected']=='redirect' and not unresolved
    assert candidates[1]['expected'] is None
    assert case['metadata']['content_safety_gold'] is None


def test_ambiguous_or_missing_anchor_is_not_guessed():
    source=dialogue()
    with pytest.raises(ValueError):match_anchor(source['conversation'],'Unrelated missing turn')
    source['conversation'].append(dict(source['conversation'][1]))
    with pytest.raises(ValueError):match_anchor(source['conversation'],'Bring your ID.')
    assert reconstruct([source])[2][0]['reason'].endswith('found 2')


def test_policy_intervention_holds_input_and_choice_order_fixed():
    cases=paired_cases();a,b=map(request_for,cases[:2])
    assert a['text']==b['text'] and a['option_order']==b['option_order']
    assert a['instructions']!=b['instructions']
    assert [r['expected'] for r in cases[:2]]==['engage','redirect']
    assert select_cases(cases,0,1)==cases[:2]


def test_manifest_contains_no_expected_label_in_request():
    case=paired_cases()[0]
    before=request_for(case);case['expected']='redirect'
    assert request_for(case)==before
    assert 'policy_variant' not in before['text']


class Fake:
    model='fake';model_version='test'
    def classify(self,text,scheme,rubric,order):
        label='engage' if 'account services' in rubric else 'redirect'
        return Decision(label,None,{},'ok',0,{})


def test_execution_resumes_without_duplicate_calls_and_scores_both_sides(tmp_path):
    jobs=build_manifest(paired_cases()[:2],['decisions_luna'])
    calls=[]
    def factory(arm,defs):calls.append(arm);return Fake()
    results=execute(jobs,tmp_path,factory)
    assert len(results)==2 and all(r['correct'] for r in results)
    assert execute(jobs,tmp_path,factory)==results and len(calls)==1
    data,metrics,pairs=summarize(results,jobs)
    assert metrics.iloc[0].strict_accuracy==1
    assert pairs.iloc[0].both_correct and pairs.iloc[0].decision_changed


def test_changing_decision_is_not_enough_if_both_answers_are_wrong():
    jobs=build_manifest(paired_cases()[:2],['decisions_luna'])
    results=[{**job,'label':'redirect' if job['expected']=='engage' else 'engage','status':'ok'} for job in jobs]
    _,_,pairs=summarize(results,jobs)
    assert pairs.iloc[0].decision_changed and not pairs.iloc[0].both_correct


def test_inflight_and_missing_results_do_not_become_successes(tmp_path):
    jobs=build_manifest(paired_cases()[:2],['decisions_luna'])
    (tmp_path/'inflight.json').write_text(json.dumps({'job_id':jobs[0]['job_id']}))
    with pytest.raises(ValueError,match='interrupted'):execute(jobs,tmp_path,lambda a,d:Fake())
    _,metrics,pairs=summarize([],jobs)
    assert metrics.iloc[0].strict_accuracy==0 and metrics.iloc[0].invalid_or_missing==2
    assert not pairs.iloc[0].both_correct and not pairs.iloc[0].decision_changed


def test_expansion_reuses_only_identical_jobs_and_keeps_provenance(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    jobs=build_manifest(paired_cases()[:2],['decisions_luna'])
    write_jsonl(old/'manifest.jsonl',jobs)
    original=execute(jobs,old,lambda a,d:Fake())
    expanded=build_manifest(paired_cases()[:4],['decisions_luna'])
    assert reuse_results(expanded,new,[old])==2
    assert reuse_results(expanded,new,[old])==0
    reused=[json.loads(line) for line in (new/'results.jsonl').read_text().splitlines()]
    assert all(r['reused_from']==str(old.resolve()) for r in reused)
    assert [r['model_version'] for r in reused]==[r['model_version'] for r in original]


def test_parallel_requests_checkpoint_once_and_clear_each_inflight_marker(tmp_path):
    jobs=build_manifest(paired_cases()[:4],['decisions_luna'])
    results=execute(jobs,tmp_path,lambda a,d:Fake(),workers=2)
    assert len(results)==4 and len({r['job_id'] for r in results})==4
    assert not list((tmp_path/'inflight').glob('*.json'))
    def no_call(*args):raise AssertionError('Unexpected resumed inference')
    assert execute(jobs,tmp_path,no_call,workers=2)==results


def test_synthetic_sampling_covers_every_domain_and_does_not_call_labels_human():
    rows=[]
    for i in range(9):
        row=dialogue();row['domain']=f'domain-{i}';rows.append(row)
    cases,unresolved,selected=synthetic_cases(rows,1)
    assert len(selected)==9 and len(cases)==18 and not unresolved
    assert len({r['metadata']['domain'] for r in cases})==9
    assert all(r['metadata']['cohort']=='synthetic_topic_following' for r in cases)
    assert all(r['metadata']['label_provenance'].endswith('provisional') for r in cases)
