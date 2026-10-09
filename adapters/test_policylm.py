from types import SimpleNamespace as NS

import pytest

from adapters.policylm import CUTOFF, PolicyLMPromptAdapter, make_prompt_request, parse_result
from prompts.runtime import instructions_for, options_for


def test_request_uses_single_user_message_and_native_policy():
    _, definitions = options_for('s2', 'prompt_harm_s2')
    req = make_prompt_request('Hello', instructions_for('prompt_harm_s2'), definitions)
    assert req['message'] == 'Hello' and req['input_context'] == 'user_only'
    assert 'Assistant turn as context' not in str(req['policy'])
    assert req['clean'] is False and req['cutoff'] == CUTOFF
    with pytest.raises(ValueError):
        make_prompt_request('x','y',{'safe':'safe','unsafe':'unsafe','other':'other'})


@pytest.mark.parametrize('score,flagged,status',[(.7,True,'ok'),(.1,False,'ok'),(float('nan'),False,'invalid_output'),(.7,False,'invalid_output')])
def test_native_score_not_converted_to_choice_probabilities(score,flagged,status):
    response=NS(to_dict=lambda:{'score':score,'cutoff':CUTOFF,'flagged':flagged,'windows':2})
    decision=parse_result(response,{'message':'x'},1)
    assert decision.status == status and decision.probs is None
    if status=='ok': assert decision.label == ('unsafe' if flagged else 'safe')


def test_adapter_uses_release_preset_and_preserves_message():
    calls=[]
    def classify(text,policy,**kwargs):
        calls.append((text,policy,kwargs))
        return NS(to_dict=lambda:{'score':.8,'cutoff':CUTOFF,'flagged':True,'windows':1})
    adapter=PolicyLMPromptAdapter({'safe':'safe','unsafe':'unsafe'},model=NS(classify=classify))
    decision=adapter.classify_prompt('User text only','Study policy')
    assert decision.label=='unsafe'
    assert calls[0][0]=='User text only' and calls[0][2]=={'cutoff':'precision','clean':False}
