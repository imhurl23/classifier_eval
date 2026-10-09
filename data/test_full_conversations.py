"""Full transcript preservation and source grouping must not depend on safety labels."""
import pytest
from data.full_conversations import parse_transcript,render,digest


def test_every_turn_and_exact_content_preserved():
    text='\n\nHuman: initial request\n\nAssistant: question\n\nHuman: yes, that one\n\nAssistant: final reply'
    m=parse_transcript(text)
    assert [x['role'] for x in m]==['user','assistant','user','assistant']
    assert [x['content'] for x in m]==['initial request','question','yes, that one','final reply']
    r=render(m)
    assert all(x['content'] in r for x in m) and 'turn_0004' in r


@pytest.mark.parametrize('text',['unassigned preamble\n\nHuman: p\n\nAssistant: r','\n\nHuman: p\n\nHuman: another','\n\nHuman: p','\n\nHuman: \n\nAssistant: r'])
def test_ambiguous_or_incomplete_role_structure_is_rejected(text):
    with pytest.raises(ValueError):parse_transcript(text)


def test_preference_branches_share_group_without_becoming_safety_labels():
    a=parse_transcript('\n\nHuman: p\n\nAssistant: answer one')
    b=parse_transcript('\n\nHuman: p\n\nAssistant: answer two')
    assert digest(a[:-1])==digest(b[:-1]) and digest(a)!=digest(b)
