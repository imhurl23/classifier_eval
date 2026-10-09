"""Pinned original HH-RLHF dialogues. No Aegis reply splicing or preference-as-safety labels."""
from __future__ import annotations
import gzip,hashlib,json,re,random
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='c72f5cee8eb7b4d2ea5617657f4430d5e333af07'
SEED=20261007


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def normalized(text):return ' '.join(text.split())


def parse_transcript(text):
    matches=list(re.finditer(r'\n\n(Human|Assistant): ?',text))
    if not matches or text[:matches[0].start()].strip():raise ValueError('Missing/ambiguous first role')
    messages=[]
    for i,match in enumerate(matches):
        role='user' if match.group(1)=='Human' else 'assistant'
        expected='user' if i%2==0 else 'assistant'
        content=text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        if role!=expected or not content.strip():raise ValueError('Nonalternating/empty source turn')
        messages.append({'turn_id':f'turn_{i+1:04d}','role':role,'content':content})
    if messages[-1]['role']!='assistant':raise ValueError('Original dialogue has no final Assistant reply')
    return messages


def render(messages):
    return '\n\n'.join(f"{m['turn_id']} | {'User' if m['role']=='user' else 'Assistant'}:\n{m['content']}" for m in messages)


def source_rows(split):
    path=ROOT/f'data/raw/upstream-hh/{REVISION}-{split}.jsonl.gz'
    for index,line in enumerate(gzip.decompress(path.read_bytes()).decode().splitlines()):
        yield index,json.loads(line)


def load_originals():
    records=[];excluded=[];seen={}
    for source_split in ('test','train'):
        for index,row in source_rows(source_split):
            try:
                choices={branch:parse_transcript(row[branch]) for branch in ('chosen','rejected')}
                contexts={digest(m[:-1]) for m in choices.values()}
                if len(contexts)!=1:raise ValueError('Preference branches have differing prior contexts')
                group_id=next(iter(contexts))
                branch=('chosen','rejected')[int(group_id[:8],16)%2]
                messages=choices[branch];raw=row[branch]
                if group_id in seen:
                    excluded.append({'source_split':source_split,'source_index':index,'reason':'duplicate_conversation_group','earlier':seen[group_id]});continue
                seen[group_id]={'split':source_split,'index':index}
                item_id=f'hh-full-{source_split}-{index:06d}-{digest(messages)[:12]}'
                provenance={'repo':'anthropics/hh-rlhf','revision':REVISION,'file':f'harmless-base/{source_split}.jsonl.gz',
                    'row_index':index,'branch':branch,'original_transcript_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                    'selection':'One branch per original conversation group selected by deterministic group hash; branch preference is not a safety label.'}
                records.append({'id':item_id,'input':{'item_id':item_id,'messages':messages,'text':render(messages),
                    'prompt':messages[-2]['content'],'response':messages[-1]['content'],'unit':'conversation','history_complete':True,
                    'target_user_turn_id':messages[-2]['turn_id'],'target_assistant_turn_id':messages[-1]['turn_id']},
                    'expected':{'s2':None,'prompt_s2':None,'response_s2':None,'s3':None,'s6':None,'s13':None},
                    'metadata':{'source':'hh_rlhf_original','source_split':source_split,'split':'validation' if source_split=='train' else 'test',
                     'conversation_group_id':group_id,'turn_count':len(messages),'gold_provenance':'No human safety labels yet; HH preference is not harm gold',
                     'label_source':None,'context_policy':'original_full_conversation_v2'},'provenance':provenance})
            except ValueError as exc:excluded.append({'source_split':source_split,'source_index':index,'reason':str(exc)})
    return records,excluded


def prepare():
    originals,excluded=load_originals()
    old_packet=ROOT/'data/processed/offline-review/f0c7bbda4b65/private/sample.jsonl'
    old=list(map(json.loads,old_packet.read_text().splitlines()))
    old_prompts=defaultdict(list)
    for r in old:old_prompts[normalized(r['record']['input']['prompt'])].append(r['audit_id'])
    all_upstream_contexts=defaultdict(set)
    for source_split in ('train','test'):
        for _,upstream in source_rows(source_split):
            for branch in ('chosen','rejected'):
                try:messages=parse_transcript(upstream[branch])
                except ValueError:continue
                for position,message in enumerate(messages):
                    if message['role']=='user':
                        for aid in old_prompts.get(normalized(message['content']),[]):
                            all_upstream_contexts[aid].add(digest(messages[:position]))
    # Any old development prompt seen as a User turn quarantines the corresponding new test dialogue.
    exposure=set(old_prompts)
    for folder in ('study-kickoff/0ecb68a006dd','study-reference/00de7797f0a6'):
        p=ROOT/'data/processed'/folder/'sample.jsonl'
        if p.exists():
            for r in map(json.loads,p.read_text().splitlines()):exposure.add(normalized(r['input']['prompt']))
    p=ROOT/'data/processed/dis-pilot.jsonl'
    for r in map(json.loads,p.read_text().splitlines()):exposure.add(normalized(r['input']['prompt']))
    links=[];contexts=defaultdict(lambda:defaultdict(set));quarantine=[];held=[];train=[];controls=[]
    for record in originals:
        messages=record['input']['messages']
        linked=[]
        for position,message in enumerate(messages):
            if message['role']=='user':
                for aid in old_prompts.get(normalized(message['content']),[]):
                    links.append({'old_audit_id':aid,'new_item_id':record['id'],'matched_turn_id':message['turn_id'],
                      'prior_context_sha256':digest(messages[:position]),'has_prior_turns':position>0,
                      'match_basis':'Exact whitespace-normalized User text; NOT verified Aegis source-ID provenance.'});linked.append(aid)
                    contexts[aid][digest(messages[:position])].add(record['id'])
        record['metadata']['old_audit_text_match_count']=len(linked)
        if len(messages)==2:controls.append(record);continue
        if record['metadata']['source_split']=='test':
            if any(m['role']=='user' and normalized(m['content']) in exposure for m in messages):
                quarantine.append({'id':record['id'],'reason':'Earlier development User-text exposure','conversation_group_id':record['metadata']['conversation_group_id']})
            else:held.append(record)
        else:train.append(record)
    # Keep previously identified dialogue candidates where possible, then fill reproducibly by length/turn coverage.
    # Generic replies such as "yes" have many histories: never use those as recovered-context anchors.
    anchored_ids=set()
    train_by_id={r['id']:r for r in train}
    for aid,candidates in contexts.items():
        if len(all_upstream_contexts[aid])==1 and digest([]) not in all_upstream_contexts[aid]:
            candidates_in_train=sorted(next(iter(candidates.values())) & set(train_by_id))
            if candidates_in_train:anchored_ids.add(candidates_in_train[0])
    linked=sorted((train_by_id[i] for i in anchored_ids),key=lambda r:r['id'])
    rng=random.Random(SEED);rng.shuffle(linked)
    selected=linked[:200];selected_ids={r['id'] for r in selected}
    groups=defaultdict(list)
    for r in train:
        if r['id'] not in selected_ids:
            groups[(min(r['metadata']['turn_count'],10), min(len(r['input']['text'])//4000,4))].append(r)
    for group in groups.values():rng.shuffle(group)
    while len(selected)<200:
        for key in sorted(groups):
            if groups[key] and len(selected)<200:selected.append(groups[key].pop())
        if not any(groups.values()) and len(selected)<200:raise ValueError('Not enough full development conversations')
    rng.shuffle(selected)
    version=digest({'source_revision':REVISION,'source_hashes':{s:hashlib.sha256((ROOT/f'data/raw/upstream-hh/{REVISION}-{s}.jsonl.gz').read_bytes()).hexdigest() for s in ('train','test')},
       'builder_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selected':[r['id'] for r in selected],'held':[r['id'] for r in held]})[:12]
    out=ROOT/'data/processed/full-conversations'/version;out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('development',selected),('heldout-candidate',held),('two-turn-controls',controls),('upstream-text-links',links),('quarantine',quarantine),('source-exclusions',excluded)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    summary={'version':version,'context_policy':'original_full_conversation_v2','source_revision':REVISION,'status':'Prepared; no new human safety labels or model results',
      'development_conversations':len(selected),'development_with_previous_audit_text_links':sum(bool(r['metadata']['old_audit_text_match_count']) for r in selected),
      'unique_context_anchor_conversations_retained':len(anchored_ids),
      'heldout_candidate_conversations':len(held),'quarantined_test_conversations':len(quarantine),'two_turn_controls':len(controls),
      'source_excluded_or_duplicate_records':len(excluded),'turn_counts':dict(Counter(r['metadata']['turn_count'] for r in selected)),
      'dev_max_turns':max(r['metadata']['turn_count'] for r in selected),'dev_max_characters':max(len(r['input']['text']) for r in selected),
      'gold':'Human annotation required on these exact original dialogues. No Aegis labels transferred; no preference-as-safety mapping.',
      'source_lineage':'Pinned original upstream row and branch verified directly. Links to old Aegis audit items are text candidates, not proven origin matches.',
      'heldout_note':'Source test split remains unexecuted. Old-development text exposures are quarantined; full dialogue group never crosses development/test.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'directory':str(out.relative_to(ROOT)),**summary},indent=2));return out

if __name__=='__main__':prepare()
