import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from admem.common import InvalidAction, Progress, Unknown, Config, digest, parse, read, rows
from admem.store import apply, augment, raw_chunks, question_key, text_size
from admem.pipeline import builder_state, attacker_state, Collector, run_case, windows, generate_bank
from admem.environment import boolean
from admem.data import prepare, contexts, history_for_builder, policy_type, hint
from admem.probes import make_suite
from admem.cli import _mark_failed, export_states
from admem.bootstrap import bootstrap
from conftest import Full, Pack, CharBudget


def memory():
    return [{'id':'m1','text':'Milo is the cat.','prov':['s1:r1'],'kind':'card'},
            {'id':'m2','text':'Gardening.','prov':['s2:r1'],'kind':'card'}]


def do_apply(m,a,**kw):
    return apply(m,a,mode=kw.get('mode','patch'),visible_ids=kw.get('visible',['m1','m2']),
                 source_ids=['s1:r1','s2:r1'],full_ids=['s1:r1','s2:r1'],counter=CharBudget(),limit=1000)


@pytest.mark.parametrize('action',[
    {'oops':[]}, {'ops':'bad'}, {'ops':[{'op':'RUN','text':'x'}]},
    {'ops':[{'op':'DELETE','id':'hidden'}]},
    {'ops':[{'op':'ADD','text':'x','prov':[]}]},
    {'ops':[{'op':'ADD','text':'x','prov':['not_real']}]},
    {'ops':[{'op':'ADD','text':'x','prov':['s1:r1'],'id':'fake'}]},
    {'ops':[{'op':'MERGE','ids':['m1'],'text':'x','prov':[]}]},
    {'ops':[{'op':'DELETE','id':'m1'},{'op':'DELETE','id':'m1'}]},
    {'ops':[{'op':'UPDATE','id':'m1','text':'','prov':[]}]},
])
def test_atomic_rejection(action):
    original=memory();before=copy.deepcopy(original)
    with pytest.raises(InvalidAction): do_apply(original,action)
    assert original==before


def test_merge_and_update_provenance_union():
    m,changed=do_apply(memory(),{'ops':[{'op':'MERGE','ids':['m1','m2'],'text':'Milo and gardening.','prov':[]}]})
    assert len(m)==1 and m[0]['prov']==['s1:r1','s2:r1']
    assert changed==m
    m,_=do_apply(memory(),{'ops':[{'op':'UPDATE','id':'m1','text':'Milo and garden.','prov':['s2:r1']}]})
    assert m[0]['id']=='m1' and m[0]['prov']==['s1:r1','s2:r1']


def test_refine_no_add_and_no_hidden_delete():
    with pytest.raises(InvalidAction):
        do_apply(memory(),{'ops':[{'op':'ADD','text':'Milo','prov':['s1:r1']}]},mode='refine')
    with pytest.raises(InvalidAction):
        do_apply(memory(),{'ops':[{'op':'DELETE','id':'m2'}]},visible=['m1'])


def test_noop_is_identity_but_copied():
    old=memory();new,changed=do_apply(old,{'ops':[]})
    assert new==old and new is not old and changed==[]


def test_empty_stream_noop_is_rejected():
    with pytest.raises(InvalidAction, match="NOOP"):
        do_apply([], {'ops':[]}, mode='stream', visible=[])


def test_split_round_preserves_all_characters(setup_case,env):
    full,_,_=setup_case
    out=list(windows(full,env.counter,7))
    for rid in full.rounds:
        chunks=[x for w in out for x in w if x['rid']==rid]
        original='\n'.join(f"{m['role']}: {m['content']}" for m in full.rounds[rid].messages)
        assert ''.join(x['text'] for x in chunks)==original
        assert all(env.counter.count(x['text'])<=7 for x in chunks)


def test_fallback_is_full_text_and_deduplicated(setup_case,env):
    full,_,_=setup_case
    r=raw_chunks(full,['s1:r1'],env.counter,150)
    assert 'Milo' in ''.join(x['text'] for x in r)
    assert augment(augment([],r),r)==r


def test_question_dedup_is_not_evidence_dedup():
    q={'q':'What name?', 'question_date':'d','type':'single-session-user','E':['s1:r1']}
    assert question_key(q)!=question_key({**q,'q':'What date?'})
    assert question_key(q)==question_key({**q,'E':['s2:r1']})


def test_builder_prompt_hides_all_test_questions(setup_case,env):
    full,path,ctx=setup_case
    tests=[{'q':'SECRET_TEST_QUESTION','a':'SECRET_TEST_ANSWER','type':'single-session-user','question_date':'d','E':['s1:r1']}]
    state=builder_state(full,path,ctx,[],[{'rid':'s1:r1','date':'2023','text':'My cat is named Milo.'}],'patch',tests,env)
    text=json.dumps(state['prompt'])
    assert 'SECRET_TEST_QUESTION' not in text and 'SECRET_TEST_ANSWER' not in text
    assert 'answer_SECRET' not in text
    payload=json.loads(state['prompt'][1]['content'])
    assert payload['task_type']=='single-session-user'
    assert payload['editable_entry_ids']==[]
    assert payload['allowed_source_rids']==['s1:r1']
    assert 'type_guidance' not in payload
    assert 'STORAGE HINT:' in state['prompt'][0]['content']
    assert 'Ask the assistant' not in text


def test_refine_prompt_requires_safe_shortening_before_noop(setup_case,env):
    full,path,ctx=setup_case
    state=builder_state(full,path,ctx,memory(),[], 'refine',[],env,old_ids=['m1','m2'])
    system=state['prompt'][0]['content']
    assert 'overrides the generic NOOP rule' in system
    assert 'no safe shortening exists' in system
    assert 'In refine mode, two identical cards are redundant' in system


def test_temporal_prompt_requires_resolved_absolute_date(setup_case,env):
    full,path,ctx=setup_case
    ctx['question_type']='temporal-reasoning'
    state=builder_state(full,path,ctx,[],[
        {'rid':'s1:r1','date':'2023-01-05','text':'Yesterday I bought a lamp.'}
    ], 'stream',[],env)
    system=state['prompt'][0]['content']
    assert 'stored text must include the resolved absolute date' in system
    assert "do not store only 'yesterday'" in system


def test_builder_prompt_separates_new_entities_and_source_provenance(setup_case,env):
    full,path,ctx=setup_case
    state=builder_state(full,path,ctx,
                        [{'id':'m1','text':'The user owns a cat.','prov':['s0:r1'],'kind':'card'}],
                        [{'rid':'s1:r1','date':'2023','text':'I adopted a dog.'}],
                        'patch',[],env,old_ids=['m1'])
    system=state['prompt'][0]['content']
    assert 'ADD when x introduces an independent topic, entity, or event' in system
    assert 'one card can retain the old and new details without conflating them' in system
    assert 'adopted Buddy' in system and 'adopted Scout' in system
    assert 'prov MUST include the corresponding rid(s) from allowed_source_rids' in system
    assert 'Use [] only when the replacement text relies entirely on the edited parents' in system


def test_assistant_task_prompt_preserves_assistant_attribution(setup_case,env):
    full,path,ctx=setup_case
    ctx['question_type']='single-session-assistant'
    state=builder_state(
        full,path,ctx,[],
        [{'rid':'s1:r1','date':'2023',
          'text':'Assistant: Consider better lighting, the Rise_0 monitor stand, and a footrest.'}],
        'stream',[],env,
    )
    system=state['prompt'][0]['content']
    assert 'TASK-TYPE SPEAKER OVERRIDE' in system
    assert 'Assistant turns are the target memory content' in system
    assert 'never "The user wants/chose/bought/uses ..."' in system
    assert 'user messages are locator context' in system
    assert 'The assistant said to consider better lighting' in system
    assert 'split it into contiguous ranges across ADD cards' in system


def test_hidden_baseline_same_card_budget(setup_case,env):
    full,path,ctx=setup_case
    ctx['question_type']='multi-session'
    a=builder_state(full,path,ctx,[],[],'stream',[],env)
    env.cfg.hint_mode='hidden'
    b=builder_state(full,path,ctx,[],[],'stream',[],env)
    assert a['entry_limit']==b['entry_limit']
    assert 'task_type' not in json.loads(b['prompt'][1]['content'])


@pytest.mark.parametrize('qtype,forbidden', [
    ('single-session-preference', 'Store only constraints explicitly stated'),
    ('single-session-assistant', 'TASK-TYPE SPEAKER OVERRIDE'),
    ('temporal-reasoning', 'TEMPORAL MEMORY CHECK'),
])
def test_hidden_builder_does_not_leak_type_rules(setup_case,env,qtype,forbidden):
    full,path,ctx=setup_case
    ctx['question_type']=qtype
    env.cfg.hint_mode='hidden'
    state=builder_state(full,path,ctx,[],[{'rid':'s1:r1','date':'2023','text':'Yesterday I bought a lamp.'}], 'stream',[],env)
    text=json.dumps(state['prompt'])
    assert qtype not in text
    assert forbidden not in text


def test_builder_budget_includes_type_specific_rules(setup_case,env):
    full,path,ctx=setup_case
    ctx['question_type']='temporal-reasoning'
    env.cfg.builder_input_tokens=100000
    state=builder_state(full,path,ctx,[],[{'rid':'s1:r1','date':'2023','text':'Yesterday I bought a lamp.'}], 'stream',[],env)
    json.loads(state['prompt'][1]['content'])
    budget=env.counter.prompt_count(state['prompt'])
    env.cfg.builder_input_tokens=budget
    bounded=builder_state(full,path,ctx,[],[{'rid':'s1:r1','date':'2023','text':'Yesterday I bought a lamp.'}], 'stream',[],env)
    assert env.counter.prompt_count(bounded['prompt'])<=budget


def test_attacker_sees_declared_type_but_not_original_ids(setup_case,env):
    full,path,ctx=setup_case
    p=Pack('p',full.fingerprint,'answer_SECRET','session',0,['s1:r1'],list(full.rounds))
    state=attacker_state(full,path,ctx,[],p,[],env,'identity')
    text=json.dumps(state['prompt'])
    assert 'answer_SECRET' not in text and 'TARGET_CANARY' not in text and 'REFERENCE_CANARY' not in text
    assert state['qtype']==ctx['question_type']
    assert 'is_seed' in text


def test_type_hint_does_not_expose_abstention_flag(env):
    ctx={'question_type':'multi-session','question_id':'anything_abs'}
    assert policy_type(ctx,env.cfg,0)=='multi-session'
    env.cfg.hint_mode='hidden'
    assert hint(ctx,env.cfg) is None


def q():
    return {'q':"What is my cat's name?",'a':'Milo','type':'single-session-user','question_date':'2023-04-01','E':['s1:r1']}


def state_for(env, setup_case):
    full,path,ctx=setup_case
    return builder_state(full,path,ctx,[],[{'rid':'s1:r1','date':'2023','text':'My cat is named Milo.'}], 'patch',[q()],env)


def test_builder_reward_uses_raw_action_no_fallback(setup_case,env):
    full,_,_=setup_case;state=state_for(env,setup_case)
    bad=env.builder_score(full,state,'garbage')
    assert bad['reward']==-1 and not bad['legal']
    noop=env.builder_score(full,state,'{"ops":[]}')
    assert noop['accuracy']==0 and noop['reward']==0
    good=env.builder_score(full,state,json.dumps({'ops':[{'op':'ADD','text':'Milo','prov':['s1:r1']}]}))
    assert good['accuracy']==1 and good['faith'] and good['reward']>1


def test_refine_noop_keeps_raw_answer_reward_but_zeroes_effective_reward(setup_case,env):
    """A correct-but-uncompressed refine proposal must not train as a success."""
    full,_,_=setup_case
    state=state_for(env,setup_case)
    state['mode']='refine'
    state['x']=[]
    state['M']=[
        {'id':'m1','text':'The user\'s cat is named Milo.','prov':['s1:r1'],'kind':'card'},
        {'id':'m2','text':'The user\'s cat is named Milo.','prov':['s1:r1'],'kind':'card'},
    ]
    state['old_ids']=['m1','m2']
    result=env.builder_score(full,state,'{"ops":[]}')
    assert result['legal'] and result['faith'] and result['accuracy']==1
    assert result['family_constraint_pass'] is False
    assert result['family_constraint_reason']=='memory was not shortened'
    assert result['reward']==1 and result['effective_reward']==0


def test_refine_merge_preserves_effective_reward(setup_case,env):
    """A valid merge that shortens memory remains eligible for reward."""
    full,_,_=setup_case
    state=state_for(env,setup_case)
    state['mode']='refine'
    state['x']=[]
    state['M']=[
        {'id':'m1','text':'The user\'s cat is named Milo.','prov':['s1:r1'],'kind':'card'},
        {'id':'m2','text':'The user\'s cat is named Milo.','prov':['s1:r1'],'kind':'card'},
    ]
    state['old_ids']=['m1','m2']
    action={'ops':[{'op':'MERGE','ids':['m1','m2'],
                    'text':"The user's cat is named Milo.",'prov':[]}]}
    result=env.builder_score(full,state,json.dumps(action))
    assert result['legal'] and result['faith'] and result['accuracy']==1
    assert result['family_constraint_pass'] is True
    assert result['after_tokens']<result['before_tokens']
    assert result['effective_reward']==result['reward'] and result['effective_reward']>0


def test_noop_repeat_rejects_redundant_update_in_effective_reward(setup_case,env):
    """NOOP family must score an unnecessary update as zero effective reward."""
    full,_,_=setup_case
    state=state_for(env,setup_case)
    state['M']=[{'id':'m1','text':"The user's cat is named Milo.",'prov':['s1:r1'],'kind':'card'}]
    state['old_ids']=['m1']
    action={'ops':[{'op':'UPDATE','id':'m1',
                    'text':"The user's cat is named Milo.",'prov':['s1:r1']}]}
    result=env.builder_score(full,state,json.dumps(action),family='noop_repeat')
    assert result['legal'] and result['faith'] and result['accuracy']==1
    assert result['family_constraint_pass'] is False
    assert result['family_constraint_reason']=='NOOP must contain an empty ops list'
    assert result['reward']>0 and result['effective_reward']==0


def test_unfaithful_action_gets_no_accuracy_bonus(setup_case,env):
    full,_,_=setup_case;state=state_for(env,setup_case)
    result=env.builder_score(full,state,json.dumps({'ops':[{'op':'ADD','text':'Milo UNSUPPORTED','prov':['s1:r1']}]}))
    assert result['reward']==0 and result['faith'] is False


def test_empty_tests_not_vacuous_reward(setup_case,env):
    full,_,_=setup_case;state=state_for(env,setup_case);state['tests']=[]
    with pytest.raises(Unknown): env.builder_score(full,state,'{"ops":[]}')


def test_environment_errors_not_negative_reward(setup_case,env):
    full,_,_=setup_case;state=state_for(env,setup_case);env.control['error']=True
    with pytest.raises(Unknown):
        env.builder_score(full,state,json.dumps({'ops':[{'op':'ADD','text':'Milo','prov':['s1:r1']}]}))


def test_defect_missing_and_already_answerable(setup_case,env):
    full,_,_=setup_case
    assert env.defect(full,[],q())['kind']=='missing'
    assert env.defect(full,memory(),q())['kind']=='none'


def test_reader_never_resolves_provenance_to_raw(setup_case,env):
    m=[{'id':'m1','text':'Some details were omitted.','prov':['s1:r1'],'kind':'card'}]
    result=env.answer(m,q())
    assert result['correct'] is False
    assert env.context(m,q())[0]=='Some details were omitted.'


def test_gate_never_repairs_candidate_in_reward(setup_case,env):
    full,path,ctx=setup_case
    pack=Pack('p',full.fingerprint,'seed','session',0,['s1:r1'],list(full.rounds))
    original=q();original['a']='Rex'
    result=env.gate(full,pack,original,ctx['question_date'],ctx['question_type'])
    assert result['status']=='rejected' and original['a']=='Rex'


def test_attacker_reward_duplicate_same_question_not_different_fact(setup_case,env):
    full,path,ctx=setup_case
    pack=Pack('p',full.fingerprint,'seed','session',0,['s1:r1'],list(full.rounds))
    state=attacker_state(full,path,ctx,[],pack,[],env,0)
    result=env.attacker_score(full,state,json.dumps({'items':[q(),q()]}))
    assert result['reward']==1/env.cfg.questions_per_pack
    assert result['effective_reward']==result['reward']
    assert result['items'][1]['status']=='duplicate'


def test_gate_unknown_rid_rejected_without_api(setup_case,env):
    full,path,ctx=setup_case
    pack=Pack('p',full.fingerprint,'seed','session',0,['s1:r1'],list(full.rounds))
    question=q();question['E']=['evil']
    result=env.gate(full,pack,question,ctx['question_date'],ctx['question_type'])
    assert result['status']=='rejected' and not env.control['prompts']


def test_boolean_string_is_unknown():
    with pytest.raises(Unknown): boolean({'correct':'true'},'correct')


def test_prepare_split_isolates_identical_histories(case,env,tmp_path):
    values=[]
    for i in range(12):
        c=copy.deepcopy(case);c['question_id']=str(i)
        c['haystack_sessions'][1][0]['content']+=str(i//2)
        values.append(c)
    source=tmp_path/'data.json';source.write_text(json.dumps(values))
    out=tmp_path/'prepared'
    prepare(source,out,env.memory_module,[6,2,4])
    manifest=read(out/'manifest.json')
    by={}
    for row in manifest['cases']:
        by.setdefault(row['history_group'],set()).add(row['split'])
        ctx=read(out/'cases'/row['key']/'context.json')
        assert set(ctx)=={'key','question_type','question_date','full_hash','split'}
    assert all(len(s)==1 for s in by.values())
    assert len(list(rows(out/'private_eval.jsonl')))==12


def test_progress_bar_non_tty_emits_one_line_per_update():
    import io
    stream=io.StringIO()
    bar=Progress(2,label='bank',stream=stream)
    bar.update(suffix='p0')
    bar.update(suffix='p1')
    bar.close()
    lines=[line for line in stream.getvalue().splitlines() if line]
    assert len(lines)==2
    assert lines[0].startswith('bank [') and '1/2' in lines[0] and lines[0].endswith('p0')
    assert '2/2' in lines[1] and lines[1].endswith('p1')


def test_contexts_per_type_balances_strata(tmp_path):
    rows=[("a","c1"),("b","c2"),("a","c3"),("b","c4"),("a","c5"),
          ("abstention","c6"),("abstention","c7")]
    cases=[]
    for stype,key in rows:
        cases.append({"key":key,"split":"val","question_type":stype,"stratum":stype})
        folder=tmp_path/"cases"/key;folder.mkdir(parents=True)
        (folder/"context.json").write_text(json.dumps({"key":key,"split":"val","question_type":stype}))
    (tmp_path/"manifest.json").write_text(json.dumps({"cases":cases}))
    got=[c["key"] for c,_ in contexts(tmp_path,"val",per_type=2)]
    assert got==["c1","c2","c3","c4","c6","c7"]
    assert [c["key"] for c,_ in contexts(tmp_path,"val",per_type=1)]==["c1","c2","c6"]


def test_mark_failed_records_unknown_and_keeps_partial_bank(tmp_path):
    folder=tmp_path/'c0000';folder.mkdir()
    (folder/'bank.json').write_text('[]')
    (folder/'bank_log.json').write_text('[]')
    _mark_failed(tmp_path,'c0000','bank',Unknown('API down'))
    marker=read(folder/'failed.json')
    assert marker=={'key':'c0000','command':'bank','status':'unknown','error':'API down'}
    # 半成品不再冒充完整case，但原文仍保留为*.partial.json。
    assert not (folder/'bank.json').exists() and not (folder/'bank_log.json').exists()
    assert (folder/'bank.partial.json').exists() and (folder/'bank_log.partial.json').exists()


def test_history_for_builder_disambiguates_duplicate_session_ids(case):
    duplicate=copy.deepcopy(case)
    duplicate['haystack_session_ids']=['s0','s0','s1','s0']
    duplicate['haystack_dates']=['2023-01-01','2023-01-02','2023-01-03','2023-01-04']
    duplicate['haystack_sessions']=[[{'role':'user','content':c}] for c in 'abcd']
    history=history_for_builder(duplicate)
    assert history['haystack_session_ids']==['s0','s0__duplicate_1','s1','s0__duplicate_2']
    # Dates, order and message text stay byte-for-byte identical.
    assert history['haystack_dates']==duplicate['haystack_dates']
    assert history['haystack_sessions'] is duplicate['haystack_sessions']
    # Cases without duplicates are left untouched.
    assert history_for_builder(case)['haystack_session_ids']==case['haystack_session_ids']


def test_raw_data_target_changes_do_not_change_full(case):
    before=Full.build(case)
    case.update(question='CHANGED',answer='CHANGED',answer_session_ids=[])
    case['haystack_sessions'][0][0]['has_answer']=False
    assert Full.build(case).fingerprint==before.fingerprint


def test_pipeline_patch_and_no_reward_for_fallback(setup_case,env,tmp_path):
    full,path,ctx=setup_case;env.control['build_noop']=True
    out=tmp_path/'run';collect=Collector(tmp_path/'states')
    result=run_case(full,path,ctx,env,out,collect=collect)
    assert result['Q_size']==1 and result['Q_solved']==1
    assert result['model_patch_successes']==1
    assert result['fallback_count']==0
    assert 'Milo' in json.dumps(read(out/'M_final.json'))
    entries=list((tmp_path/'states'/'train'/'builder').glob('*.json'))
    assert entries
    for e in entries:
        assert 'TARGET_CANARY' not in json.dumps(read(e)['prompt'])


def test_empty_q_skips_refine(setup_case,env,tmp_path):
    full,path,ctx=setup_case
    result=run_case(full,path,ctx,env,tmp_path/'run')
    assert result['Q_size']==0
    payloads=[json.loads(p[1][1]['content']) for p in env.control['prompts'] if p[0]=='BUILDER']
    assert all(p.get('mode')!='refine' for p in payloads)


def test_attacker_probe_reports_learning_signal(setup_case,env,tmp_path):
    from admem.probes import run_attacker_probe
    full,path,ctx=setup_case
    out=tmp_path/'probe'
    summary=run_attacker_probe([(ctx,path)],env,out,samples=3)
    assert summary['groups']>=1 and summary['samples_per_group']==3
    assert len(summary['per_group'])==summary['groups']
    assert (out/'samples.jsonl').exists() and (out/'summary.json').exists()
    assert all('reward' in row for row in rows(out/'samples.jsonl'))
    assert all('raw_text' in row for row in rows(out/'raw_samples.jsonl'))


def test_attacker_probe_pack_sample_is_limited_and_deterministic(setup_case,env,tmp_path):
    from admem.probes import run_attacker_probe
    full,path,ctx=setup_case
    first=run_attacker_probe([(ctx,path)],env,tmp_path/'p1',samples=2,pack_sample=1)
    second=run_attacker_probe([(ctx,path)],env,tmp_path/'p2',samples=2,pack_sample=1)
    assert first['groups']==1
    assert [(g['case_key'],g['pack']) for g in first['per_group']]==\
           [(g['case_key'],g['pack']) for g in second['per_group']]


def test_bank_retries_transient_api_failure_without_dropping_case(setup_case,env,tmp_path,monkeypatch):
    full,path,ctx=setup_case
    remote=env.policy('ATTACKER');original=remote.complete;calls={'n':0}
    def flaky(prompt,nonce,temperature=0,max_tokens=4096):
        calls['n']+=1
        if calls['n']==1:
            raise Unknown('synthetic connection reset')
        return original(prompt,nonce,temperature=temperature,max_tokens=max_tokens)
    monkeypatch.setattr(remote,'complete',flaky)
    monkeypatch.setenv('PACK_RETRIES','2')
    monkeypatch.setenv('PACK_BACKOFF','0')
    assert generate_bank(full,path,ctx,env,tmp_path/'bank')==1
    assert calls['n']>=2
    assert all(row['status']!='transport_error' for row in read(tmp_path/'bank'/'bank_log.json'))


def test_bank_generation_and_export(setup_case,env,tmp_path):
    full,path,ctx=setup_case
    assert generate_bank(full,path,ctx,env,tmp_path/'bank')==1
    collector=Collector(tmp_path/'collected')
    state=state_for(env,setup_case)
    collector.save(state,'{"ops":[]}',{'faith':True})
    out=tmp_path/'states.jsonl'
    assert export_states(tmp_path/'collected',out,'builder','train','config.json')==1
    assert list(rows(out))[0]['state']['tests']==[q()]


def test_probe_suite_no_official_target_and_modes(env,tmp_path):
    suite=make_suite(env,tmp_path/'suite',variants=2)
    assert len(suite)==16
    assert {r['state']['mode'] for r in suite}=={'stream','patch','refine'}
    assert all(r['state']['split']=='probe' for r in suite)
    assert all('TARGET_CANARY' not in json.dumps(r['prompt']) for r in suite)


def test_bootstrap_constructs_reward_bearing_states(setup_case,env,tmp_path):
    full,path,ctx=setup_case
    collector=Collector(tmp_path/'collected')
    result=bootstrap(full,path,ctx,memory(),[q()],env,collector,limit=1)
    assert result['states']==3
    loaded=[read(p) for p in (tmp_path/'collected/train/builder').glob('*.json')]
    assert all(r['state']['tests'] for r in loaded)
    assert {r['state']['mode'] for r in loaded}=={'patch','refine'}


def test_resume_configuration_change_rejected(setup_case,env,tmp_path):
    full,path,ctx=setup_case
    run_case(full,path,ctx,env,tmp_path/'run',mode='build')
    env.cfg.hint_mode='hidden'
    with pytest.raises(ValueError): run_case(full,path,ctx,env,tmp_path/'run',mode='build')


def test_environment_fingerprint_portable_for_identical_legacy(tmp_path):
    from admem.common import Config
    a,b=tmp_path/'host_a',tmp_path/'host_b'
    a.mkdir(); b.mkdir()
    for name in ['memory.py','packs.py','retrieve.py','llm.py']:
        (a/name).write_text('# same legacy code')
        (b/name).write_text('# same legacy code')
    assert Config(project=str(a)).fingerprint()==Config(project=str(b)).fingerprint()


def test_collector_identity_excludes_only_full_path(tmp_path):
    from admem.pipeline import Collector
    base={'split':'train','role':'builder','prompt':[], 'full_hash':'content_hash','full_path':'/host_a/full.json'}
    c=Collector(tmp_path/'a')
    first=c.save(base)
    base['full_path']='/host_b/full.json'
    assert c.save(base)==first
    base['full_hash']='different_content'
    assert c.save(base)!=first


def mixed_case():
    """种子s1是个人陈述；s2是无一人称的通用问答（类似sharegpt）。"""
    return {'question_type':'single-session-user','question_date':'2023-04-01',
            'haystack_session_ids':['personal','generic'],'haystack_dates':['2023-01-01','2023-02-01'],
            'haystack_sessions':[[{'role':'user','content':'My cat is named Milo.'},
                                  {'role':'assistant','content':'Noted. LONG_ASSISTANT_TEXT'}],
                                 [{'role':'user','content':'Explain binary search.'},
                                  {'role':'assistant','content':'Binary search halves the range. HIDDEN_NEIGHBOR'}]]}


def test_compact_view_hides_neighbor_assistant_and_restricts_evidence(env,tmp_path):
    full=Full.build(mixed_case());ctx={'key':'k','split':'train','question_type':'multi-session','question_date':'2023-04-01'}
    env.cfg.attacker_view='compact'
    pack=Pack('p',full.fingerprint,'personal','session',0,['s1:r1'],list(full.rounds))
    state=attacker_state(full,tmp_path/'f.json',ctx,[],pack,[],env,0)
    text=json.dumps(state['prompt'])
    assert 'LONG_ASSISTANT_TEXT' in text and 'HIDDEN_NEIGHBOR' not in text and 'assistant_elided' in text
    assert state['pack']['rids']==['s1:r1','s2:r1']
    ctx['question_type']='single-session-user'
    single=attacker_state(full,tmp_path/'f.json',ctx,[],pack,[],env,0)
    assert single['pack']['rids']==['s1:r1'] and 'binary search' not in json.dumps(single['prompt'])
    question={'q':'Explain?','a':'Milo','type':'single-session-user','question_date':'2023-04-01','E':['s2:r1']}
    result=env.gate(full,Pack(**single['pack']),question,'2023-04-01','single-session-user')
    assert result['status']=='rejected'


def test_type_filter_skips_infeasible_pack_without_calling_attacker(env,tmp_path):
    full=Full.build(mixed_case());ctx={'key':'k','split':'train','question_type':'single-session-user','question_date':'2023-04-01'}
    env.cfg.attacker_type_filter=True
    generic=Pack('p',full.fingerprint,'generic','session',0,['s2:r1'],list(full.rounds))
    assert attacker_state(full,tmp_path/'f.json',ctx,[],generic,[],env,0) is None
    ctx['question_type']='single-session-assistant'
    assert attacker_state(full,tmp_path/'f.json',ctx,[],generic,[],env,0)['qtype']=='single-session-assistant'
    env.cfg.hint_mode='hidden'
    for i in range(20):
        assert attacker_state(full,tmp_path/'f.json',ctx,[],generic,[],env,i)['qtype']=='single-session-assistant'
    assert not env.control['prompts']


def test_attacker_reward_decays_on_reused_evidence(setup_case,env):
    full,path,ctx=setup_case
    pack=Pack('p',full.fingerprint,'seed','session',0,['s1:r1'],list(full.rounds))
    other=q();other['q']="What did I name my cat?"
    fresh=attacker_state(full,path,ctx,[],pack,[],env,0)
    reused=attacker_state(full,path,ctx,[],pack,[q()],env,0)
    first=env.attacker_score(full,fresh,json.dumps({'items':[q(),other]}))
    assert [i['weight'] for i in first['items']]==[1.0,0.5]
    assert env.attacker_score(full,reused,json.dumps({'items':[other]}))['items'][0]['weight']==0.5
    env.cfg.evidence_novelty=False
    assert env.attacker_score(full,fresh,json.dumps({'items':[q(),other]}))['reward']==2/env.cfg.questions_per_pack


def test_attacker_reward_downweights_impersonal_evidence(env,tmp_path):
    case=mixed_case();case['haystack_sessions'][0][0]['content']='The cat is named Milo.'
    full=Full.build(case);ctx={'key':'k','split':'train','question_type':'single-session-user','question_date':'2023-04-01'}
    pack=Pack('p',full.fingerprint,'personal','session',0,['s1:r1'],list(full.rounds))
    state=attacker_state(full,tmp_path/'f.json',ctx,[],pack,[],env,0)
    result=env.attacker_score(full,state,json.dumps({'items':[q()]}))
    assert result['items'][0]['weight']==env.cfg.impersonal_weight
