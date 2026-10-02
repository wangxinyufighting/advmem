"""离线验证英文模板、协议、标签隔离和受控源码补丁；不声称模型效果已测。"""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_prompts as mp
import install_prompt_v2 as installer


@pytest.fixture
def memory():
    sessions = {'answer_secret': NS(date='2023-01-03', node_id='s1'),
                'private_sid': NS(date='2023-04-07', node_id='s2')}
    rounds = {
        's1:r1': NS(session_id='answer_secret', messages=[
            {'role':'user', 'content':'I worked at Company A for two years.', 'has_answer':True},
            {'role':'assistant', 'content':'You also asked about further study.', 'secret_label':'GOLD'}]),
        's2:r1': NS(session_id='private_sid', messages=[
            {'role':'user', 'content':'After a one-year gap, I worked at Company B for three years.'}])}
    full = NS(fingerprint='case-hash', rounds=rounds, sessions=sessions, ordered=lambda ids: sorted(set(ids)))
    pack = NS(full_hash='case-hash', rids=['s2:r1','s1:r1'], seed_rids=['s2:r1'], seed_id='private_sid')
    return full, pack


class Stub:
    def __init__(self, response):
        self.response, self.calls = response, []
    def json(self, system, data, **kwargs):
        self.calls.append((system, data, kwargs))
        return self.response


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    module = ModuleType('llm')
    module.ModelError = type('ModelError', (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, 'llm', module)


@pytest.mark.parametrize('kind', mp.TYPES)
def test_all_six_prompts_are_english_and_type_specific(kind):
    prompt = mp.attacker_prompt(kind)
    assert prompt.isascii()
    assert f'TYPE: {kind}' in prompt
    assert 'all of the requested type' in prompt
    assert 'items' in prompt


def test_known_mismatches_have_explicit_rules():
    assert 'those are factual recall questions' in mp.TYPE_GUIDANCE['single-session-preference']
    assert 'earlier or initial value' in mp.TYPE_GUIDANCE['knowledge-update']
    assert 'ordinal' in mp.TYPE_GUIDANCE['single-session-assistant']
    assert 'at least two distinct sessions' in mp.TYPE_GUIDANCE['multi-session']
    assert 'One session may contain enough' in mp.TYPE_GUIDANCE['temporal-reasoning']
    assert 'gaps, and overlaps' in mp.TYPE_GUIDANCE['multi-session']


def test_payload_whitelist_and_neutral_ids(memory):
    full, pack = memory
    payload = mp.build_payload(full, pack, 'multi-session', '2023-09-01')
    text = json.dumps(payload)
    assert all(term not in text for term in ['answer_secret','private_sid','has_answer','secret_label','GOLD'])
    assert {r['session'] for r in payload['rounds']} == {'s1','s2'}
    assert [r['rid'] for r in payload['rounds'] if r['is_seed']] == ['s2:r1']
    assert payload['max_evidence_rounds'] == 8
    assert payload['rounds'][0]['messages'][0]['content'] == full.rounds['s1:r1'].messages[0]['content']


def test_does_not_modify_full_or_pack(memory):
    full, pack = memory
    before = copy.deepcopy(full.rounds['s1:r1'].messages)
    ids = pack.rids[:]
    mp.build_payload(full, pack, 'single-session-user', '2023-09-01')
    assert before == full.rounds['s1:r1'].messages and ids == pack.rids


def test_labels_do_not_change_generation_input(memory):
    full, pack = memory
    a = mp.build_payload(full, pack, 'single-session-user', '2023-09-01')
    full.rounds['s1:r1'].messages[0]['has_answer'] = False
    full.rounds['s1:r1'].messages[0]['target_question'] = 'TARGET_CANARY'
    assert a == mp.build_payload(full, pack, 'single-session-user', '2023-09-01')


def test_provenance_is_not_treated_as_fact_coverage(memory):
    full, pack = memory
    a = mp.build_payload(full, pack, 'single-session-user', '2023-09-01', marks={'s1:r1':['m1']})
    assert a['rounds'][0]['referenced_by_memory'] == ['m1']
    assert 'does not mean every' in mp.ATTACKER_SYSTEM


def test_wrong_memory_and_unknown_rids_fail(memory):
    full, pack = memory
    pack.full_hash = 'other-case'
    with pytest.raises(ValueError): mp.build_payload(full, pack, 'multi-session','2023-01-01')
    pack.full_hash = full.fingerprint
    pack.rids.append('s999:r1')
    with pytest.raises(ValueError): mp.build_payload(full, pack, 'multi-session','2023-01-01')


def test_seed_rounds_cannot_disappear(memory):
    full, pack = memory
    pack.rids = ['s1:r1']
    with pytest.raises(ValueError): mp.build_payload(full, pack, 'multi-session','2023-01-01')


def test_prior_context_is_explicit_bounded_and_case_bound(memory):
    full, pack = memory
    prior = [{'q':f'Question {i}?','a':'A','type':'single-session-user','E':['s1:r1']} for i in range(30)]
    context = {'full_hash': full.fingerprint, 'accepted_items':prior}
    payload = mp.build_payload(full, pack, 'multi-session','2023-01-01',audit_context=context)
    assert len(payload['audit_context']['accepted_items']) == 24
    assert payload['audit_context']['omitted_relevant_items'] == 6
    context['full_hash']='wrong'
    with pytest.raises(ValueError): mp.build_payload(full, pack, 'multi-session','2023-01-01',audit_context=context)


def test_prior_context_rejects_target_metadata(memory):
    full, pack = memory
    with pytest.raises(ValueError):
        mp.build_payload(full,pack,'multi-session','2023-01-01',audit_context={'target_question':'secret'})
    assert 'audit_context' not in mp.build_payload(full,pack,'multi-session','2023-01-01')


def test_generator_preserves_list_contract_and_nonce(memory):
    full, pack = memory
    rows = [{'q':'How long did I work across the two positions?', 'a':'Five years.',
             'type':'multi-session', 'question_date':'2023-09-01', 'E':pack.rids}]
    stub = Stub({'items':rows})
    assert mp.generate_questions(stub,full,pack,'multi-session','2023-09-01',nonce='trial-3') == rows
    system, data, kw = stub.calls[0]
    assert system.isascii() and kw['temperature'] == .7
    assert kw['nonce'].endswith('trial-3') and mp.PROMPT_VERSION in kw['nonce']


def test_bad_output_is_error_not_empty(memory):
    full, pack = memory
    with pytest.raises(RuntimeError): mp.generate_questions(Stub({'items':'bad'}),full,pack,'multi-session','2023-01-01')
    with pytest.raises(RuntimeError): mp.generate_questions(Stub({'items':[{}]*5}),full,pack,'multi-session','2023-01-01')


def test_zero_question_budget_makes_no_call(memory):
    full, pack = memory
    stub=Stub({'items':[]})
    assert mp.generate_questions(stub,full,pack,'multi-session','2023-01-01',0) == []
    assert not stub.calls


def test_aliases_keep_canonical_type_and_limits(memory):
    full,pack=memory
    assert mp.build_payload(full,pack,'preference','date')['type'] == 'single-session-preference'
    assert mp.build_payload(full,pack,'preference','date')['max_evidence_rounds'] == 3
    assert mp.build_payload(full,pack,'temporal','date')['max_evidence_rounds'] == 8
    with pytest.raises(ValueError): mp.attacker_prompt('abstention')


def test_support_verifier_receives_evidence_and_actual_question():
    stub=Stub({'correct':True,'reason':'Both requested store names were supplied.'})
    value=mp.verify_support(stub,'Which two stores?','Store X: 10%; Store Y: 15%.',
                            'Store X and Store Y.','multi-session',evidence='source',date='2023-01-01')
    assert value['correct'] is True
    assert stub.calls[0][1]['E_history'] == 'source'
    assert 'incidental details' in stub.calls[0][0]
    # The stub verifies the protocol only; it does not prove a real judge will agree.


def test_support_bool_validation():
    with pytest.raises(RuntimeError):
        mp.verify_support(Stub({'correct':'true','reason':'bad bool'}),'q','a','p','multi-session',evidence='e',date='d')


def test_evidence_renderer_hides_raw_session_ids(memory):
    full,pack=memory
    text=mp.render_evidence(full,pack.rids)
    assert 'answer_secret' not in text and 'has_answer' not in text
    assert len(json.loads(text)) == 2


SOURCE='''"""Old module documentation."""
from __future__ import annotations

class Attacker:
    def __init__(self, model, full, marks=None):
        self.model, self.full, self.marks = model, full, marks

    def generate(self, pack, qtype, date, n_questions=4, nonce=""):
        return self.model.json("旧出题规则", {})["items"]

def grade(*args):
    return {"correct": True, "reason": "unchanged"}

def gate(item, pack, full, oracle, date):
    context = full.render(item["E"])
    screen = oracle.json("旧筛查规则", {"candidate": item, "history": context})
    decision = oracle.json("旧独立答案规则", {"q":item["q"], "E_history":context})
    support = grade(oracle, item["q"], item["a"], decision["answer"], item["type"])
    return support
'''
LLM_SOURCE='''class Client:
    def json(self, system, data):
        system = system + "\\n仅返回一个有效 JSON 对象。"
        return {"system": system, "data": data}
'''


def test_installer_only_generator_by_default():
    out=installer.patch_agents(SOURCE)
    assert 'generate_questions' in out and '旧出题规则' not in out
    assert '旧筛查规则' in out and '旧独立答案规则' in out
    assert 'def grade(*args):\n    return {"correct": True, "reason": "unchanged"}' in out
    ast.parse(out, feature_version=(3,10))


def test_optional_gate_patch_changes_support_not_baseline_grade():
    out=installer.patch_agents(SOURCE,True)
    assert 'GATE_SCREEN_SYSTEM' in out and 'GATE_ORACLE_SYSTEM' in out
    assert 'verify_support(oracle' in out and 'render_evidence(full' in out
    assert 'def grade(*args):\n    return {"correct": True, "reason": "unchanged"}' in out
    assert '旧筛查规则' not in out
    assert installer.patch_agents(out,True) == out


def test_installer_fails_on_unknown_signature():
    with pytest.raises(ValueError): installer.patch_agents(SOURCE.replace('n_questions=4','count=4'))
    with pytest.raises(ValueError): installer.patch_agents(SOURCE.replace('support = grade','result = grade'),True)


def test_json_suffix_english_not_network_change():
    out,n=installer.patch_llm(LLM_SOURCE)
    assert n==1 and '仅返回' not in out and 'Return exactly' in out
    assert installer.patch_llm(out) == (out,0)


def test_install_dryrun_backup_and_apply(tmp_path):
    (tmp_path/'agents.py').write_text(SOURCE)
    (tmp_path/'llm.py').write_text(LLM_SOURCE)
    installer.main(['--project',str(tmp_path)])
    assert (tmp_path/'agents.py').read_text() == SOURCE
    assert not (tmp_path/'memory_prompts.py').exists()
    installer.main(['--project',str(tmp_path),'--apply'])
    assert (tmp_path/'memory_prompts.py').is_file()
    assert len(list(tmp_path.glob('*.bak'))) == 2
    assert list(tmp_path.glob('agents.py.pre_prompt_v2.*.bak'))[0].read_text()==SOURCE


def test_all_gate_templates_are_english():
    for s in (mp.GATE_ORACLE_SYSTEM,mp.GATE_SUPPORT_SYSTEM,mp.GATE_SCREEN_SYSTEM):
        assert s.isascii()
    assert 'not an exhaustive' in mp.GATE_SCREEN_SYSTEM
    assert 'unknown' not in mp.TYPE_GUIDANCE  # no invented qtype


def test_runtime_templates_do_not_contain_original_targets():
    # Canonical hard-coded answers and distinctive target phrases must not enter prompts.
    joined='\n'.join([mp.ATTACKER_SYSTEM,*mp.TYPE_GUIDANCE.values()])
    for forbidden in ['Arcadia','Pasadena','UCLA','10 years','What degree did I graduate with?',
                      "How many years in total did I spend in formal education"]:
        assert forbidden not in joined
