"""显式离线接口替身，只供测试；生产代码不会自动回退到这些实现。"""
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from admem.common import Config, TokenBudget, digest, read, write
from admem.environment import Environment


class CharBudget:
    """测试用字符计数，不冒充Qwen tokenizer。"""
    def count(self, text):
        return len(text)
    def prompt_count(self, messages):
        return sum(len(m['content']) for m in messages)
    split = TokenBudget.split


class Full:
    def __init__(self, sessions, rounds):
        self.sessions = {s.session_id: s for s in sessions}
        self.rounds = {r.rid: r for r in rounds}
        self.fingerprint = digest(self.export_history())

    @classmethod
    def build(cls, case):
        sessions, rounds = [], []
        for si, (sid, date, messages) in enumerate(zip(case['haystack_session_ids'], case['haystack_dates'], case['haystack_sessions'])):
            ids = []
            for mi in range(0, len(messages), 2):
                rid = f's{si+1}:r{mi//2+1}'
                ms = [{k:v for k,v in m.items() if k != 'has_answer'} for m in messages[mi:mi+2]]
                rounds.append(NS(rid=rid, session_id=sid, messages=ms, message_indices=list(range(mi,min(mi+2,len(messages))))))
                ids.append(rid)
            sessions.append(NS(session_id=sid, date=date, original_index=si, rids=ids))
        return cls(sessions, rounds)

    def ordered(self, ids):
        return sorted(set(ids), key=lambda r: (self.sessions[self.rounds[r].session_id].date, r))

    def export_history(self):
        ss = sorted(self.sessions.values(), key=lambda s:s.original_index)
        return {'haystack_session_ids':[s.session_id for s in ss], 'haystack_dates':[s.date for s in ss],
                'haystack_sessions':[[m for rid in s.rids for m in self.rounds[rid].messages] for s in ss]}

    def save(self, path):
        write(path, {'schema':1,'history':self.export_history(),'fingerprint':self.fingerprint})

    @classmethod
    def load(cls,path):
        data=read(path)
        if 'history' in data:
            result=cls.build(data['history'])
        else:
            result=cls([NS(**s) for s in data['sessions']], [NS(**r) for r in data['rounds']])
        return result

    def documents(self):
        return [Document(r.rid,' '.join(m['content'] for m in r.messages),'',[r.session_id],[r.rid],[]) for r in self.rounds.values()]


@dataclass
class Document:
    id:str
    text:str
    date:str
    session_ids:list
    prov:list
    neighbors:list


@dataclass
class Pack:
    pack_id:str
    full_hash:str
    seed_id:str
    kind:str
    sweep:int
    seed_rids:list
    rids:list
    dropped_rids:list=None
    searches:list=None
    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


class Retriever:
    def __init__(self,docs,*args):
        self.docs={d.id:d for d in docs}
    def search(self,q,k,*args,**kwargs):
        terms=set(re.findall(r'\w+',q.lower()))
        ranked=sorted(self.docs.values(), key=lambda d:-len(terms & set(re.findall(r'\w+',d.text.lower()))))
        return [NS(id=d.id) for d in ranked[:k]]
    def search_many(self,qs,k,*args,**kwargs):
        return self.search(' '.join(qs),k)


class Sampler:
    def __init__(self,full,index,**kwargs):
        self.full=full
        self.seeds=list(full.sessions)
    def sample(self,n):
        return [Pack(f'p{i+1}',self.full.fingerprint,sid,'session',0,self.full.sessions[sid].rids,
                     list(self.full.rounds)) for i,sid in enumerate(self.seeds[:n])]


class FakeRemote:
    def __init__(self,role,control):
        self.role,self.control=role,control
        self.tag=role+'_OFFLINE_STUB'
        self.client=NS(model='OFFLINE_STUB',base_url='http://not-used')
    def complete(self,prompt,nonce,temperature=0,max_tokens=4096):
        self.control['prompts'].append((self.role,prompt,nonce))
        payload=json.loads(prompt[1]['content'])
        if 'mode' in payload:
            if payload['mode']=='stream' and self.control.get('build_noop'):
                return '{"ops":[]}'
            if payload['mode']=='refine':
                return '{"ops":[]}'
            return json.dumps({'ops':[{'op':'ADD','text':x['text'],'prov':[x['rid']]} for x in payload['x']]})
        return json.dumps({'items':[{'q':"What is my cat's name?",'a':'Milo','type':payload['type'],
                                     'question_date':payload['question_date'],'E':['s1:r1']}]})
    def json(self,prompt,nonce):
        from admem import prompts
        from admem.audit_prompts import GATE_ORACLE_SYSTEM,GATE_SUPPORT_SYSTEM
        from admem.common import Unknown
        self.control['prompts'].append((self.role,prompt,nonce))
        if self.control.get('error'):
            raise Unknown('explicit synthetic environment error')
        system=prompt[0]['content'];p=json.loads(prompt[1]['content'])
        if system==prompts.READ:
            return {'answer':'Milo' if 'Milo' in p['history'] else 'Unknown'}
        if system==prompts.GRADE:
            return {'correct':p['prediction'].lower()==p['reference'].lower(),'reason':'stub'}
        if system==prompts.FAITH:
            return {'faithful':'UNSUPPORTED' not in json.dumps(p['entries']),'reason':'stub'}
        if system==GATE_ORACLE_SYSTEM:
            return {'answerable':True,'answer':'Milo','user_relevant':True,'type_valid':True,'no_answer_leak':True}
        if system==GATE_SUPPORT_SYSTEM:
            return {'correct':p['a']=='Milo','reason':'stub'}
        if system==prompts.SCREEN:
            return {'stable':not self.control.get('unstable'), 'additional_rids':[],'reason':'stub'}
        raise AssertionError(system[:80])


@pytest.fixture
def env(tmp_path):
    cfg=Config(embedding='none',cache=str(tmp_path/'cache'),window_tokens=300,entry_tokens=1000,
               builder_input_tokens=20000,attacker_input_tokens=30000,judge_input_tokens=100000,
               reader_tokens=20000,sweeps=1,refine_rounds=1)
    legacy=(NS(FullMemory=Full,Document=Document),NS(Pack=Pack,Sampler=Sampler),NS(Retriever=Retriever),NS(Client=None))
    e=Environment(cfg,CharBudget(),legacy)
    e.control={'prompts':[]}
    e.clients={r:FakeRemote(r,e.control) for r in ['BUILDER','ATTACKER','JUDGE','DEFENDER','TEACHER']}
    return e


@pytest.fixture
def case():
    return {'question_id':'private_id', 'question_type':'single-session-user','question':'TARGET_CANARY',
            'answer':'REFERENCE_CANARY','question_date':'2023-04-01','answer_session_ids':['answer_SECRET'],
            'haystack_session_ids':['answer_SECRET','another_source'], 'haystack_dates':['2023-01-01','2023-02-01'],
            'haystack_sessions':[[{'role':'user','content':'My cat is named Milo.','has_answer':True},
                                  {'role':'assistant','content':'Noted.'}],
                                 [{'role':'user','content':'I enjoy gardening.'},{'role':'assistant','content':'Noted.'}]]}


@pytest.fixture
def setup_case(case,tmp_path):
    f=Full.build(case);p=tmp_path/'full.json';f.save(p)
    context={'key':'c0000','split':'train','question_type':case['question_type'],'question_date':case['question_date']}
    return f,p,context
