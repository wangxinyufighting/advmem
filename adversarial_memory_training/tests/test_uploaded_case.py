"""可选：只读取用户上传的真实case，检查中性输入和原文分片；不调用LLM。"""
import json
import os
import zipfile
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

from admem.pipeline import attacker_state, windows
from conftest import Full, Pack


def test_real_case_windows_and_all_packs(env,tmp_path):
    path=os.getenv('ADMEM_CASE_ZIP')
    if not path:
        pytest.skip('set ADMEM_CASE_ZIP to the uploaded 0123.zip for this optional integration fixture')
    with zipfile.ZipFile(path) as z:
        obj=json.loads(z.read('0123/full_memory.json'))
        packs=[Pack(**json.loads(s)) for s in z.read('0123/input_packs.jsonl').decode().splitlines() if s.strip()]
    full=Full([NS(**s) for s in obj['sessions']],[NS(**r) for r in obj['rounds']])
    assert full.fingerprint==obj['fingerprint']
    chunks=defaultdict(list)
    for window in windows(full,env.counter,1800):
        for piece in window: chunks[piece['rid']].append(piece['text'])
    for rid,r in full.rounds.items():
        expected='\n'.join(f"{m['role']}: {m['content']}" for m in r.messages)
        assert ''.join(chunks[rid])==expected
    context={'key':'c0123','split':'probe','question_type':'multi-session','question_date':'2021/08/20 (Fri) 23:34'}
    env.cfg.attacker_input_tokens=200000  # CharBudget测试，不宣称真实Qwen token数。
    for i,pack in enumerate(packs):
        state=attacker_state(full,tmp_path/'unused.json',context,[],pack,[],env,i)
        payload=json.loads(state['prompt'][1]['content'])
        assert payload['type']=='multi-session'
        assert {r['rid'] for r in payload['rounds']}==set(pack.rids)
        assert all(r['session']==r['rid'].split(':')[0] for r in payload['rounds'])
        assert all('session_id' not in r for r in payload['rounds'])
    assert len(packs)==45 and len(full.rounds)==236
