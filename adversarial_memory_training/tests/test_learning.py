"""真实PyTorch前向/反向/参数更新的小模型测试；不代表Qwen4B效果验证。"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from admem.learning import Rollout, advantages, completion_logp, grpo_loss, optimize_groups
from admem.common import write_rows
from admem.train import load_training_records
from admem import verl_reward


class TinyLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.embedding=torch.nn.Embedding(7,5)
        self.base=torch.nn.Linear(5,7,bias=False)
        for p in list(self.embedding.parameters())+list(self.base.parameters()): p.requires_grad_(False)
        self.a=torch.nn.Parameter(torch.randn(5,2)*.05)
        self.b=torch.nn.Parameter(torch.zeros(2,7))
        self.active=True
    def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=0):
        h=self.embedding(input_ids)
        out=self.base(h)
        if self.active: out=out+h@self.a@self.b
        return NS(logits=out[:,-logits_to_keep:] if logits_to_keep else out)
    @contextmanager
    def disable_adapter(self):
        old=self.active;self.active=False
        try: yield
        finally: self.active=old


def rollout(model,token,reward):
    with torch.no_grad():
        old=completion_logp(model,[1],[token],'cpu').clone()
        with model.disable_adapter(): ref=completion_logp(model,[1],[token],'cpu').clone()
    return Rollout([1],[token],str(token),old,ref,reward,{})


def test_logprob_is_only_completion_and_aligns_next_token():
    m=TinyLoRA()
    actual=completion_logp(m,[1,2],[3,4],'cpu')
    logits=m(torch.tensor([[1,2,3]])).logits[0]
    expected=torch.log_softmax(logits.float(),-1)[[1,2],[3,4]]
    assert actual.shape==(2,)
    assert torch.allclose(actual,expected)


def test_group_advantage_zero_and_finite():
    a,s=advantages([1,1,1,1])
    assert s==0 and torch.equal(a,torch.zeros(4))
    a,s=advantages([0,1])
    assert a.tolist()==pytest.approx([-1,1],abs=1e-5)
    with pytest.raises(ValueError): advantages([0,float('nan')])


def test_clipped_positive_advantage_has_no_gradient_past_clip():
    logp=torch.tensor([.7],requires_grad=True)
    loss=grpo_loss(logp,torch.zeros(1),torch.zeros(1),torch.tensor(1.),clip=.2,beta=0)
    loss.backward()
    assert loss.item()==pytest.approx(-1.2)
    assert logp.grad.item()==0


def test_clipped_negative_advantage_has_no_gradient_below_clip():
    logp=torch.tensor([-.7],requires_grad=True)
    loss=grpo_loss(logp,torch.zeros(1),torch.zeros(1),torch.tensor(-1.),clip=.2,beta=0)
    loss.backward()
    assert loss.item()==pytest.approx(.8)
    assert logp.grad.item()==0


def test_kl_zero_at_reference_positive_away():
    p=torch.tensor([-.5],requires_grad=True)
    assert grpo_loss(p,p.detach(),p.detach(),torch.tensor(0.)).item()==0
    loss=grpo_loss(p,p.detach(),torch.tensor([-1.0]),torch.tensor(0.),beta=1.)
    assert loss.item()>0
    loss.backward()
    assert p.grad is not None


def test_grpo_really_updates_lora_and_not_base():
    m=TinyLoRA();before={n:p.clone() for n,p in m.named_parameters()}
    group=[rollout(m,2,1),rollout(m,3,0)]
    optimizer=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.8)
    olddiff=(completion_logp(m,[1],[2],'cpu')-completion_logp(m,[1],[3],'cpu')).item()
    loss=optimize_groups(m,optimizer,[group],'cpu',beta=.01)
    newdiff=(completion_logp(m,[1],[2],'cpu')-completion_logp(m,[1],[3],'cpu')).item()
    assert newdiff>olddiff and not torch.equal(before['b'],m.b)
    assert torch.equal(before['base.weight'],m.base.weight)
    assert torch.equal(before['embedding.weight'],m.embedding.weight)
    assert isinstance(loss,float)


def test_two_identical_groups_normalize_like_one():
    m1=TinyLoRA();m2=deepcopy(m1)
    g=[rollout(m1,2,1),rollout(m1,3,0)]
    o1=torch.optim.SGD([p for p in m1.parameters() if p.requires_grad],lr=.1)
    o2=torch.optim.SGD([p for p in m2.parameters() if p.requires_grad],lr=.1)
    optimize_groups(m1,o1,[g],'cpu',beta=.02)
    optimize_groups(m2,o2,[g,g],'cpu',beta=.02)
    assert torch.allclose(m1.b,m2.b,atol=1e-7)


def test_sft_real_gradient_reduces_completion_loss():
    m=TinyLoRA();optimizer=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.5)
    before=-completion_logp(m,[1],[2,0],'cpu').mean()
    optimizer.zero_grad();before.backward();optimizer.step()
    after=-completion_logp(m,[1],[2,0],'cpu').mean()
    assert after.item()<before.item()


def test_separate_policies_do_not_share_trainable_weights():
    builder=TinyLoRA();attacker=deepcopy(builder)
    snapshot=attacker.b.clone()
    group=[rollout(builder,2,1),rollout(builder,3,0)]
    optimizer=torch.optim.SGD([p for p in builder.parameters() if p.requires_grad],lr=.8)
    optimize_groups(builder,optimizer,[group],'cpu')
    assert torch.equal(attacker.b,snapshot)
    assert not torch.equal(builder.b,attacker.b)


def test_nonfinite_loss_rejected():
    with pytest.raises(FloatingPointError):
        grpo_loss(torch.tensor([1000.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor(-1.))


@pytest.mark.parametrize('partition',['val','test','probe'])
def test_training_rejects_nontrain_states(tmp_path,partition):
    p=tmp_path/'states.jsonl'
    write_rows(p,[{'prompt':[], 'state':{'role':'builder','split':partition,'prompt':[],'tests':[{}]}}])
    with pytest.raises(ValueError): load_training_records(p,'builder','grpo')


def test_sft_and_grpo_filter_different_states(tmp_path):
    p=tmp_path/'states.jsonl'
    base={'id':'a','prompt':[], 'state':{'role':'builder','split':'train','prompt':[],'tests':[]},'completion':'{"ops":[]}'}
    write_rows(p,[base,{**base,'id':'b','state':{**base['state'],'tests':[{}]},'completion':None}])
    assert [x['id'] for x in load_training_records(p,'builder','sft')]==['a']
    assert [x['id'] for x in load_training_records(p,'builder','grpo')]==['b']


def test_verl_reward_uses_effective_reward(monkeypatch,tmp_path):
    """The distributed reward hook must apply gates, not the diagnostic score."""
    state_path=tmp_path/'state.json'
    state={'split':'train','role':'builder','full_path':'unused.json',
           'full_hash':'full-hash','environment_fingerprint':'cfg-hash'}
    from admem.common import write
    write(state_path, {'state':state})

    class FakeCfg:
        def fingerprint(self):
            return 'cfg-hash'

    class FakeFull:
        fingerprint='full-hash'

        @classmethod
        def load(cls,path):
            assert path == 'unused.json'
            return cls()

    fake_env=NS(
        cfg=FakeCfg(),
        memory_module=NS(FullMemory=FakeFull),
        score=lambda full, current_state, completion: {
            'reward': 1.25, 'effective_reward': 0.0,
        },
    )
    monkeypatch.setattr(verl_reward, 'environment', lambda path: fake_env)
    score=verl_reward.compute_score(
        'admem_builder', '{}', ground_truth='',
        extra_info={'split':'train','state_path':str(state_path),
                    'config':'config.json'},
    )
    assert score == 0.0
