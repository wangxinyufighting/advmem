"""v2奖励的纯算术部分（不依赖torch/模型），供environment、train.py与测试共用。

Attacker v2（attacker_reward="v2"）：
- 被gate拒绝的题奖励 ≤ 0，且严格低于空列表（0）。仍按gate阶段保留梯度：越晚失败扣得越少，
  阶段顺序与Environment._gate_once的检查顺序一致：schema → oracle → support → screen → closed-book。
- 整个输出非法时低于任何合法输出的下界：-(reject_penalty + 0.1)。
- 重复题（字面或语义）按最后阶段扣分，不再是0。

Builder v2的组内整形（shape_group）：GRPO用组内均值/标准差归一化。一个非法样本（奖励远低于合法下界）
会把组内std撑大，压扁合法样本之间的差别。整形把非法样本的奖励改成“组内最低合法奖励 − 一个
合法std”，保持它排在所有合法样本之下，但不再主导归一化尺度。
"""
from __future__ import annotations

import math
from statistics import pstdev

# gate阶段 → 进度（0=最早失败，1=通过）。只用于拒绝题的梯度。
REJECTION_PROGRESS = {
    "schema": 0.0,
    "oracle_type": 0.2,      # 题型不符 / 与用户无关 / 题面泄露答案 / multi-session证据只有一个session
    "oracle_answerable": 0.35,
    "unsupported_answer": 0.5,
    "scope_or_completeness": 0.65,
    "closed_book": 0.8,
    "duplicate": 0.8,
}


def rejection_stage(gate):
    reason = gate.get("reason")
    if reason in {"unsupported_answer", "scope_or_completeness", "closed_book"}:
        return reason
    if reason == "oracle_or_type":
        oracle = gate.get("oracle") or {}
        type_ok = oracle.get("type_valid", True) and oracle.get("user_relevant", True) \
            and oracle.get("no_answer_leak", True)
        if type_ok and not oracle.get("answerable", True):
            return "oracle_answerable"
        return "oracle_type"
    return "schema"


def rejection_reward_v2(gate, penalty):
    """被拒绝题的每题奖励，∈[-penalty, 0)。"""
    progress = REJECTION_PROGRESS[rejection_stage(gate)]
    return -penalty * (1.0 - progress)


def duplicate_reward_v2(penalty):
    return -penalty * (1.0 - REJECTION_PROGRESS["duplicate"])


def attacker_invalid_reward_v2(penalty):
    return -(penalty + 0.1)


def novelty_weight_v2(usage, evidence):
    """按E里被问得最多的round衰减；用max而不是mean，往E里塞没用过的round不能把权重拉回来。"""
    return 1.0 / (1.0 + max(usage.get(rid, 0) for rid in evidence))


def builder_invalid_reward_v2(cfg):
    """v2合法奖励的下界：loss、faith、retention各≤1×权重，growth≤size_weight，冗余≤cap个操作。"""
    return -(cfg.builder_loss_weight + cfg.builder_faith_weight + cfg.builder_retention_weight
             + cfg.size_weight + cfg.builder_redundancy_weight * cfg.builder_redundancy_cap + 1.0)


def flip_delta(before_k, after_k, before_2k=None, after_2k=None):
    """一道题的正确性变化。k下没变化时为0；变了时用2k读取复核，取两次变化的平均：
    只是top-k排名被挤动、2k下仍答对的“翻转”只算一半。"""
    delta = float(after_k) - float(before_k)
    if not delta or before_2k is None:
        return delta
    return (delta + float(after_2k) - float(before_2k)) / 2.0


def shape_group(rewards, legal, floor=0.05):
    """返回整形后的组内奖励（见模块说明）。全部合法或全部非法时原样返回。"""
    rewards = [float(r) for r in rewards]
    if len(rewards) != len(legal):
        raise ValueError("rewards/legal length mismatch")
    valid = [r for r, ok in zip(rewards, legal) if ok]
    if not valid or len(valid) == len(rewards):
        return rewards
    spread = pstdev(valid) if len(valid) > 1 else 0.0
    replacement = min(valid) - max(spread, floor)
    return [r if ok else replacement for r, ok in zip(rewards, legal)]


def builder_state_scorable(state):
    """有测试题，或v2下有源窗口可按覆盖率评分（builder_state写入coverage_scorable）。"""
    return bool(state.get("tests")) or bool(state.get("coverage_scorable"))


def normalized_advantages(rewards, std_floor=0.0, eps=1e-6):
    """与learning.advantages相同的公式，外加std下限：std<floor时按floor归一化，
    避免组内奖励只差几个百分点（例如只差长度项）时被放大成整个信号。"""
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("GRPO requires >=2 finite rewards")
    mean = sum(rewards) / len(rewards)
    std = pstdev(rewards)
    scale = max(std, std_floor) + eps
    return [(r - mean) / scale for r in rewards], std
