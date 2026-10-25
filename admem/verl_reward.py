"""标准VERL custom_reward_function接口；不依赖Code-A1定制的代码执行reward manager。"""
from functools import lru_cache

from admem.common import Config, TokenBudget, Unknown, connect, read
from admem.environment import Environment


@lru_cache(maxsize=4)
def environment(config_path):
    cfg = Config.load(config_path)
    return Environment(cfg, TokenBudget(cfg.tokenizer, cfg.tokenizer_revision), connect(cfg.project))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if not extra_info or extra_info.get("split") not in {"train", "val"}:
        raise ValueError("VERL reward requires train/val environment metadata")
    row = read(extra_info["state_path"])
    state = row["state"]
    if state["split"] != extra_info["split"] or data_source != "admem_" + state["role"]:
        raise ValueError("Reward state partition/role mismatch")
    env = environment(extra_info["config"])
    if state.get("environment_fingerprint") != env.cfg.fingerprint():
        raise ValueError("Reward environment configuration mismatch")
    full = env.memory_module.FullMemory.load(state["full_path"])
    if full.fingerprint != state["full_hash"]:
        raise ValueError("Reward full memory changed")
    # 不使用ground_truth字段中的官方答案（导出时它为空）。默认环境失败抛错停止job。
    try:
        detail = env.score(full, state, solution_str)
    except Unknown as exc:
        return _unknown_reward(exc, extra_info)
    # Both fields use the same generic scoring rules as probe. Keep the
    # effective_reward interface for existing VERL exports and callers.
    if "effective_reward" not in detail:
        raise Unknown("Reward detail missing effective_reward")
    return float(detail["effective_reward"])


_UNKNOWN = {"count": 0}


def _unknown_reward(exc, extra_info):
    """ADMEM_REWARD_ON_UNKNOWN=raise（默认，旧行为）：抛错停止job。
    =neutral：返回0.0（两个角色的合法NOOP/空列表都是0，即“什么都不做”的奖励），把失败写入
    ADMEM_REWARD_UNKNOWN_LOG（默认admem_reward_unknown.jsonl），累计超过ADMEM_MAX_UNKNOWN（默认20）
    次仍然抛错。VERL按样本独立评分，无法像train.py那样跳过整组，所以这是带偏差的折中，请监控日志。"""
    import json
    import os
    if os.getenv("ADMEM_REWARD_ON_UNKNOWN", "raise") != "neutral":
        raise exc
    _UNKNOWN["count"] += 1
    with open(os.getenv("ADMEM_REWARD_UNKNOWN_LOG", "admem_reward_unknown.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"state_path": extra_info.get("state_path"), "error": str(exc),
                            "count": _UNKNOWN["count"]}, ensure_ascii=False) + "\n")
    if _UNKNOWN["count"] > int(os.getenv("ADMEM_MAX_UNKNOWN", "20")):
        raise exc
    return 0.0
