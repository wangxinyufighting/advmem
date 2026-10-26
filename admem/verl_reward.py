"""标准VERL custom_reward_function接口；不依赖Code-A1定制的代码执行reward manager。"""
import os
from functools import lru_cache

from admem.common import Config, TokenBudget, Unknown, connect, read
from admem.environment import Environment


@lru_cache(maxsize=4)
def environment(config_path):
    cfg = Config.load(config_path)
    return Environment(cfg, TokenBudget(cfg.tokenizer, cfg.tokenizer_revision), connect(cfg.project))


# 作为reward_extra_info一起返回、由verl汇总进训练日志的数值字段。每个角色字段固定，缺失补0。
_EXTRA_FIELDS = {
    "builder": ("legal", "noop", "faith_ratio", "gain", "loss", "information_loss", "growth_ratio"),
    "attacker": ("legal", "accepted", "rejected", "duplicates", "over_budget"),
}


def _details(role, detail):
    if role == "attacker":
        items = detail.get("items") or []
        status = [item.get("status") for item in items]
        values = {"legal": float(bool(detail.get("legal"))), "accepted": float(status.count("accepted")),
                  "rejected": float(status.count("rejected")), "duplicates": float(status.count("duplicate")),
                  "over_budget": float(status.count("over_budget"))}
    else:
        values = {key: float(detail.get(key) or 0.0) for key in _EXTRA_FIELDS["builder"]}
        values["legal"] = float(bool(detail.get("legal")))
        for name, value in (detail.get("reward_components") or {}).items():
            values["component_" + name] = float(value)
    return values


def _result(role, score, detail):
    """ADMEM_VERL_RETURN_DICT=1（默认）返回{"score", ...诊断字段}，verl会把诊断字段记进日志；=0只返回float。"""
    if os.getenv("ADMEM_VERL_RETURN_DICT", "1") != "1":
        return float(score)
    values = {key: 0.0 for key in _EXTRA_FIELDS[role]}
    if role == "builder":
        values.update({"component_" + k: 0.0 for k in ("gain", "loss", "faith", "retention", "coverage",
                                                         "compression", "growth", "redundancy")})
    values.update(_details(role, detail or {}))
    return {"score": float(score), **values}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    role = str(data_source).removeprefix("admem_")
    detail = _compute(data_source, solution_str, extra_info)
    if detail is None:  # 环境失败且ADMEM_REWARD_ON_UNKNOWN=neutral
        return _result(role, 0.0, {"legal": True})
    return _result(role, detail["effective_reward"], detail)


def _compute(data_source, solution_str, extra_info=None):
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
        _unknown_reward(exc, extra_info)
        return None
    # Both fields use the same generic scoring rules as probe. Keep the
    # effective_reward interface for existing VERL exports and callers.
    if "effective_reward" not in detail:
        raise Unknown("Reward detail missing effective_reward")
    return detail


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
