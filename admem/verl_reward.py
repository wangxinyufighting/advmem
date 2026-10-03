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
    # 不使用ground_truth字段中的官方答案（导出时它为空）。环境失败抛错停止job，绝不return0。
    detail = env.score(full, state, solution_str)
    # ``reward`` is the diagnostic/base score.  The effective score applies
    # family constraints (e.g. refine must shorten and noop must be empty),
    # and is the only score suitable for policy optimization.
    if "effective_reward" not in detail:
        raise Unknown("Reward detail missing effective_reward")
    return float(detail["effective_reward"])
