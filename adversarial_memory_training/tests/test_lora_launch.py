"""LoRA启动/合并、角色检查与环境模板的离线测试（真实合并用一个很小的随机Qwen2，CPU即可）。"""
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from admem.verl_launch import LaunchOptions, effective_lora_alpha, verl_env, verl_overrides

ROOT = Path(__file__).resolve().parents[2]
V2 = str(ROOT / "configs" / "rl_hidden_4b_v2.json")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(**kw):
    return LaunchOptions(role="builder", config=V2, train="t.parquet", model="/m", out="/o", **kw)


# ---------------------------------------------------------------- launcher

def test_full_finetune_has_no_lora_or_offload_keys():
    joined = "\n".join(verl_overrides(options()))
    assert "lora" not in joined and "offload" not in joined and "load_format" not in joined


def test_lora_overrides_set_rank_alpha_and_load_the_base_weights():
    keys = verl_overrides(options(lora_rank=64))
    assert "actor_rollout_ref.model.lora_rank=64" in keys
    assert "actor_rollout_ref.model.lora_alpha=128" in keys            # 默认2×rank
    assert "actor_rollout_ref.model.target_modules=all-linear" in keys
    # 不能保持默认的dummy：LoRA只同步adapter，底座权重要vLLM自己加载。
    assert "actor_rollout_ref.rollout.load_format=safetensors" in keys
    assert "actor_rollout_ref.model.lora_alpha=32" in verl_overrides(options(lora_rank=16, lora_alpha=32))
    assert effective_lora_alpha(options(lora_rank=16, lora_alpha=32)) == 32
    assert effective_lora_alpha(options()) == 0


def test_lora_alpha_without_rank_and_negative_values_are_rejected():
    with pytest.raises(ValueError):
        verl_overrides(options(lora_alpha=8))
    with pytest.raises(ValueError):
        verl_overrides(options(lora_rank=-1))


def test_offload_parks_actor_optimizer_and_ref_on_cpu():
    keys = verl_overrides(options(offload=True))
    for key in ("actor.fsdp_config.param_offload", "actor.fsdp_config.optimizer_offload",
                "ref.fsdp_config.param_offload"):
        assert f"actor_rollout_ref.{key}=True" in keys


def test_launcher_raises_api_call_cap_unless_user_set_it():
    assert verl_env(options(), base={})["MAX_API_CALLS"] == "100000000"
    assert verl_env(options(), base={"MAX_API_CALLS": "5"})["MAX_API_CALLS"] == "5"


def test_train_verl_sh_forwards_lora_flags():
    out = subprocess.run(["bash", str(ROOT / "scripts" / "train_verl.sh")], capture_output=True, text=True,
                         env={"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "MODEL": "/m", "TRAIN": "t", "OUT": "/tmp/o",
                              "CONFIG": V2, "DRY_RUN": "1", "LORA_RANK": "64", "ROLLOUT_GPU_UTIL": "0.5",
                              "GPUS": "1", "LR": "1e-5"})
    text = out.stdout + out.stderr
    if "No module named" in text:
        pytest.skip("python on the minimal PATH lacks dependencies")
    assert "actor_rollout_ref.model.lora_rank=64" in text and "gpu_memory_utilization=0.5" in text, text


# ---------------------------------------------------------------- rl_loop.merge

def loop_for(tmp_path, *extra):
    module = load_script("rl_loop")
    args = module.parser().parse_args(["--role", "builder", "--config", V2, "--train", "t", "--model", "base",
                                       "--prepared", "p", "--out", str(tmp_path / "rl"), *extra])
    return module, module.Loop(args), args


def fake_subprocess(monkeypatch, module, calls, lora_written=True):
    def run(command, **kwargs):
        calls.append([str(c) for c in command])
        if "verl.model_merger" in command:
            target = Path(command[command.index("--target_dir") + 1])
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}")
            if lora_written:
                (target / "lora_adapter").mkdir()
                (target / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"x")
        else:  # merge_lora.py
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "config.json").write_text("{}")
    monkeypatch.setattr(module.subprocess, "run", run)


def test_rl_loop_folds_the_lora_adapter_before_validation(tmp_path, monkeypatch):
    module, loop, args = loop_for(tmp_path, "--lora-rank", "64")
    (loop.out / "verl" / "global_step_20" / "actor").mkdir(parents=True)
    calls = []
    fake_subprocess(monkeypatch, module, calls)
    merged = loop.merge(20)
    assert merged == loop.out / "merged" / "step_00020" and (merged / "config.json").exists()
    assert not (loop.out / "merged" / "step_00020.parts").exists()          # 中间产物已清理
    verl_call, lora_call = calls
    assert verl_call[verl_call.index("--target_dir") + 1].endswith("step_00020.parts")
    assert lora_call[1].endswith("merge_lora.py")
    assert lora_call[lora_call.index("--base") + 1].endswith("step_00020.parts")
    assert lora_call[lora_call.index("--adapter") + 1].endswith("step_00020.parts/lora_adapter")
    assert lora_call[lora_call.index("--alpha") + 1] == "128" and lora_call[lora_call.index("--rank") + 1] == "64"
    assert lora_call[lora_call.index("--out") + 1] == str(merged)


def test_rl_loop_refuses_lora_run_whose_checkpoint_has_no_adapter(tmp_path, monkeypatch):
    module, loop, _ = loop_for(tmp_path, "--lora-rank", "64")
    (loop.out / "verl" / "global_step_20" / "actor").mkdir(parents=True)
    fake_subprocess(monkeypatch, module, [], lora_written=False)
    with pytest.raises(RuntimeError, match="lora_adapter"):
        loop.merge(20)
    assert not (loop.out / "merged" / "step_00020").exists()


def test_rl_loop_full_finetune_merge_is_one_step_and_atomic(tmp_path, monkeypatch):
    module, loop, _ = loop_for(tmp_path)
    (loop.out / "verl" / "global_step_20" / "actor").mkdir(parents=True)
    calls = []
    fake_subprocess(monkeypatch, module, calls)
    merged = loop.merge(20)
    assert len(calls) == 1 and (merged / "config.json").exists()
    assert loop.merge(20) == merged and len(calls) == 1                     # 已合并则不重复


def test_rl_loop_dry_run_prints_both_merge_commands(tmp_path, capsys):
    _, loop, _ = loop_for(tmp_path, "--lora-rank", "8", "--dry-run")
    loop.merge(20)
    out = capsys.readouterr().out
    assert "verl.model_merger" in out and "merge_lora.py" in out and "--alpha 16" in out


# ---------------------------------------------------------------- merge_lora (real, tiny model)

def tiny_model_and_adapter(tmp_path, rank=4, zero_b=False):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=2)
    transformers.Qwen2ForCausalLM(config).save_pretrained(tmp_path / "base")
    tensors = {}
    for layer in range(2):
        for module, name, shape in (("self_attn", "q_proj", (16, 16)), ("mlp", "up_proj", (32, 16))):
            key = f"base_model.model.model.layers.{layer}.{module}.{name}"
            tensors[key + ".lora_A.weight"] = torch.randn(rank, shape[1])
            tensors[key + ".lora_B.weight"] = torch.zeros(shape[0], rank) if zero_b else torch.randn(shape[0], rank)
    (tmp_path / "adapter").mkdir()
    save_file(tensors, str(tmp_path / "adapter" / "adapter_model.safetensors"))
    # verl的merger写的config：lora_alpha恒为0
    (tmp_path / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": None, "r": rank, "lora_alpha": 0,
         "target_modules": ["q_proj", "up_proj"]}))
    return torch, tensors


def test_merge_lora_applies_alpha_over_rank_scaled_delta(tmp_path):
    torch, tensors = tiny_model_and_adapter(tmp_path)
    from transformers import Qwen2ForCausalLM
    merge_lora = load_script("merge_lora")
    with pytest.raises(ValueError, match="lora_alpha=0"):
        merge_lora.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")      # alpha必须显式给出
    out = merge_lora.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out", alpha=8, rank=4,
                           dtype="float32")
    assert (out / "config.json").exists() and not (tmp_path / "out.tmp").exists()
    base = Qwen2ForCausalLM.from_pretrained(tmp_path / "base", dtype=torch.float32).state_dict()
    new = Qwen2ForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
    prefix = "base_model.model.model.layers.1.mlp.up_proj"
    expected = (8 / 4) * tensors[prefix + ".lora_B.weight"] @ tensors[prefix + ".lora_A.weight"]
    assert torch.allclose(new["model.layers.1.mlp.up_proj.weight"] - base["model.layers.1.mlp.up_proj.weight"],
                          expected, atol=1e-5)
    assert torch.equal(new["model.layers.0.self_attn.k_proj.weight"], base["model.layers.0.self_attn.k_proj.weight"])


def test_merge_lora_rejects_rank_mismatch_and_untrained_adapter(tmp_path):
    tiny_model_and_adapter(tmp_path, zero_b=True)
    merge_lora = load_script("merge_lora")
    with pytest.raises(ValueError, match="--rank 8"):
        merge_lora.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out", alpha=8, rank=8)
    with pytest.raises(ValueError, match="untrained"):
        merge_lora.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out", alpha=8, rank=4)
    assert not (tmp_path / "out").exists()


def test_merge_lora_refuses_adapter_whose_keys_do_not_match_the_model(tmp_path):
    torch, tensors = tiny_model_and_adapter(tmp_path)
    from safetensors.torch import save_file
    save_file({k.replace("model.model.layers", "model.model.decoder.layers"): v for k, v in tensors.items()},
              str(tmp_path / "adapter" / "adapter_model.safetensors"))
    with pytest.raises(Exception):
        load_script("merge_lora").merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out", alpha=8)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- check_roles / env template

class Chat(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Chat.seen.append(body)
        if body["model"] == "bad-model":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
            return
        data = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Chat)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    Chat.seen = []
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_check_roles_reports_ok_and_failed_with_effective_request(server, monkeypatch, capsys):
    check = load_script("check_roles")
    for key in ("LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("JUDGE_MODEL", "good")
    monkeypatch.setenv("JUDGE_BASE_URL", server)
    monkeypatch.setenv("JUDGE_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    monkeypatch.setenv("DEFENDER_MODEL", "bad-model")
    monkeypatch.setenv("DEFENDER_BASE_URL", server)
    monkeypatch.setenv("API_RETRIES", "1")
    monkeypatch.setenv("ATTACKER_MODEL", "")
    code = check.main(["--config", V2, "--roles", "JUDGE", "DEFENDER", "ATTACKER"])
    out = capsys.readouterr().out
    assert code == 1
    assert "[OK] JUDGE" in out and "[FAILED] DEFENDER" in out and "[FAILED] ATTACKER" in out
    assert "HTTP 400" in out and "1/3 roles OK" in out
    good = next(b for b in Chat.seen if b["model"] == "good")
    assert good["thinking"] == {"type": "disabled"} and good["response_format"] == {"type": "json_object"}
    assert "messages" not in out.split("JUDGE", 1)[1].split("DEFENDER")[0].split("request:")[1]
    assert "sk-" not in out


def test_check_roles_warns_about_settings_that_break_long_runs(monkeypatch):
    check = load_script("check_roles")
    from admem.common import Config
    for key in ("MAX_API_CALLS", "API_TIMEOUT", "API_RETRIES", "EMBED_DEVICE", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    text = "\n".join(check.warnings(Config.load(V2)))
    for word in ("MAX_API_CALLS", "API_TIMEOUT", "API_RETRIES", "EMBED_DEVICE", "CUDA_VISIBLE_DEVICES"):
        assert word in text


def test_env_template_sets_roles_and_needs_a_key():
    script = ROOT / "examples" / "env_2gpu_api.sh"
    show = 'echo "$JUDGE_MODEL|$JUDGE_API_KEY|$VAL_ATTACKER_MODEL|$DEFENDER_MODEL|$BUILDER_BASE_URL|' \
           '$MAX_API_CALLS|$EMBED_DEVICE|$API_TIMEOUT|$JUDGE_EXTRA_BODY"'
    ok = subprocess.run(["bash", "-c", f"source {script} && {show}"], capture_output=True, text=True,
                        env={"PATH": "/usr/bin:/bin", "DEEPSEEK_API_KEY": "sk-x"})
    fields = ok.stdout.strip().split("|")
    assert fields[:8] == ["deepseek-flash", "sk-x", "deepseek-v4-pro", "deepseek-flash", "http://localhost:8001/v1",
                          "100000000", "cpu", "600"]
    assert json.loads(fields[8]) == {"thinking": {"type": "disabled"}}
    missing = subprocess.run(["bash", "-c", f"source {script}; echo after"], capture_output=True, text=True,
                             env={"PATH": "/usr/bin:/bin"})
    assert "DEEPSEEK_API_KEY" in missing.stderr
