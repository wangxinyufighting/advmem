#!/usr/bin/env python3
"""把LoRA adapter折进底座权重，得到可直接给vLLM/validation使用的完整HF模型（TRAINING.md 第6、11节）。

verl的 `python -m verl.model_merger merge --backend fsdp` 在LoRA训练下写出的是*未合并*的底座权重
（<dir>.parts）加 <dir>.parts/lora_adapter。直接服务 <dir>.parts 评估的是没训练过的底座。
并且它写的adapter_config.json里lora_alpha恒为0，所以这里必须用 --alpha 给出训练时的alpha
（与verl_launch的 --lora-alpha 相同，默认2×rank）。

  python scripts/merge_lora.py --base runs/x/merged/step_00020.parts \\
      --adapter runs/x/merged/step_00020.parts/lora_adapter --alpha 128 --out runs/x/merged/step_00020

在CPU上用float32合并（避免bf16里小增量被舍入掉），再转成 --dtype（默认bfloat16）保存。
输出先写到 <out>.tmp，完成并自检后再改名，所以 <out>/config.json 存在即代表合并完整。
自检：adapter里每个lora_A/lora_B都必须被PEFT加载（否则键名不匹配，合并会静默地什么也不做），
合并后至少有一个权重相对底座发生变化。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def prepare_adapter(adapter, work, alpha=None, rank=None):
    """把adapter复制到work并按训练时的alpha改写配置（不改原目录）；返回(work, config)。"""
    adapter = Path(adapter)
    config_path, weights = adapter / "adapter_config.json", adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise FileNotFoundError(f"{adapter} must contain adapter_config.json and adapter_model.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if rank is not None and int(config["r"]) != int(rank):
        raise ValueError(f"adapter rank is {config['r']} but --rank {rank} was given")
    saved = float(config.get("lora_alpha") or 0)
    if alpha is None:
        if saved <= 0:
            raise ValueError("adapter_config.json has lora_alpha=0 (verl's merger does not record it); "
                             "pass --alpha, the lora_alpha used in training (default 2 x rank)")
        alpha = saved
    elif saved > 0 and abs(saved - float(alpha)) > 1e-9:
        raise ValueError(f"adapter_config.json says lora_alpha={saved} but --alpha {alpha} was given")
    config["lora_alpha"] = alpha
    config["lora_dropout"] = 0.0
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(weights, work / "adapter_model.safetensors")
    return work, config


def merge(base, adapter, out, alpha=None, rank=None, dtype="bfloat16", allow_untrained=False):
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, out = Path(base), Path(out)
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"{base} is not a Hugging Face model directory (no config.json)")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    adapter_dir, config = prepare_adapter(adapter, tmp / ".adapter", alpha, rank)
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        keys = [k for k in f.keys() if "lora_" in k]
        b_norm = sum(float(f.get_tensor(k).float().abs().sum()) for k in keys if "lora_B" in k)
    n_pairs = sum(1 for k in keys if "lora_B" in k)
    if n_pairs == 0 or n_pairs * 2 != len(keys):
        raise ValueError(f"unexpected adapter tensors in {adapter}: {len(keys)} lora tensors, {n_pairs} lora_B")
    if b_norm == 0 and not allow_untrained:
        raise ValueError(f"every lora_B in {adapter} is zero: the adapter is untrained, merging would return "
                         "the base model (pass --allow-untrained to do it anyway)")

    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float32, device_map={"": "cpu"})
    probe = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("proj.weight")}
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    loaded = [m for m in peft_model.modules() if "default" in getattr(m, "lora_B", {})]
    if len(loaded) != n_pairs:
        raise RuntimeError(f"adapter has {n_pairs} LoRA modules but {len(loaded)} matched the base model "
                           "(key names differ?); refusing to write a merged model that ignores the adapter")
    merged = peft_model.merge_and_unload()
    if b_norm > 0 and not any(n in probe and not torch.equal(p.detach(), probe[n])
                              for n, p in merged.named_parameters()):
        raise RuntimeError("merge_and_unload left the base weights unchanged")
    merged.to(getattr(torch, dtype)).save_pretrained(str(tmp), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(tmp))
    shutil.rmtree(tmp / ".adapter")
    (tmp / "merge_lora.json").write_text(json.dumps(
        {"base": str(base), "adapter": str(adapter), "rank": config["r"], "alpha": config["lora_alpha"],
         "target_modules": config.get("target_modules"), "lora_modules": n_pairs}, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="底座HF目录（verl merger输出的<dir>.parts）")
    p.add_argument("--adapter", required=True, help="含adapter_config.json/adapter_model.safetensors的目录")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, help="训练时的lora_alpha（verl写出的adapter_config里为0，必须给）")
    p.add_argument("--rank", type=int, help="可选：校验adapter的秩")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="保存精度")
    p.add_argument("--allow-untrained", action="store_true", help="允许合并全零lora_B（即未训练的adapter）")
    a = p.parse_args(argv)
    print(f"[merge_lora] wrote {merge(a.base, a.adapter, a.out, a.alpha, a.rank, a.dtype, a.allow_untrained)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
