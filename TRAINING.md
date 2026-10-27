# Training the Builder and Attacker with GRPO (verl)

This guide covers the whole path from LongMemEval to a selected Builder/Attacker checkpoint:
data preparation → optional SFT warm start → GRPO in verl with periodic validation and early
stopping → one final test evaluation. Everything below runs from the repository root.

## 1. Is "add verl" enough?

Not on its own. The repository already had the right reward interface (`admem/verl_reward.py`
exposes `compute_score(data_source, solution_str, ground_truth, extra_info)` and `admem.cli export
--parquet` writes verl-format rows), but three things were missing or broken:

| Problem | What was added |
| --- | --- |
| Current verl (0.10.x, `main_ppo` with the reward loop) reads the reward function from `reward.custom_reward_function.*`. The old `scripts/train_verl.sh` passed the top-level `custom_reward_function.*`, which `main_ppo` no longer migrates, so the admem reward was silently not loaded and verl's default scorer would fail on `data_source="admem_builder"`. | `admem/verl_launch.py` builds the verl command for both key layouts (`--reward-keys new` default, `legacy` for older verl). `scripts/train_verl.sh` is now a thin wrapper around it. |
| GRPO's per-group std normalisation lets one invalid sample (Builder v2 ≈ −8.2) flatten the differences between legal samples (often ≈0.05). verl scores each sample independently, so the reward function cannot fix this. | `admem/verl_adv.py` registers a verl advantage estimator `admem_grpo`: invalid samples are reshaped to "lowest legal reward − legal std" inside each group, then normalised with `max(std, ADMEM_STD_FLOOR)`. |
| No training data pipeline for the Attacker, no validation, no stopping rule. | `scripts/prepare_rl_data.sh` (all states and validation data), `bank --collect` (Attacker states), `admem/validation.py` (checkpoint metrics + `EarlyStopper`), `scripts/rl_loop.py` (train → merge → serve → validate → decide). |

The launcher's override keys were checked against verl commit `3c6d9b4` (0.10.0.dev, 2026-10-23).
verl moves fast: with another version run `--dry-run` once and compare the keys with
`verl/trainer/config/_generated_ppo_trainer.yaml` of that version.

## 2. Environments and hardware

Use one Python environment for verl + admem (the reward function runs inside verl's Ray workers and
imports `admem`):

```bash
conda create -n admem-rl python=3.11 -y && conda activate admem-rl
git clone https://github.com/volcengine/verl.git && (cd verl && git checkout 3c6d9b4 && pip install -e .)
pip install -r requirements-train.txt
```

The judge and reader models are **separate services** (vLLM or any OpenAI-compatible API). They must
not be the model being trained; `admem.cli` and the probe runner warn when they are
(`self_judging` in the summaries).

### 2.1 Layout: 2×80 GB, local frozen reader, API judge

Qwen3-4B-Instruct-2507 is the backbone of the Builder, the Attacker and the `DEFENDER` (the answer agent
that reads memory). The Builder and Attacker are trained, never at the same time; the Defender is frozen.
The judge (`JUDGE`) and the one-time validation question writer (`VAL_ATTACKER`) are an API (DeepSeek
here; any OpenAI-compatible API works).

| | GPU 0 | GPU 1 | API |
| --- | --- | --- | --- |
| always | `DEFENDER`: vLLM, original Qwen3-4B weights, `--gpu-memory-utilization 0.85` | — | — |
| data preparation (§4), alternation data steps (§9) | Defender | vLLM serving the current Builder/Attacker | JUDGE, VAL_ATTACKER |
| training chunk (§6, §8) | Defender (reward) | verl: actor + rollout engine (`--gpus 1`, LoRA) | JUDGE (reward) |
| merge | — | — (CPU) | — |
| validation (§7) | Defender | vLLM serving the checkpoint as `BUILDER_EVAL`/`ATTACKER_EVAL` | JUDGE |

Why this split and not "Defender + policy server on one GPU, training on the other":

* **The trained model never needs a server during training.** verl generates the rollouts with its own
  vLLM engine on the training GPU. A served copy of the Builder/Attacker is only needed for data
  preparation and validation, which `rl_loop.py` and `alternate.py` run *between* training chunks, when
  GPU 1 is free. A policy slot on GPU 0 would sit idle most of the time.
* **The Defender is the busiest local model.** verl scores the 64 samples of a step concurrently, and
  every sample asks the reader several questions with up to `reader_tokens` (12000) tokens of context.
  Qwen3-4B needs ~144 KiB of KV cache per token: at 0.85 of an 80 GB GPU the Defender has room for roughly
  400k tokens (~30 full-length reads at once), at 0.4 only ~150k (~11). A 40/40 split works, but the
  reader queue then sets the step time.

**Start the Defender once and leave it running** for the whole project:

```bash
CUDA_VISIBLE_DEVICES=0 NAME=qwen3-4b-defender MODEL=/models/Qwen3-4B-Instruct-2507 PORT=8003 \
  GPU_MEMORY_UTILIZATION=0.85 bash scripts/serve_policy.sh
```

(The default `MAX_MODEL_LEN` 24576 covers `reader_tokens` plus the 4096-token output budget; vLLM rejects
requests whose prompt plus `max_tokens` exceed it.)

* It always serves the **original** weights under one name. Never point it at a trained checkpoint: the
  reward, the bank's validity check and every validation score are defined by this reader, and API
  responses are cached by served name.
* It has its own name and port, so the `self_judging` check (which compares model name + URL of the
  policy roles with JUDGE/DEFENDER) does not fire. Sharing the backbone is intended here: the Builder
  learns to write memory that *this* reader can use, which is what you want if the deployed answer agent
  is the same Qwen3-4B. Results then describe this reader, not readers in general. The judge, where
  self-grading would bias the reward, stays on a different model.

**Everything that runs on GPU 1 is started with `CUDA_VISIBLE_DEVICES=1`.** `rl_loop.py` and
`alternate.py` pass it on to verl and to the vLLM servers they start, so `--serve-gpus` is not needed:

```bash
export CUDA_VISIBLE_DEVICES=1        # in the shell that runs §4–§10; the Defender runs in its own shell
```

Embeddings (bge-m3) run on the CPU (`EMBED_DEVICE=cpu`) in every stage, so data preparation, the verl
reward workers and validation retrieve identically and nothing competes with vLLM or FSDP for GPU memory.

**Training on one GPU: use LoRA.** Full fine-tuning of a 4B model keeps fp32 master weights, gradients
and Adam states (~16 bytes/parameter ≈ 64 GB). On one 80 GB GPU that leaves too little for the rollout
engine and the activations and logits of ~20k-token sequences (Builder prompts are up to
`builder_input_tokens` = 16384 tokens); `--offload` only helps outside the optimizer step. LoRA fits:

```text
--gpus 1 --lora-rank 64 --lr 1e-5 --rollout-gpu-util 0.5
```

verl uses the base model without the adapter as the KL reference (no second copy), recommends rank ≥ 32
and about 10× the full fine-tuning learning rate. verl's checkpoint merger writes the *base* weights plus
a separate `lora_adapter/`; `rl_loop.py` folds the adapter into full weights (`scripts/merge_lora.py`)
before serving, so validation never scores the untrained base. If a rollout runs out of memory, lower
`--rollout-gpu-util` or `--response-tokens` (Attacker: 2048 is enough).

**Check the reader's ceiling before training.** With `require_reader_answerable: true` (on in the `rl_*`
configs) a question counts only if the Defender answers it from the raw evidence. A 4B reader fails more
multi-session and temporal-reasoning questions, so those types yield fewer bank questions and less
training signal. That is correct (the Builder cannot fix the reader), but measure it once the
`builder_states` stage has run: `probe_breakdown.py --ceiling` (§4) reports `raw(E) only` accuracy per
question type, and `bank_log.json` lists the `reader_unanswerable` rejections. A type near zero gives the
Builder almost nothing to learn from.

### 2.2 API roles

```bash
export DEEPSEEK_API_KEY=sk-...                 # never commit it; .env is git-ignored
source examples/env_2gpu_api.sh                # API roles, timeouts, MAX_API_CALLS, EMBED_DEVICE=cpu
# local Defender instead of the template's API Defender:
export DEFENDER_MODEL=qwen3-4b-defender DEFENDER_BASE_URL=http://localhost:8003/v1 DEFENDER_API_KEY=EMPTY
unset DEFENDER_EXTRA_BODY                      # the "thinking" field is for DeepSeek only
export CUDA_VISIBLE_DEVICES=1
python scripts/check_roles.py --config configs/rl_hidden_4b_v2.json
```

`check_roles.py` sends one short JSON-mode request per role through the same code path as the reward
(`*_EXTRA_BODY`, JSON mode, token budget) and prints the status, latency and effective request (without
messages or keys). A wrong model id, key, port or unsupported field shows up here as `FAILED: ...`, not
hours into training. It also warns about settings that break long runs. `BUILDER`/`ATTACKER` report
`FAILED` until the policy server of §3 is up.

What the template sets for the API roles, and why:

| Setting | Value | Why |
| --- | --- | --- |
| `JUDGE_MODEL` | `deepseek-flash` | cheap non-thinking model for short JSON verdicts. DeepSeek renamed its models in 2026 (`deepseek-chat`/`deepseek-reasoner` were retired); check the ids on [api-docs.deepseek.com](https://api-docs.deepseek.com) |
| `VAL_ATTACKER_MODEL` | `deepseek-v4-pro` | writes the frozen val bank once; a model other than the 4B backbone, so the validation questions are not written by the reader that answers them |
| `*_BASE_URL` | `https://api.deepseek.com` | the client appends `/chat/completions` |
| `*_EXTRA_BODY` | `{"thinking": {"type": "disabled"}}` | no reasoning tokens for verdicts; delete it if `check_roles.py` reports the field as rejected |
| `LLM_JSON_MODE` | `1` (default) | `response_format=json_object`; DeepSeek requires the word "json" in the prompt, and admem adds a one-line system message when a prompt lacks it |
| `API_TIMEOUT`, `API_RETRIES`, `API_BACKOFF` | 600, 6, 2 | a busy public API can hold a request open for minutes; 429/5xx/disconnects are retried, 4xx (bad key/model, no balance) fail immediately |
| `MAX_API_CALLS` | 100000000 | the client's cap is **per process and per role** (default 10000). Data preparation and validation each run in one process and can exceed it; the verl launcher raises it automatically when unset. Your real budget is the API balance — measure with `scripts/api_usage.py` |
| `EMBED_DEVICE` | `cpu` | see §2.1 |

**Collect everything with the final role setup.** The environment fingerprint covers the config and the
reward code, not which models the judge and reader are. States, banks and the val bank collected with
another judge or reader would silently mix two reward functions. Change a role → re-run §4.

**Provider updates.** A public API can change the model behind an id (DeepSeek updated `deepseek-flash`
in October 2026). Answers are cached by model id, so after an update old and new judge answers would mix,
and checkpoints evaluated before and after it are not comparable. Before a long run check the provider's
change log. If the model changes mid-run, start a new `rl_loop.py` run (new `--out`) from the current
best checkpoint so the baseline is re-evaluated under the new judge, and point `cache` in the config to a
new directory (the cache path is not part of the fingerprint, so the collected states stay valid). The
local Defender does not have this problem as long as its weights and served name stay fixed.

### 2.3 Step time and API cost

The reward sets the pace. verl's reward loop scores all `batch × group` samples of a step concurrently
(64 by default); each sample makes its judge (API) and reader (GPU 0) calls one after another. A step
therefore takes about *calls per sample × latency per call* for the slowest sample, plus any queueing at
the Defender, and costs about *64 × API cost per sample*. Measure both before a long run: probe the start
policy on as many states as one step scores, then read the usage from the API cache (only real, billed
calls write cache files):

```bash
python -m admem.cli probe --config configs/rl_hidden_4b_v2.json \
  --states data/rl/builder_train.jsonl --samples 8 --limit 8 --out runs/probe_cost
python scripts/api_usage.py --cache .cache/admem/api --since 1h --samples 64 \
  --price-in <miss $/1M> --price-in-hit <hit $/1M> --price-out <$/1M>
```

Without `--base-url` the report also lists the local Defender and policy calls (free, but their call
counts and `calls_per_minute` show where the time goes); `--base-url deepseek` restricts the cost to the
API. Multiply the cost per sample by `batch × group × max-steps` for training, and add the validation
runs (measure the step-0 evaluation the same way: `--since` the start of the run). `cache_hit_tokens`
shows how much input DeepSeek served from its prompt cache (samples of one GRPO group share long prompt
prefixes). Cost levers: `--group 4` (fewer samples per state, noisier advantages), fewer val cases
(`--val-per-type`) or bank questions (`--max-bank-questions`), and a larger `--eval-every`. Rate-limit
errors (429) are retried with backoff; if they persist, reduce `--batch`.

### 2.4 Other layouts

* **Everything frozen on the API, both GPUs for the policy.** Keep the template's API `DEFENDER`
  (`deepseek-flash`), train with `--gpus 2`, and fit full fine-tuning with `--gpus 2 --offload
  --rollout-gpu-util 0.5` (actor parameters/optimizer and the reference model are parked on the CPU while
  the other phase uses the GPU; plan ≥ ~100 GB free host RAM, plus ~3 GB per reward worker for bge-m3) or
  LoRA with `--gpus 2 --lora-rank 64 --lr 1e-5 --rollout-gpu-util 0.6`. The reader is then a different,
  stronger model than the backbone, and every reader call is billed.
* **More GPUs** (e.g. 8×80 GB): `--gpus 4` for training and the judge, reader and val attacker served
  locally on the remaining GPUs (§3).

## 3. Roles, environment variables and the config

Every model call goes through a role. A role reads `<ROLE>_MODEL`, `<ROLE>_BASE_URL`, `<ROLE>_API_KEY`
and falls back to `LLM_*`.

| Role | Used for | Trained? |
| --- | --- | --- |
| `BUILDER` | Builder during data preparation (start/SFT model) | source of states |
| `ATTACKER` | Attacker during data preparation | source of states |
| `VAL_ATTACKER` | one-time frozen validation question bank | never |
| `JUDGE` | gate, faithfulness, retention/coverage judges, grading | never |
| `DEFENDER` | reader that answers questions from memory | never |
| `BUILDER_EVAL` / `ATTACKER_EVAL` | the checkpoint under validation; set automatically by `rl_loop.py` | — |

On the 2-GPU node, §2.2 sets these (API judge and val attacker, local Defender on GPU 0). For
reference, the same roles with every model served locally (more GPUs):

```bash
export LLM_API_KEY=EMPTY
export BUILDER_MODEL=qwen3-4b-start  BUILDER_BASE_URL=http://localhost:8001/v1
export ATTACKER_MODEL=qwen3-4b-start ATTACKER_BASE_URL=http://localhost:8001/v1
export VAL_ATTACKER_MODEL=judge-32b  VAL_ATTACKER_BASE_URL=http://localhost:8002/v1
export JUDGE_MODEL=judge-32b         JUDGE_BASE_URL=http://localhost:8002/v1
export DEFENDER_MODEL=reader-8b      DEFENDER_BASE_URL=http://localhost:8003/v1
export JUDGE_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'   # thinking judges only
```

For data preparation on the 2-GPU node, serve the start policy on GPU 1 (the Defender is already up on
GPU 0, §2.1); it stays up until §4 finishes:

```bash
CUDA_VISIBLE_DEVICES=1 NAME=qwen3-4b-start MODEL=/models/Qwen3-4B-Instruct-2507 \
  bash scripts/serve_policy.sh   # port 8001
```

Stop this server before training (§6): verl needs GPU 1. `rl_loop.py` and `alternate.py` start and stop
their own servers. The Defender on GPU 0 keeps running throughout.

Serve a model with a **versioned name** (`scripts/serve_policy.sh`, `NAME=...`): API responses are
cached by model name, so a new checkpoint under an old name would reuse stale answers.

**Config.** Use `configs/rl_hidden_4b_v2.json` (main setting; the Attacker never sees the official
question type) or `configs/rl_type_aware_4b_v2.json` (type-aware ablation; identical except
`hint_mode`). Both enable the v2 rewards, the v2 Builder prompt, token-budgeted packs and
`faith_gate: true`, so collected states do not start from memories polluted by earlier
unfaithful edits and validation can measure the faithful-write rate. Edit `tokenizer` and
`embed_model` to your local paths.

**One config end to end.** Every collected state records the config fingerprint (config values +
hash of the reward-relevant `admem/*.py` files). `verl_reward.py` refuses a state whose fingerprint
differs from the training config. If you change the config or reward-relevant code, re-collect the
states. Training/validation tooling (`validation.py`, `verl_launch.py`, `verl_adv.py`, `train.py`,
`learning.py`) is excluded from the fingerprint.

## 4. Prepare data

```bash
CONFIG=configs/rl_hidden_4b_v2.json DATA=data/longmemeval_s.json OUT=data/rl \
  TRAIN_PER_TYPE=10 VAL_PER_TYPE=2 bash scripts/prepare_rl_data.sh
```

Stages (rerun any subset with `STAGES="..."`):

| Stage | What it does | Output in `data/rl/` |
| --- | --- | --- |
| `prepare` | LongMemEval → `data/prepared_longmemeval` (300/50/150, identical histories kept in one split; official Q/A only in `private_eval.jsonl`) | — |
| `builder_states` | for train and val: cold-start bank → stream build with `BUILDER`, collecting every window as a state (with lookahead tests for cross-session evidence and coverage scoring) → export | `builder_{train,val}.jsonl(.parquet)` |
| `val_bank` | `VAL_ATTACKER` writes gated questions on the val cases; the same bank is used for every checkpoint | `val_bank/` |
| `attacker_states` | reference memories (`build_{train,val}/<case>/M_build.json`) → banks generated *against* those memories, collecting each pack as an Attacker state → export | `attacker_{train,val}.jsonl(.parquet)` |

`TRAIN_PER_TYPE` / `VAL_PER_TYPE` limit the number of cases per question type (cost control).
`VAL_PER_TYPE=2` gives 12 val cases × up to 30 bank questions.

**Check the reward before training.** Probe the start model on a few states and read the breakdown:

```bash
python -m admem.cli probe --config configs/rl_hidden_4b_v2.json \
  --states data/rl/builder_train.jsonl --samples 8 --limit 40 --out runs/probe_start
python scripts/probe_breakdown.py runs/probe_start/samples.jsonl --ceiling \
  --config configs/rl_hidden_4b_v2.json --states data/rl/builder_train.jsonl
```

Look for: groups with reward variance; which components drive it (`within_state_std`); how many
states have a mean below 0 (= below NOOP — RL will drift toward writing nothing); and whether
tests are answerable at all (`raw(E) only` accuracy). If the ceiling is low, fix the reader/judge or
the bank before spending GPU time.

## 5. Optional but recommended: SFT warm start

GRPO only sharpens behaviour the model already produces. If the start model's faithful-write rate is
low (the multi-session probe showed ~51%), first fine-tune on the verified demonstrations that the
collectors saved (only legal, faithful, test-passing outputs are stored as `completion`):

```bash
CUDA_VISIBLE_DEVICES=1 python -m admem.train --config configs/rl_hidden_4b_v2.json --data data/rl/builder_train.jsonl \
  --role builder --method sft --model /models/Qwen3-4B-Instruct-2507 --out runs/sft_builder --steps 500
python -m admem.cli merge --config configs/rl_hidden_4b_v2.json --model /models/Qwen3-4B-Instruct-2507 \
  --adapter "$(python -c "import json;print(json.load(open('runs/sft_builder/latest.json'))['checkpoint'])")/adapter" \
  --out /models/builder_sft
```

Use the merged model as `--model` below. If you change the Builder this way, re-run the
`builder_states` stage with it, so the states reflect the policy you start RL from.

## 6. Builder GRPO with validation and early stopping

Dry run first (prints every command, trains nothing):

```bash
python scripts/rl_loop.py --role builder --config configs/rl_hidden_4b_v2.json \
  --train data/rl/builder_train.jsonl.parquet --val data/rl/builder_val.jsonl.parquet \
  --model /models/builder_sft --prepared data/prepared_longmemeval --val-bank data/rl/val_bank \
  --val-per-type 2 --out runs/rl_builder --eval-every 20 --max-steps 300 \
  --gpus 1 --lora-rank 64 --lr 1e-5 --rollout-gpu-util 0.5 --dry-run
```

Run it in the shell with `CUDA_VISIBLE_DEVICES=1` and the Defender up on GPU 0 (§2.1). Then the same
command without `--dry-run`. Each round:

1. **Train** `--eval-every` steps: `python -m verl.trainer.main_ppo` with `resume_mode=auto`, so each
   chunk resumes from the previous checkpoint (optimizer and dataloader state included). Metrics go to
   `runs/rl_builder/logs/train_to_XXXXX.jsonl` (verl file logger, one file per chunk).
2. **Merge** `verl/global_step_N/actor` into a Hugging Face model (`python -m verl.model_merger merge
   --backend fsdp`) under `merged/step_N`; with LoRA, `scripts/merge_lora.py` then folds the adapter in.
3. **Validate**: start `vllm serve merged/step_N` as `BUILDER_EVAL` (unique served name per
   checkpoint), run `python -m admem.validation builder` on the val split, stop the server.
4. **Decide** with the stopping rules (§7) and record everything in `loop_state.json`.

The initial model is evaluated once before training (`eval/step_00000`) and serves as both the
guard baseline and the first "best". Interrupting and re-running the same command resumes; finished
evaluations are not repeated. The result is `runs/rl_builder/summary.json`:

```json
{"best": {"tag": "step_00080", "step": 80, "primary": 0.47, "model_path": "runs/rl_builder/merged/step_00080"},
 "stopped_early": true, "reason": "no significant improvement in 3 evaluations", "history": [...]}
```

With `--keep-merged best` (default) only the best and the latest merged models are kept. verl keeps
only the latest actor checkpoint (`max_actor_ckpt_to_keep=1`).

**Main training options** (`python scripts/rl_loop.py -h`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--gpus` | 2 | GPUs for verl; must match the GPUs visible to the job (`--gpus 1` with `CUDA_VISIBLE_DEVICES=1` in the layout of §2.1), otherwise Ray waits for resources forever |
| `--offload` | off | park actor parameters/optimizer and the reference model on the CPU (2-GPU full fine-tuning, §2.4) |
| `--lora-rank` / `--lora-alpha` | 0 / 2×rank | LoRA instead of full fine-tuning (use ~1e-5 learning rate); merged into full weights before validation |
| `--rollout-gpu-util` | 0.45 | GPU memory fraction of the vLLM rollout engine |
| `--batch` / `--group` | 8 / 8 | prompts (states) per step / samples per prompt (GRPO group) → 64 reward calls per step |
| `--lr`, `--kl-coef` | 1e-6, 0.02 | actor LR (constant schedule, no warmup, so chunked restarts don't change it) and low-var KL loss to the start model |
| `--response-tokens` | 3072 | longer outputs are truncated → invalid JSON → invalid reward |
| `--prompt-tokens` | from config | `builder_input_tokens + 256` (Attacker: `attacker_input_tokens + 256`); `truncation=error` |
| `--reward-workers` | 8 | verl reward-loop workers; each scores samples in a thread pool, so judge calls run in parallel |
| `--estimator` | `admem_grpo` | `grpo` = verl's original estimator |
| `--std-floor` | 0.1 | advantage normalisation floor for `admem_grpo` |
| `--reward-keys` | `new` | `legacy` for older verl (top-level `custom_reward_function`) |
| `-- key=value ...` | — | any extra Hydra override, e.g. `-- actor_rollout_ref.model.lora_rank=64` |

**Throughput.** Reward scoring dominates. In the multi-session probe one Builder sample took about 27 s
of judge calls when scored serially (v1 reward). With 64 samples per step, the reward-loop workers
and judge throughput decide the step time. Measure one step before planning a run, and scale the
judge service, not the trainer.

## 7. Stopping criteria

Training stops and the checkpoint is chosen **on the val split only**, never on test and never on the
official target question of a case.

* **Why not "memory answers the target question".** That question is the evaluation label (early
  stopping on test). A single question per case is far too noisy (±7 points on 50 cases), and a Builder
  can pass it while dropping everything else. At deployment no target question exists.
* **Why not training reward.** The Builder reward can rise by writing less (faithfulness and
  retention penalties dominate it); it says nothing about held-out QA.

**Builder metrics** (`admem.validation builder`, written to `eval/<tag>/metrics.json`):

| Metric | Role |
| --- | --- |
| `bank_accuracy` on the frozen `val_bank` (≤`--max-bank-questions` per case) | **primary**; per-question results feed the paired bootstrap |
| `official_accuracy` on the val cases' LongMemEval questions | reported, not used for decisions |
| `legal_rate`, `faith_rate`, `noop_rate`, `fallback_rate`, `mean_m_tokens` | guards |
| `kl_loss` from the verl log of the chunk | optional guard (`--max-kl`) |

**Attacker metrics** (`admem.validation attacker`, against fixed reference memories
`data/rl/build_val/<case>/M_build.json`): primary `defects_per_pack` (gate-accepted questions that
expose a repairable defect of the reference memory, per sampled pack); guards `invalid_rate`,
`accept_rate`.

**Rules** (`admem.validation.EarlyStopper`):

| Rule | Default | Effect |
| --- | --- | --- |
| Improvement | paired bootstrap over shared questions: P(new > best) ≥ `--confidence` 0.9 and mean gain > `--min-delta` 0 | becomes the new best |
| Patience | `--patience 3` evaluations without improvement | stop |
| Guards vs. the step-0 baseline | Builder: legal −0.05, faith −0.05, NOOP +0.15, fallback +0.10, memory size ×1.5; Attacker: invalid +0.10, accept −0.10; optional `--max-kl` | a violating checkpoint cannot become best; `--guard-patience 1` consecutive violations stop training |
| Budget | `--max-steps` | stop; the best checkpoint so far is the result |

The NOOP guard is the collapse detector: if the policy learns to write nothing, its reward rises to 0
while `noop_rate` jumps. Tune tolerances with `--faith-tol`, `--noop-tol`, `--size-ratio` etc.

To also see how far memory is from the raw history on the same questions, run once:

```bash
python -m admem.validation builder --reference full --config configs/rl_hidden_4b_v2.json \
  --prepared data/prepared_longmemeval --val-bank data/rl/val_bank --per-type 2 --out runs/val_full_ceiling
```

## 8. Attacker GRPO

```bash
python scripts/rl_loop.py --role attacker --config configs/rl_hidden_4b_v2.json \
  --train data/rl/attacker_train.jsonl.parquet --val data/rl/attacker_val.jsonl.parquet \
  --model /models/Qwen3-4B-Instruct-2507 --prepared data/prepared_longmemeval \
  --memory-run data/rl/build_val --snapshot M_build --packs-per-case 8 --val-per-type 2 \
  --out runs/rl_attacker --eval-every 20 --max-steps 200 --response-tokens 2048 \
  --gpus 1 --lora-rank 64 --lr 1e-5 --rollout-gpu-util 0.5
```

The Attacker reward (v2) is ≤0 for rejected items, below 0 for invalid output, and positive only for
accepted, non-duplicate questions with necessary, personal evidence; the defect bonus uses the memory
the state was collected against.

## 9. Alternating Builder and Attacker

`scripts/alternate.py` trains the two roles in turns, each against a frozen opponent. This differs from
Code-A1, which updates both models every step from one shared rollout. In advmem the Attacker attacks a
*whole memory* built over hundreds of Builder steps, and the rewards come from slow judge calls. So the
opponent has to be frozen for a phase, and memories are rebuilt between phases.

```bash
python scripts/alternate.py --config configs/rl_hidden_4b_v2.json --prepared data/prepared_longmemeval \
  --builder /models/builder_sft --attacker /models/attacker_sft --val-bank data/rl/val_bank \
  --out runs/alternate --rounds 4 --phase-steps 60 --eval-every 20 --patience 2 --gpus 1 \
  --train-per-type 10 --val-per-type 2 \
  --rl-args "--batch 8 --group 8 --lora-rank 64 --lr 1e-5 --rollout-gpu-util 0.5" --dry-run
```

Each round, with current Builder `B` and Attacker `A`:

| Step | What happens | Output in `runs/alternate/round_k/` |
| --- | --- | --- |
| `memories` | serve `B`; build train and val memories | `build_{train,val}/` |
| `attacker_states` | serve `A`; bank against a **memory pool** with `--collect`, then export: `--pool-share` (0.5) of the packs use `B`'s memories, the rest are spread over up to `--pool-size` (3) earlier rounds' memories | `collected_attacker/`, `attacker_train.jsonl.parquet` |
| `attacker_phase` | `rl_loop.py --role attacker`, validated against `B`'s val memories → `A'` | `rl_attacker/` |
| `adversarial_bank` | serve `A'`; bank against `B`'s train memories with `--defects-only`, merged into the cumulative bank (`--base-bank`) | `bank/` |
| `builder_states` | serve `B`; rebuild with the new bank and `--collect`. Builder calls hit the API cache (build nonces don't depend on the bank), so only the tests are recomputed | `collected_builder/`, `builder_train.jsonl.parquet` |
| `builder_phase` | `rl_loop.py --role builder` with the frozen `--val-bank` → `B'` | `rl_builder/` |

How this maps to the risks of alternating training:

* **Forgetting.** The cumulative bank is the Mistake Book: questions from every round stay in the bank
  and come back as replay tests.
* **The Attacker overfitting one Builder.** The memory pool keeps earlier Builders' weaknesses in the
  Attacker's training data. Each pack's memory is chosen deterministically and stored in the state, so the
  defect reward uses the memory the question was written against (`memory_pool_index` in `bank_log.json`).
* **Unfixable "attacks".** With `require_reader_answerable: true` (on in the `rl_*` configs), a question
  is valid only if the actual reader (`DEFENDER`) answers it from the raw evidence. This is the analogue of
  Code-A1 checking tests against reference code. The v2 Attacker reward rejects other questions (stage
  `reader_unanswerable`, just below zero) and banks drop them. If the memory plus evidence fails but the
  evidence alone works, the question stays valid as `retrieval_composition_candidate`: noise in the
  memory crowded the evidence out, which the Builder can fix.
* **Overfitting a frozen opponent.** Phases are short (`--phase-steps`, `--patience 2`). Each phase starts
  from the previous best and stops at its own early-stopping rule (§7).
* **Measuring progress.** The Builder is always selected on the frozen val bank written by
  `VAL_ATTACKER`, which is never trained, so its score doesn't move just because the Attacker changed.
  The Attacker is selected on defects found in the *current* Builder's val memories.

**Stopping.** The run stops after a round in which neither phase selected a checkpoint better than its
starting model: the new Attacker found no more defects in the current Builder's memories, and the Builder
did not improve significantly on the frozen val bank. It also stops after `--rounds`. `--start builder`
skips the first Attacker phase. Every finished step is recorded in `alternate_state.json`, and re-running
the same command resumes. `runs/alternate/summary.json` lists the final models and each round's phase
results. In `--dry-run` no phase produces a checkpoint, so the dry run stops after round 1.

The orchestrator serves models on `--serve-port` (8200) for data steps; `rl_loop` validation uses the next
port. On the 2-GPU node run it with `CUDA_VISIBLE_DEVICES=1`: every Builder/Attacker server and training
phase then uses GPU 1, while the Defender keeps GPU 0. Each phase starts a fresh LoRA adapter on the
previous phase's merged model. Served names include a hash of the weights path, so cached API responses are never shared between
different weights.

## 10. Final test evaluation (once)

Serve the selected checkpoint and evaluate the test split with the native evaluator:

```bash
CUDA_VISIBLE_DEVICES=1 NAME=builder-final MODEL=runs/rl_builder/merged/step_00080 \
  bash scripts/serve_policy.sh   # port 8001
BUILDER_FINAL_MODEL=builder-final BUILDER_FINAL_BASE_URL=http://localhost:8001/v1 \
python -m admem.cli longmemeval-eval --config configs/rl_hidden_4b_v2.json \
  --prepared data/prepared_longmemeval --split test --builder-role BUILDER_FINAL \
  --snapshot M_final --out runs/test_final
```

`runs/test_final/eval/predictions.jsonl` can be scored with the official LongMemEval evaluator. The
summary carries a `benchmark_informed` note: the prompts were written with all 500 LongMemEval
questions visible, so a clean held-out claim needs another benchmark.

## 11. One-shot training without validation

```bash
CUDA_VISIBLE_DEVICES=1 ROLE=builder CONFIG=configs/rl_hidden_4b_v2.json MODEL=/models/builder_sft \
TRAIN=data/rl/builder_train.jsonl.parquet OUT=runs/grpo_builder STEPS=200 \
GPUS=1 LORA_RANK=64 LR=1e-5 ROLLOUT_GPU_UTIL=0.5 bash scripts/train_verl.sh
```

(`DRY_RUN=1` prints the command.) Checkpoints are saved every `SAVE_FREQ` steps; select among them
with `python -m admem.validation builder` as in §7. With LoRA, merge each checkpoint with
`python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <dir>.parts`
followed by `python scripts/merge_lora.py --base <dir>.parts --adapter <dir>.parts/lora_adapter --out <dir>`.

## 12. Monitoring

* verl logs per step, besides its own metrics, the reward diagnostics returned by `compute_score`:
  `legal`, `noop`, `faith_ratio`, `gain`, `loss`, `information_loss`, `growth_ratio`, and every reward
  component as `component_*` (Attacker: `legal`, `accepted`, `rejected`, `duplicates`, `over_budget`).
  Rising `noop` with flat `component_coverage` is an early sign of collapse.
* `runs/<run>/logs/train_to_*.jsonl`: per-step metrics for each chunk.
* `runs/<run>/eval/<tag>/metrics.json`: per-checkpoint validation; `loop_state.json`: decisions.
* API usage and cost so far: `python scripts/api_usage.py --cache .cache/admem/api --since 1d --base-url
  deepseek --price-in ... --price-out ...` (counts only real, billed calls; cache hits are free). Without
  `--base-url` it also shows the local Defender's call rate.

## 13. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Reward environment configuration mismatch` | states were collected with another config or other reward-relevant code; re-run `prepare_rl_data.sh` with the training config |
| `NotImplementedError`/unknown data source `admem_builder` in verl's default scorer | the custom reward was not loaded: wrong `--reward-keys` for your verl version |
| `Prompt length ... exceeds` (truncation=error) | a state is longer than `--prompt-tokens`; raise it (default comes from the config budgets) |
| a judge returns malformed JSON and the job stops | default is fail-fast. `ADMEM_REWARD_ON_UNKNOWN=neutral` returns 0 (the NOOP / empty-list reward), logs to `admem_reward_unknown.jsonl` and still stops after `ADMEM_MAX_UNKNOWN` (20) failures |
| `admem_grpo` unknown estimator | `admem.verl_adv` was not imported: the launcher sets `VERL_USE_EXTERNAL_MODULES=admem.verl_adv` and `PYTHONPATH`; for multi-node Ray pass them via `ray_kwargs.ray_init.runtime_env.env_vars` |
| validation server never starts | see `logs/serve_<name>.log`; set `--serve-gpus`, `--serve-port`, `--max-model-len`, or a custom `--serve-cmd` |
| rollout or training OOM | single GPU: use `--lora-rank 64 --lr 1e-5`, lower `--rollout-gpu-util` / `--response-tokens`; two GPUs: also `--offload` |
| verl hangs at start waiting for resources | `--gpus` is larger than the number of visible GPUs (`--gpus 1` with `CUDA_VISIBLE_DEVICES=1`), or a vLLM server (e.g. the data-preparation one on port 8001) still holds GPU 1 |
| verl or a policy server lands on GPU 0 and runs out of memory | `CUDA_VISIBLE_DEVICES=1` was not set in that shell; GPU 0 belongs to the Defender |
| steps much slower than the API latency explains | the Defender is the queue: give it more of GPU 0 (`GPU_MEMORY_UTILIZATION`), check `calls_per_minute` for its model in `scripts/api_usage.py` (no `--base-url`) |
| `MAX_API_CALLS exhausted` / `达到 API 调用预算` | the per-process, per-role cap; `export MAX_API_CALLS=100000000` (the template does) |
| `HTTP 401` / `402` / `400 ... model` from the API | wrong key / no balance / wrong model id or rejected `*_EXTRA_BODY` field; run `scripts/check_roles.py` |
| many `HTTP 429` or timeouts | the API is throttling; retries back off automatically (`API_RETRIES`, `API_BACKOFF`); lower `--batch` if they persist |

## 14. What has been verified

Offline (no GPU, no verl install): the launcher's override keys against verl `3c6d9b4`'s generated
trainer config; the `admem_grpo` estimator's registration and masking with stand-ins for torch/verl;
`compute_score` on exported Builder and Attacker states through a fake OpenAI-compatible server;
`prepare_rl_data.sh`, both `admem.validation` modes and a dry run of `rl_loop.py` end to end; the
stopping rules and the loop's stop/resume behaviour in unit tests; a pooled adversarial bank with
`require_reader_answerable` through the real CLI; `alternate.py`'s round order, memory pool, cumulative
bank, stopping rule and resume in unit tests, plus its dry run. Not yet run: an actual verl GRPO job, the
model merger and vLLM serving on GPUs. Do a short run (`--max-steps 2 --eval-every 1` on a handful of
states; for `alternate.py`, `--rounds 1 --phase-steps 2 --eval-every 1`) before the real one.
