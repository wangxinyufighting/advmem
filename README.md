# LongMemEval Full Memory Lab

把一个 `longmemeval_s.json` case 构建为无损原文库，进行检索问答、pack 覆盖实验、attacker 出题实验。Python 3.10+；中文注释；不依赖 MemOS 服务、图数据库或 agent 框架。

**full memory / full graph = 全量原文 F，不是 builder 编辑的记忆 M。**

本项目实现的是这三个实验的推理与评测代码，不包含 builder、GRPO 训练、patch/refine，也不把通过 gate 的问题冒称为经过 defect 筛选的 Mistake Book Q。

## 1. 文件结构

```text
memory.py       无损节点、时间顺序、结构边、原文/可编辑M的统一Document接口
retrieve.py     BM25+dense、RRF、邻接候选扩展、可选重排/补检索
packs.py        无放回种子遍历、近邻pack、可选主题簇、一次主动检索
agents.py       attacker、reader、独立oracle、gate
metrics.py      官方标签隔离、覆盖曲线、最小N、目标问答等价判分
run.py          build / answer / packs / attack / weights 命令行入口
llm.py          OpenAI-compatible HTTP、重试、JSON检查、缓存
common.py       小型读写/分词工具，大JSON流式读取
examples/       明确标注的合成样例及实际离线运行结果
tests/          离线单元测试；模型用例使用脚本stub，不是真实LLM
```

## 2. MemOS 检索策略：参考范围与本项目的取舍

核对日期：**2026-09-30**。参考的是 MemOS `main` 中的 **Python TreeTextMemory 检索路径**，不是把不同的 cloud/plugin 检索都描述成一个算法。下面链接指向当时可访问的 `main`，不是固定 commit；正式实验应自行固定依赖及模型版本。

| 源码环节                         | MemOS 中的做法                                                                                              | 本项目采用什么                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `TaskGoalParser`               | fast 使用原查询；开启 fast_graph 时提取词作为 key/tag；fine 可由 LLM 解析、改写查询                         | 默认原查询；需要时用有界 planner 生成互补查询，不建立实体标签层 |
| `GraphMemoryRetriever`         | 按配置并发调用结构化 key/tag 查询、向量召回、BM25、全文检索，合并后按 ID 去重；WorkingMemory 有专门读取路径 | 每个 case 内 BM25+dense 两路召回，按 rid 合并；用 RRF 融合排名  |
| `Searcher`                     | 按记忆类型分路，可配置额外通道；调用配置的 reranker，再去重、排序、截取 top-k，并记录使用                   | 只搜索本 case；可选 cross-encoder 重排，按原文预算输出          |
| `AdvancedSearcher.deep_search` | 初次召回后评估能否回答，分阶段生成补检索词；代码还包含 memory recreation/enhancement                        | 可配置有限次补检索，保留完整原文和 rid，**不做原文重写**  |

不能把 MemOS 简化为“向量 top-k”，但也不能声称所有路径总会开启。`TreeTextMemory` 中的默认 reranker 配置包含 `cosine_local`，不等于默认必然使用 cross-encoder。

**本项目独立选择的部分：**RRF 不是这里声称的 MemOS 默认公式；round 前后邻接扩展不是 MemOS key/tag 图查询的逐行复现。没有移植其外网检索、记忆类型路由、图数据库或记忆重写。邻接项只进入候选集合，是否进入最终 top-k 取决于融合/重排，不保证扩展项必然可见。

源码依据：

- [TaskGoalParser](https://github.com/MemTensor/MemOS/blob/main/src/memos/memories/textual/tree_text_memory/retrieve/task_goal_parser.py)
- [GraphMemoryRetriever / recall.py](https://github.com/MemTensor/MemOS/blob/main/src/memos/memories/textual/tree_text_memory/retrieve/recall.py)
- [Searcher](https://github.com/MemTensor/MemOS/blob/main/src/memos/memories/textual/tree_text_memory/retrieve/searcher.py)
- [AdvancedSearcher](https://github.com/MemTensor/MemOS/blob/main/src/memos/memories/textual/tree_text_memory/retrieve/advanced_searcher.py)
- [TreeTextMemory 配置入口](https://github.com/MemTensor/MemOS/blob/main/src/memos/memories/textual/tree.py)

## 3. 安装和模型配置

```bash
python -m venv .venv
source .venv/bin/activate
# BM25-only 或 API embedding：
pip install -r requirements.txt
# 使用本地 embedding 或本地 cross-encoder 时再装：
pip install -r requirements-local.txt
# 运行测试时：
pip install pytest
```

三个 embedding 模式：

| 参数                  | 意义                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--embedding none`  | 只有 BM25；可完全离线验证构建和覆盖管线，不冒充混合检索                                                             |
| `--embedding local` | BM25 + SentenceTransformer；默认`sentence-transformers/all-MiniLM-L6-v2`，也可 `--embed-model 本地目录或模型名` |
| `--embedding api`   | BM25 +`/embeddings` API；必须提供 `EMBED_MODEL` 或 `--embed-model`                                            |

本地模型第一次使用需要已有权重或能下载权重；默认模型只是便于启动的工程默认值，不是经过本项目选优的 LongMemEval 模型。配置需要 query/passage 前缀的模型时，使用 `EMBED_QUERY_PREFIX`、`EMBED_DOCUMENT_PREFIX`。可选重排通过 `--reranker-model 模型名或本地目录` 启用，要求模型每对文本输出一个分数。

LLM 的最少配置：

```bash
export LLM_BASE_URL="https://api.openai.com/v1"   # 也可换成兼容服务或本地vLLM地址
export LLM_API_KEY="你的密钥"
export LLM_MODEL="你实际可用的Chat-Completions模型"
```

`ATTACKER_MODEL`、`DEFENDER_MODEL`、`JUDGE_MODEL`、`PLANNER_MODEL` 可分别覆盖；相应的 `*_BASE_URL` / `*_API_KEY` 也可覆盖。未覆盖时共用 `LLM_*`。测试 `--retrieval-only`、无 `--probe-target` 的 `packs` 不需要 LLM。

只在使用 API embedding 时配置：

```bash
export EMBED_BASE_URL="https://api.openai.com/v1"
export EMBED_API_KEY="你的embedding密钥"
export EMBED_MODEL="你的embedding模型名"
```

`.env.example` 提供完整示例；复制并填写后可以 `source .env`，脚本不自动读取 dotenv。不支持 `response_format` 的服务可设 `LLM_JSON_MODE=0`；不接受默认 temperature/max_tokens 的模型通过 `LLM_EXTRA_BODY` 显式覆盖。HTTP 接口依据 [Chat Completions](https://developers.openai.com/api/reference/resources/chat) 和 [Embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create)，没有假定任意兼容服务都支持全部参数。

如果判官返回 `finish_reason=length`，日志中的 `[TRUNCATED_OUTPUT]` 表示输出预算耗尽，不是记忆或 faithfulness 判定失败。判官 JSON 的第二次尝试会自动将输出预算翻倍，默认上限为 `16384`；可用 `JUDGE_MAX_TOKENS`、`JUDGE_JSON_RETRY_MAX_TOKENS`（或对应的 `LLM_` 变量）调整。使用 Qwen3/vLLM 等默认启用思考的服务时，也可以在服务支持的前提下关闭判官思考，例如 `JUDGE_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'`，以免思考 token 吃完整个 JSON 输出预算。

## 4. 数据准备：不要悄悄换成 cleaned

官方原始数据仓库的 S 文件名是 `longmemeval_s`，远端没有 `.json` 后缀；本地可以保存成 `longmemeval_s.json`：

```bash
mkdir -p data
curl -L 'https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s' \
  -o data/longmemeval_s.json
```

已有文件时不需要下载。另一个仓库提供 `longmemeval_s_cleaned.json`，是不同数据版本。本项目不自动替换；通过 `--data` 明确指定，保存 case/full 内容哈希，拒绝将另一个版本的 full memory / pack 混入当前 case。

依据：[LongMemEval README](https://github.com/xiaowu0162/LongMemEval)、[原版数据文件列表](https://huggingface.co/datasets/xiaowu0162/longmemeval/tree/main)、[cleaned 文件列表](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main)。数据下载及使用遵守源数据的许可。

## 5. 构建 full memory

```bash
python run.py build \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --out runs/case0
```

可用 `--question-id 某个ID` 替代 `--case-index`。大文件使用 ijson 按 case 流式扫描；不会为了一个 case 必须把全部 S 文件加载进内存。后续命令会自动构建/校验同一份 full memory，embedding 索引按内容哈希复用；也可加 `--full runs/case0/full_memory.json` 显式加载。

构建过程不调用生成式 LLM；embedding 可以是本地 encoder 或 API。依次执行：

1. 只提取三个历史字段，剥离消息级 `has_answer` 等标签；保留正文和其余消息属性。官方 q/a 不进入 F。
2. 为每个 session 保留原始 ID、日期、原始位置；为 user+紧随assistant 建 round，稳定 ID 为 `s1:r1` 等。孤立消息不丢弃。
3. 原始顺序和时间排序分开存储。包含边、round先后边、session时间边由顺序结构确定性导出，不重复保存冗余图。
4. 自动断言 `full.export_history() == history_only(case)`：比较历史结构和逐字文本，**不是 JSON 缩进/转义方式的文件字节相等**。
5. 构建 BM25 与可选 dense 索引。embedding 长文本分块仍映射原 rid；dense 以块间 max-sim 聚合到 round。检索最终返回原文，不返回 embedding 切片。

输出包括 `full_memory.json`、`index_manifest.json`、`run.build.json`。

### 5.1 原生 LongMemEval 划分与 Builder 评测

如果要在真实 LongMemEval 上测 Builder，请使用 `admem` 的准备流程。它会把完整历史写入 `cases/`，把官方问题、答案和证据标签单独写入 `private_eval.jsonl`，并按完整历史分组后划分 train/val/test；Builder 阶段看不到这些评测标签。

`longmemeval-eval --prepared` 接受下面命令生成的原生 prepared 目录。你也可以直接把 `prepare_longmemeval_cases.py` 生成的原始 `test.json` 和 `test_labels.jsonl` 传给 `--cases/--labels`；不要传 `*_builder.jsonl`，后者是按 session 导出的 SFT 格式，不是完整历史评测输入。

以官方 S 数据为例（`--sizes` 三个数字必须加起来等于输入 case 总数）：

```bash
python -m admem.cli prepare \
  --config configs/type_aware.json \
  --data data/longmemeval_s.json \
  --out data/prepared_longmemeval \
  --sizes 300 50 150
```

使用现有配置中的 Builder（例如 Qwen）在独立 test split 上先构建 memory，再读取私有标签做问答评测：

```bash
python -m admem.cli longmemeval-eval \
  --config configs/type_aware.json \
  --prepared data/prepared_longmemeval \
  --split test \
  --builder-role BUILDER \
  --snapshot M_final \
  --out outputs/longmemeval_builder_test
```

已有按 question_id 划分的真实 LongMemEval split 时，可以直接运行：

```bash
python -m admem.cli longmemeval-eval \
  --config configs/type_aware.json \
  --cases data/longmemeval_splits/test.json \
  --labels data/longmemeval_splits/test_labels.jsonl \
  --builder-role BUILDER \
  --out outputs/longmemeval_test_qwen3_4b
```

`--cases` 只用于构建无损历史；问题和答案只在 Builder 完成后从 `--labels` 读取。文件名为 `test.json` 时会自动记录为 test split，也可显式传 `--split test`。

可用 `--question-types` 只评测指定题型，并与 `--keys`、`--limit` 组合。例如：

```bash
python -m admem.cli longmemeval-eval \
  --config configs/type_aware.json \
  --cases data/longmemeval_splits/test.json \
  --labels data/longmemeval_splits/test_labels.jsonl \
  --split test --question-types temporal-reasoning knowledge-update \
  --limit 2 --builder-role BUILDER \
  --out outputs/longmemeval_temporal_update_smoke
```

题型名称包括 `single-session-user`、`single-session-assistant`、`single-session-preference`、`multi-session`、`knowledge-update` 和 `temporal-reasoning`。原始 `--cases` 路径仍会扫描并校验整个 split 的 case/label 对齐，但只为筛选后的 case 构建 `FullMemory` 并调用模型。

先用 `--limit 1` 做配置和 API smoke test；正式运行请使用新的输出目录。结果包括 `builder_summary.json`（每个窗口的 JSON 合法率、faithfulness 和 fallback 统计）、`run/` 下的 `M_build.json`/`M_final.json`，以及 `eval/summary.json`。`longmemeval-eval` 沿用现有 Builder prompt 和 `run_case(mode="build")` 路径，不使用 probes 目录中的合成状态。`eval/` 是项目内的诊断 judge 结果；如需官方分数，仍应使用 LongMemEval 官方评测器。

若要在 Python 中直接使用：

```python
from common import load_case
from memory import FullMemory
from retrieve import Embedder, Retriever
from packs import Sampler

case = load_case("data/longmemeval_s.json", case_index=0)
full = FullMemory.build(case)
retriever = Retriever(full.documents(), Embedder("local"))
packs = Sampler(full, retriever, seed=0).sample(40)
# sampler 不接受官方 question / answer / answer_session_ids。
```

## 6. 实验1：full memory → retrieve → 回答 target question

基础 fast 版本：

```bash
python run.py answer \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --k 10 --steps 0 \
  --out runs/case0_answer
```

带两次上限的补检索：

```bash
python run.py answer \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --k 10 --steps 2 --expand 1 \
  --context-chars 80000 --out runs/case0_answer_deep
```

`--steps` 是上限，planner 判断足够时提前停止。planner 只读目标问题、日期和已检索原文，不读标准答案；这是测试1的正常 query-time retrieval，不参与测试2/3的种子构建。多个查询各自召回后合并，最终仍约束在 top-k 和上下文预算内。

只测检索、不调用 reader/judge，追加 `--retrieval-only`。

输出：

| 文件/字段                               | 内容                                              |
| --------------------------------------- | ------------------------------------------------- |
| `retrieved_context.txt`               | reader 真正可见的完整原文                         |
| `answer_report.json / retrieved_hits` | 召回 ID、分数、来源通道                           |
| `answer_report.json / evidence`       | 可见上下文的 session/round 证据覆盖               |
| `answer_report.json / hypothesis`     | reader 答案                                       |
| `answer_report.json / llm_judge`      | 本项目的类型感知 LLM 判分，**不是官方分数** |
| `predictions.jsonl`                   | `question_id` / `hypothesis`，可交官方脚本    |

正式得分使用官方评测器，例如：

```bash
# 官方脚本还需要其依赖与OPENAI_API_KEY；不使用本项目的LLM_API_KEY变量。
python /path/to/LongMemEval/src/evaluation/evaluate_qa.py \
  gpt-4o runs/case0_answer/predictions.jsonl data/longmemeval_s.json
```

调用形式和题型规则依据[官方 evaluate_qa.py](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)。示例是其模型参数，不声称该名称对应最新模型快照。本项目同时报告规范化 exact match，但不以它替代偏好/时间题的官方判分。

## 7. 实验2：采样 N 个 pack，测证据覆盖与最小 N

```bash
python run.py packs \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --n 40 --seed 0 --neighbors 8 \
  --trials 100 --out runs/case0_packs
```

每个 sweep 对 session 种子**无放回**随机遍历，全部种子用完才进入下一个 sweep。N=40 是示例预算，不假定每个真实 case 恰好有40或50个session。输出构建统计后，可将 N 调到至少一个完整 sweep。

每个 session pack 包含：完整种子 session + 以该 session 各条用户发言分别检索、取并集排序后的最多8个其他session round。按日期排序，加 provenance 标记。`--memory M.jsonl` 提供当前 M 时读取其 prov 生成标记；没有 M 则所有 round 标 `[∉M]`。标记是 pack 视图，不修改 F。

仅采样成功时账本仍为 `unaudited`、visit为`pack_ready`；真正调用 attacker 后才记 `asked/audited_empty`。`audited_empty` 不等于这段历史无价值。

### 指标定义

官方 `answer_session_ids` 映射到原 session ID；官方消息 `has_answer` 映射到其所属 round。这些标签只在评测侧读取。[官方说明](https://github.com/xiaowu0162/LongMemEval)本身区分 session 与 turn 召回。

记 gold session 为 S*，gold round 为 E*，第i个pack的可见round为 P_i。报告包含：

| 字段                                 | 含义                                                                               |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| `this_pack_all_sessions`           | 当前pack至少碰到每个gold session；**不保证碰到其证据round**                  |
| `this_pack_all_rounds`             | 当前pack包含所有标注证据round                                                      |
| `any_pack_all_rounds`              | 前N个pack中，至少一个pack自己就拿齐证据                                            |
| `union_all_rounds`                 | 前N个pack合并后拿齐证据，可能没有任何单个pack能完成出题                            |
| `min_observed_prefix_N`            | 这次固定采样顺序里首次达到对应条件的N；预算内未找到为null                          |
| `min_subset_within_generated_pool` | 从已产生的有限pack池中，事后用标签求最少几个pack的并集能覆盖；不是全局最优出题策略 |

最小集合覆盖采用精确位掩码DP，只对最多18个gold单位求解，过大则明确跳过，不冒用贪心结果声称精确。session和round各算一次。

`--trials 100` 会生成完整的固定种子池，重复随机重排，报告各N的覆盖成功概率和**仅在成功试验中的**N中位数；失败试验不从成功率分母删除。它测的是种子顺序随机性，**不是100次重新调用attacker**。固定检索/固定M下一个sweep之后的重复pack不会带来新的证据覆盖。

输出：`packs.jsonl`、`ledger.json`、`pack_report.json`；多次试验还保存 `pack_pool.jsonl`。

### 目标题本身是否可由 pack 回答

证据命中不等于题目可回答，尤其涉及全量聚合或标签不完备时。可加：

```bash
python run.py packs \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --n 40 --seed 0 --probe-target \
  --out runs/case0_packs_probe
```

所有 pack 先固定，之后 judge 才检查“此pack原文是否支持官方答案”，返回 `target_support_probes` 和 `target_supported_min_observed_prefix_N`。这只是 LLM 支持性诊断，不是答案完备性的证明，也不是实际reader答对率。

### 可选主题簇

加 `--cluster-threshold 0.75`：对round向量做确定性贪心余弦聚类，保留跨session簇、按session轮询截取最多12个round，作为额外种子。阈值是预算/算法参数，不是已调优结论；BM25-only模式不能开启。

## 8. 实验3：同一批 pack → attacker → gate → 覆盖曲线

先用实验2保存的pack，隔离“pack变化”和“出题能力变化”：

```bash
python run.py attack \
  --data data/longmemeval_s.json --case-index 0 \
  --embedding local --packs runs/case0_packs/packs.jsonl --n 40 \
  --questions-per-pack 4 --seed 0 --gate full \
  --out runs/case0_attack
```

`--n` 是pack数，`--questions-per-pack`是**每pack最多题数**；不是4次采样每次再输出4题。给定空列表是合法行为，不用问题填满预算。

attacker 输入只有：原文pack、M provenance标记、题型、case允许使用的 `question_date`。不传目标问题/答案/证据标签。`question_date`逐字复制；题型使用官方名称，并接受设计中的 `preference` / `temporal` 别名。

题型来源由 `hint_mode` 决定（admem 里 attacker 侧现由 `type_policy` 单独控制，默认 `auto` 保持本段行为，`balanced` 不读官方题型，见第13节）。`configs/type_aware.json`（`target_type`）把**当前case的官方题型**交给 attacker/builder，属于有意的类型已知诊断基线，不是无标签设定；`configs/type_hidden.json`（`hidden`）按pack确定性随机抽题型，才是主结果设定（现已与 `type_aware.json` 只差 `hint_mode`；旧的不匹配版本保留为 `type_hidden_legacy.json`，见第14.4节）。admem 中可用以下配置给 4B attacker 减负、抑制刷分：

- `attacker_view: "compact"`：单session题型只给种子session；跨session题型给种子全文+邻居的user消息（`assistant_elided`），被隐去的round不允许作为E。
- `attacker_type_filter: true`：SSA 要求种子有 assistant 回复；multi-session 只检查输入中有至少两个实际可见的 session，个人相关性和跨会话必要性由 gate 验证。其余题型保留 `seed_min_personal` 词表预筛。不能用人称词表未命中来证明 multi-session 不可出题，例如 `I bought…`、`It cost…` 都可能未命中。不可出时该 pack 记为 `type_infeasible`，不调用 attacker、不产生奖励。
- `evidence_novelty`：同一round被反复出题时奖励按 `1/(1+已用次数均值)` 递减；`impersonal_weight`：非SSA题的E里没有一人称个人陈述时乘该权重。
- `attacker_memory_entries` / `attacker_context_items`：控制附带的M条目和已接受题提示数量，设为0即不附带。

```json
{"items":[{"q":"...","question_date":"...","a":"...",
           "type":"multi-session","E":["s1:r1","s3:r1"]}]}
```

默认题型均匀采样，报告明确写 `uniform_not_training_distribution`，不会装成真实训练分布。使用你已划分的训练集统计权重：

```bash
# train_ids.json是你明确选择的训练question_id数组，不由当前目标自动选取。
python run.py weights --data data/longmemeval_s.json \
  --train-ids data/train_ids.json --out data/type_weights.json
# 在attack命令追加：--weights data/type_weights.json
```

后三类 `multi-session / knowledge-update / temporal-reasoning` 只对包含至少两个session的pack抽取；multi-session的实际跨session必要性再交oracle校验。

### Gate

`--gate full`：格式检查 → 必要时全库top30宽检索筛查 → oracle只读E独立作答 → 核对答案、用户价值、题型、题面不泄露 → 空记忆defender应答错。单session证据≤3 round，跨session≤8 round。

全库筛查允许追加证据、重算a，但不改q和日期；超过证据预算则拒绝。原始候选保存在 `generated`，最终候选保存在 `item`，额外可见原文保存在 `validation_rids`。无法可靠补齐时拒绝，不截掉多余证据冒充完备。

`--gate basic` 去掉全库宽检索；`--gate off` 只做格式检查，通过者标`unchecked`，不会计入`accepted_questions`。不开gate用于隔离attacker原始出题能力，不能称为验证通过的问答。

**Top30宽检索不是全历史穷尽检查，无法证明没有遗漏。**本版本没有自动调用强模型全文审计子样本；需要这一实验时，可用保留的原文与问题结果另行人工/全历史审计。没有接builder/defect，所以accepted只是gate通过的问题，不是Mistake Book Q。

### 一次主动检索

追加 `--active-search`：正式出题前attacker最多提出2条查询，每条召回10个新round并入pack。它不知道目标题。这会改变输入证据；报告因此分别保存 `input_pack_coverage` / `expanded_pack_coverage`，不能把这项提升全算作出题模型提升。

### 出题覆盖的评判

所有pack完成生成/gate后才做目标匹配。judge比较：实体、属性、时间点/范围、事件与聚合范围是否相同；只是同主题或使用同一批证据**不算等价**。可用 `--skip-target-match` 跳过该项，省去匹配API调用；此时匹配结果是未知而非失败。

`attack_report.json` 分 `raw_questions`（原始候选）和 `accepted_questions`（gate后候选），均报告随pack数N变化的曲线：

| 字段                                        | 判断                                     |
| ------------------------------------------- | ---------------------------------------- |
| `target_equivalent`                       | 是否已有一道候选问题与官方目标题语义等价 |
| `target_equivalent_and_answer_consistent` | 题目等价，并且候选答案也满足目标题答案   |
| `any_question_all_sessions`               | 是否有一道问题的E涉及全部官方证据session |
| `any_question_all_rounds`                 | 是否有一道问题的E包含全部标注证据round   |
| `question_E_union_all_rounds`             | 所有已生成问题的E合并后是否覆盖证据round |

所有指标各有 `min_observed_prefix_N`。这些N按**处理过的pack数**计，不按问题条数计。源E被gate扩大的收益不会混入raw结果；语义判分仍然是模型估计，不能冒称确定性事实。API错误另计，不当成答错或合法空列表。

输出包括 `input_packs.jsonl`、`expanded_packs.jsonl`、`questions.jsonl`、`ledger.json`、`attack_report.json`。报告、question文件与缓存可能包含官方标签/答案；**不要把整个输出目录当作attacker上下文**。生成器仅读取full原文和pack ID。

## 9. 检索器也可用于你的 M

`Retriever` 的输入统一为 `Document(id,text,date,session_ids,prov,neighbors)`。原文round与M条目走同一检索代码：

```python
from memory import memory_documents
from retrieve import Retriever

# entries由你自己的builder产生。例子文本不是本项目实际训练结果。
entries = [{"id": "m1", "text": "2023-03领养猫Milo；2023-08领养狗Rex。",
            "prov": ["s1:r1", "s3:r1"]}]
m_retriever = Retriever(memory_documents(entries, full), retriever.embedder)
hits = m_retriever.search("我领养过哪些宠物？", k=10)
```

对M进行目标问答可在`answer`命令加`--memory M.jsonl`。只把M的text给reader，**不会通过prov偷偷把完整原文附上**。此时报告中的证据命中标为`provenance_only`，不能把“引用过证据”解释成“保留了所有证据事实”。M没有预建相邻关系时不会伪造结构边。

## 10. 预算、可复现与边界

- `--pack-chars`、`--context-chars`、`--gate-chars`是**可见原文字符数**，不是精确token上限，也未包含JSON转义和system prompt。根据实际模型窗口留余量。种子session超预算会报错，不偷偷截断；其他候选只按完整round剔除并记录。设0取消本地字符限制，不意味着模型支持无限上下文。
- 原文库无损，不等于每次召回/pack无截预算。指标使用裁剪后实际可见的round。索引切块不是删原文；本地embedding按模型tokenizer分块，API embedding默认1200字符块。重排长文按块评分；非常长的重排query只取前192个token，不改变原始问题或F。
- official abstention由`question_id`的`_abs`后缀识别，其证据覆盖置null；不能利用空集合子集规则报100%召回。缺少证据round标签时同样置null。具体是否未找到、标签不适用或判官失败，需要结合对应status和error阅读。
- 一次采样取最大N，读其prefix曲线即可，不必对每个N重跑LLM。多次独立attacker试验使用不同`--seed`及不同输出目录；API nonce区分采样，模型服务是否完全可复现取决于服务自身。
- HTTP缓存键包含服务、请求、模型名及nonce；embedding缓存包含full内容与模型配置。缓存记录原文/响应但不记录API key，仍应当按敏感历史保护。不把`.env`、缓存或原始数据上传仓库。
- 固定本地权重目录/模型快照用于正式实验。更换同名权重或API别名指向后要换cache目录；代码不能推断同名远端模型是否悄悄更新。
- `MAX_API_CALLS`是每个Client实例的网络调用上限（包括重试），不是整个实验总token/费用预算。每个候选可能触发多次oracle/reader/匹配调用，应先以少量pack验证配置再扩大N。
- Entity/alias/type标注层按你的要求暂不开启。没有真实数据结果支撑“8/12/30已足够”“召回一定提高”等结论。

## 11. 实际验证情况

本交付运行了 **40 项离线测试，全部通过**。测试包括无损还原、标签隔离、异常消息保留、chunk→rid聚合、种子覆盖与预算、active search、gate、跨session补证据、最小集合覆盖、abstention、未知/失败状态、完整attack入口的脚本stub管线。见 `VALIDATION.md`。

附带的 `examples/demo_case.json` 是**人工合成**的4-session、5-round宠物示例，**不是官方LongMemEval样本**。`examples/offline_results/`来自实际运行的BM25-only构建、检索、pack实验。合成样例的最小N不能当成LongMemEval结果。

```bash
python run.py build --data examples/demo_case.json --embedding none --out runs/demo
python run.py answer --data examples/demo_case.json --embedding none --out runs/demo --retrieval-only --k 3
python run.py packs --data examples/demo_case.json --embedding none --out runs/demo --n 8 --neighbors 1 --trials 20
pytest -q
```

**没有在交付环境中运行真实LongMemEval_S、远端LLM或真实本地embedding/cross-encoder权重。**因此没有提供虚构的目标正确率、真实case的最小N或attacker覆盖率。代码接口和离线逻辑已测试，模型/网络兼容性及正式研究指标需按上面的命令在实际环境中测量。
## 12. Builder 奖励与 Probe 诊断

`admem/environment.py:builder_score` 是 probe、在线 GRPO 和 VERL 的共同评分入口。
`family` 只用于 probe 汇总，`reward` 和 `effective_reward` 返回相同的通用评分。
合法 NOOP（包括 refine）得 0 分。默认奖励为：

```text
R = 1.0 × gain × all_faithful
    - 2.0 × loss
    - 2.0 × (1 - faith_ratio)
    - 2.0 × information_loss
    + 0.2 × safe_shrink_ratio
    - 0.2 × growth_ratio
    - 0.1 × redundant_operation_count
```

- `gain` / `loss`：同一组测试中，错→对 / 对→错的题数分别除以总题数。即使平均正确率不变，也会独立惩罚遗忘。
- `faith`：`all(faith_scores)`；`faith_ratio` 另行记录。NOOP、纯 DELETE 没有新断言，忠实度视为通过，但 DELETE 必须检查信息保留。`faith_gate: false` 只控制构建时的执行门控，不关闭训练/probe 奖励中的忠实度检查。
- `information_loss`：对 DELETE/UPDATE/MERGE 涉及的旧条目，依据原文列出必须保留的原子事实，再在编辑后的记忆中逐项寻找文本证据。原文支持的纠错使用正确事实，历史上成立的旧状态仍需保留。相同文本的存活副本直接通过；比例的分母来自需要语义检查的事实清单。
- `safe_shrink_ratio`：仅在所有写入忠实、`loss == 0` 且事实保留检查通过时，奖励 `(before_tokens-after_tokens)/before_tokens` 的正值。单纯减少条目数不等于缩短文本。
- `growth_ratio`：增长 token 数除以本轮来源与可编辑旧记忆 token 数的较大者，上限为 1，避免 refine 的空来源导致异常大罚分。
- 冗余判断不使用 probe 类型或目标问答：重复 ADD、仅改来源的 UPDATE、没有信息增益且没有缩短文本的等价改写会被处罚；安全缩短的 UPDATE/MERGE 不视为无效编辑。按冗余操作数扣分，不能靠增加其他操作稀释惩罚。

权重对应配置项 `builder_gain_weight`、`builder_loss_weight`、`builder_faith_weight`、
`builder_retention_weight`、`builder_redundancy_weight`；长度项沿用 `size_weight`。
上述默认权重尚未经过真实数据调优。非法动作的分数低于这些配置下合法动作的理论下界。

事实清单和覆盖检查使用 JUDGE，增加的调用遵循现有缓存机制。覆盖检查按完整条目分批，只有全部必需事实找到证据才提前通过；缺失判定会遍历全部候选记忆，不把检索未命中当作已丢失。不截断待检查的旧条目或原文，预算/接口/返回结构错误记为 Unknown。语义判官仍可能误判；跨批次才能联合推出的同一事实可能被保守判为缺失，因此这不是无损性的形式化证明。

`samples.jsonl` 保存 `reward_components`、`gain/loss`、`faith_scores/faith_ratio`、
`retention_checks`、`redundant_ops` 等明细。`summary.json` 中：

- `faith_rate_all_samples` 是全部尝试中严格忠实的已观测比例；同时报告 `unknown_samples`、已评估数量和 `faith_rate_evaluated`。Unknown 样本的 reward 仍为 null，不作为 0 参与训练或方差计算。
- `success_rate_all_samples` 需要合法、严格忠实、所有测试答对及通用编辑质量通过；refine 的 NOOP 可以满足这些条件。
- `compression_success_rate` 单独统计 refine 是否安全缩短；NOOP 不计为压缩成功。
- 旧字段 `family_constraint_rate` 为通用 `edit_quality_rate` 的兼容别名，不再执行 family 专属约束。

修改奖励代码或配置后重新生成状态（环境指纹会变化），使用新的输出目录：

```bash
python -m admem.cli probe-suite --config configs/type_aware_4b_nogate.json --out outputs/suite_v4
CUDA_VISIBLE_DEVICES=4 python -m admem.cli probe --config configs/type_aware_4b_nogate.json \
  --states outputs/suite_v4/probes.jsonl --samples 8 --out outputs/builder_probe_v8
```

这会调用配置好的模型服务；离线回归测试不代表真实 Qwen Builder 已通过这些场景。

### 12.1 在真实 LongMemEval 历史上运行 Builder reward probe

`scripts/probe_longmemeval.py` 串联现有 prepare、bank、build/collect、export 和 probe。
它使用真实会话按时间顺序产生的 stream 编辑状态，不调用合成 `probe-suite`，也不人为删除/复制记忆制造状态。
奖励题由 ATTACKER 从真实历史生成并经 gate 验证；官方问题答案保留在 `private_eval.jsonl`，不用于奖励。

在配置好 `BUILDER`、`ATTACKER`、`DEFENDER`、`JUDGE` 服务的服务器上，从项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/probe_longmemeval.py \
  --config configs/type_aware_4b_nogate.json \
  --data data/longmemeval/longmemeval_s.json \
  --prepared data/prepared_longmemeval \
  --split val --per-type 2 --states-per-case 16 \
  --samples 8 --temperature 1.0 \
  --out outputs/real_builder_probe_all_types_v1
```

- 首次运行默认按 `--sizes 300 50 150` 准备 500 个 case；大 JSON 的 prepare 需要 `ijson`。已有 prepared 目录会复用原划分，其他规模数据须明确传入匹配的 `--sizes`。
- 默认按六种实际 `question_type` 各选 2 个 case，共 12 个：`single-session-user`、`single-session-assistant`、`single-session-preference`、`multi-session`、`knowledge-update`、`temporal-reasoning`。这里不把 prepare 的 `abstention` 分层算作第七种题型。某一类不足 2 个会在建题库前报错，不会静默少跑。
- `selected_cases.json` 固定实际选中的 case；bank 和 build 使用同一批 key。顶层 `summary.json` 的 `case_counts_by_type` 记录每类 case 数，`state_selection.json` 记录每个 case 的可评分状态数和本次采样数。
- `--states-per-case 16` 表示每个 case 最多选 16 个有 tests 的状态，每个状态采样 8 次。12 个 case 最多为 192 个状态、1536 条候选；某个 case 可用状态不足时全部选取，不把空余额转给其他 case。建题库和构建记忆仍处理所选 case 的全部历史，一个 case 可能包含几十到几百个 session。
- 先过滤无 tests 的状态，再按种子抽样；`probe/summary.json` 和 `probe/samples.jsonl` 保存奖励统计及明细。`family` 按 case 的六种题型汇总，仅用于诊断分组；底层状态仍为 `stream`，奖励公式不随题型变化。
- 若只做单 case 冒烟测试，可改用 `--question-types single-session-user --case-limit 1 --state-limit 16`。`--case-limit` 与 `--per-type` 互斥；全局 `--state-limit` 与 `--states-per-case` 互斥。单独传 `--keys` 时处理全部指定 key，不启用默认每类 2 个的配额。
- 沿用配置中的构建 fallback 设置；fallback 会影响后续状态的旧记忆，但每条 probe 候选仍按模型原始编辑评分，不使用 fallback 替它修复。
- 默认只测自然构建阶段，不保证包含 patch/refine。若需要官方问题上的完整 QA 表现，使用前文的 `longmemeval-eval`；那是独立评测目标。

脚本拒绝覆盖已有输出目录，并在题库失败/为空、没有可评分状态时给出错误，避免把没有测到误报为模型失败。模型参数或源码变更后，重新收集状态到新输出目录。

加速默认启用，不需要减少每类 case 数或修改奖励权重：

- 同一进程的多个记忆索引共享文本切分和 embedding 缓存，记忆变化时优先复用未改动文本、相同问题的向量。`EMBED_CACHE_ITEMS` 默认 `4096`，分别限制切分缓存和向量缓存的条目数；设为 `0` 可关闭这两层内存缓存。原有磁盘索引/API 缓存继续使用。
- 同一个 session 的多条检索 query 每组最多 32 条批量编码，再按原顺序检索和融合；不会把它们拼成一条长 query。
- 每个 probe 状态仍独立生成指定数量的候选；完全相同的原始输出复用已完成的评分，并保留每次采样的日志。旧记忆的答题结果在同状态内复用；失败的评分不会当作成功结果缓存。缓存有界，离开该状态即释放。
- `probe/samples.jsonl` 记录 `score_cache_hit`、`generation_seconds`、`scoring_seconds`，并逐样本显示进度；`probe/summary.json` 的 `performance` 汇总采样/评分次数、评分缓存命中数和耗时。这些次数是客户端逻辑调用次数，不是 HTTP 请求次数；耗时包含缓存读取等客户端工作，不等同于服务端纯推理时间。
- 顶层 `stage_timings.json` 记录 bank/build/export/probe 的状态和耗时；进入阶段即写入 `running`，完成或失败时更新。这些优化保留全部 gate、忠实度和信息保留检查；LLM 请求目前仍串行，实际提速取决于重复比例和服务端延迟。向量缓存仅在当前进程共享，未承诺跨进程复用或真实模型浮点结果逐位相同。

已经有 `states.jsonl`、希望在固定状态上重新评估采样/评分时，可直接跳过 bank 和 build：

```bash
CUDA_VISIBLE_DEVICES=4 python -m admem.cli probe \
  --config configs/type_aware_4b_nogate.json \
  --states outputs/real_builder_probe_all_types_v1/states.jsonl \
  --samples 8 --temperature 1.0 \
  --out outputs/real_builder_probe_only_v2
```

这会沿用旧状态的历史记忆和 prompt；如果要检验新的 Builder 构建过程，仍需重新收集状态。`--limit` 只减少最终 probe 状态总数，不保证各 case 等额抽样。

如果 multi-session 的 bank 只有 0～2 题，先分析已有日志，无需重新调用模型：

```bash
python scripts/attacker_report.py \
  --prepared data/prepared_longmemeval \
  outputs/real_builder_probe_all_types_v1/bank \
  --question-types multi-session \
  --json outputs/real_builder_probe_all_types_v1/multi_session_diagnostics.json
```

报告按 case 区分 `type_infeasible`（预筛跳过）、`empty`（模型未提出问题）、`invalid`（输出格式错误）、`rejected`（gate 拒绝）和服务失败；输出拒绝原因、oracle 未通过的字段及少量示例。兼容旧日志缺少 qtype 的跳过记录，也会单独列出失败 case 的 partial 日志。`nogate` 配置只关闭 Builder 的 `faith_gate`，`gate_mode: full` 仍校验 bank 的答案、跨会话必要性、问题范围和闭卷可答性。每包最多 4 题不是保证 4 题；`neighbors: 4` 指额外召回的 round 数，也不保证找到 4 个有关联的 session。

bank 保存 gate 实际接受的题目；若 gate 修复过证据 `E`，后续评分和建记忆使用修复后的 `E`。旧 bank 不会自动改写，验证修复效果时使用新输出目录。

联合抽查实际输入与 gate 日志：新运行会在每个 case 的 `pack_traces/` 保存逐次尝试的输入快照、原始输出和 gate 结果，`bank_log.json` 用 `trace_file` 关联。`request_body` 保存应用 `EXTRA_BODY` 后的有效请求（含缓存命中）；`prompt` 保存交给策略接口的输入。预筛跳过的 pack 没有模型输入，失败后的重试会分别保留。以下命令按状态/拒绝原因各抽 2 包，不调用模型、tokenizer 或 embedding：

```bash
python scripts/audit_bank_packs.py \
  --bank outputs/real_builder_probe_all_types_v1/bank \
  --prepared data/prepared_longmemeval \
  --cache .cache/admem/api \
  --question-types multi-session \
  --per-status 2 --seed 0 \
  --out outputs/real_builder_probe_all_types_v1/pack_audit
```

先看 `pack_audit/index.json`，再看其中列出的 `<case>/pack_<编号>.json`。每份报告包含同包的 `bank_log`、输入/原始输出、可见文本与原文的对照，以及未展示的前后 round。人工检查：两个会话的事实有何联系、是否缺少前提、隐藏文本是否关键、是否能构造合格问题、gate 的拒绝是否合理；填写 `manual_review` 即可。

旧运行没有 trace 时，脚本可从 API 缓存按历史内容和模型输出查找候选；即使只有一个候选也标为 `cache_candidates_unverified`，不宣称已恢复原始 pack。没有日志关联的快照标为 `trace_candidates_unverified`；有明确关联时，只合并同一输入对应的此前重试，排除旧目录残留的其他尝试。无匹配则标为 `unavailable`，预筛跳过标为 `not_sent`。`debug_bank_pack.py` 会重新构造输入，不带历史时刻的完整已接受题上下文，因此不应用来证明旧运行当时发送了什么。需要指定包时追加 `--keys c0070 --packs 0 3`，编号与 bank 日志一致，从 0 开始。脚本拒绝覆盖已有抽查目录。

## 13. 出题流水线修复（pipeline v2）

新增配置项的默认值都保持旧行为；推荐组合见 `configs/hidden_4b_nogate_v2.json`，它与 `type_aware_4b_nogate.json` 的 Builder/judge 设置相同，只改了 `hint_mode` 和出题相关项。源码改动会改变环境指纹，旧的 states/bank 需要重新生成到新目录。

### 13.1 预算溢出不再伪装成网络失败

- 新异常 `BudgetExceeded`（`Unknown` 的子类，仍不产生负奖励）：attacker prompt、judge、reader、builder prompt、edit coverage 的预算检查都抛它，并带 `stage/tokens/limit`。
- bank：只剩种子仍装不下的 pack 记一行 `status=over_budget, stage=attacker_prompt`，不重试、不计入连续失败，不会再让整个 case 因“连续 5 个 pack API 失败”被放弃。gate 里某道题的 judge 装不下时，只把这道题记为 `over_budget`，同包其他题照常判定。
- attacker prompt 超预算时：先同时去掉 `memory_entries` 与 `audit_context`（旧行为，之前能装下的 pack 得到逐字相同的 prompt）；仍不够则从只含种子开始，按 `pack.rids` 顺序逐个加回邻居，装不下的跳过。被裁掉的 round 记在 state 的 `budget_dropped_rids` 和日志的 `budget_dropped_rounds`，gate 不接受引用它们的 E。
- gate 的宽检索 screen：E∪top-`screen_k` 装不进 judge 预算时，E 完整保留，其余按检索排名放到装满为止，记录在 gate 结果的 `screen_budget`；全部装得下时与旧实现逐字相同。
- closed_loop 中预算溢出的 pack 记为 `over_budget` 并跳过；attacker-probe 中不可评分的样本 reward 为 null，不计入统计。

### 13.2 token 预算的 pack 与事实句邻居（`pack_builder: token`）

- 用 attacker tokenizer 和最长的 system prompt 估算种子长度。session 超过 `(1-pack_neighbor_reserve)×attacker_input_tokens` 时切成多个连续片段（`kind=session_part`），连 system 都装不下的单个 round 记入 bank 目录 `pack_pool.json` 的 `skipped_rounds`，不截断、不静默丢弃。
- `neighbor_mode`：`topical` 为旧做法（种子每条用户发言整段检索 round）；`facts` 从种子用户发言中挑一人称事实句（`I bought a road bike for $900`），逐句检索其他 session 的用户句，RRF 合并后先每个 session 取 1 个 round，再放宽到 `neighbor_session_cap`；`hybrid`（推荐）先 facts，名额不足再用 topical 补齐。LongMemEval 的证据多是任务型对话里顺带提到的个人事实，整段发言作查询时会被对话主题主导。
- 事实句是启发式：必须含一人称，数字/专名/“I bought”“my X”类事件加分，“can you / write / explain”等请求开头扣分；超长缺标点的段落只取一人称词附近的窗口。它只决定检索查询，不决定出什么题，也不读评测标签。`pack.rids` 为种子 round 加按排名的邻居，渲染时仍按时间重排。
- `scripts/pack_coverage.py` 离线比较各组合下，官方跨 session 证据被同一个 pack（按 attacker 跨会话视图和 token 预算）同时看到的比例。它读 `private_eval.jsonl`，只用于比较设计，默认拒绝 test split：

```bash
python scripts/pack_coverage.py --config configs/hidden_4b_nogate_v2.json \
  --prepared data/prepared_longmemeval --split train \
  --modes legacy:topical token:topical token:facts token:hybrid --out runs/pack_coverage_train.json
```

### 13.3 attacker 题型与官方题型解耦（`type_policy`）

| 取值 | 行为 |
| --- | --- |
| `auto`（默认） | 旧行为：`target_type`→`official`，`hidden`→`uniform` |
| `official` | 每个 pack 用 case 官方题型。这是评测标签：bank 会打印警告，`bank_manifest.json` 记 `label_informed_types: true`，只能作诊断 |
| `uniform` | 旧 hidden：按 hash 在可行题型里选一个 |
| `balanced` | `TypeScheduler` 按结构可行性与本 case 已请求槽位的缺口，每包分配 `types_per_pack` 个题型，`questions_per_pack` 平均分给它们；目标比例由 `type_weights` 给出（默认六类均匀）；完全不读官方题型 |

结构可行性：single-session-assistant 需要种子里有 assistant 文本；multi-session 需要跨会话视图下至少两个 session；其余题型需要种子里有 user 文本；开启 `attacker_type_filter` 时再与人称词表预筛取交集。多题型 prompt 用 `slots` 和各题型的 `max_evidence_rounds`；超出计划（题型不在计划内或超出槽位数）的题按题型不符拒绝，单 session 题型只能引用种子 round。bank 日志按每道题的实际题型统计。`hint_mode` 仍只决定 Builder/Reader 是否看到题型提示。

### 13.4 针对当前记忆出题（对抗题库）

```bash
# 第0轮：空M冷启动题库（与旧行为相同）
python -m admem.cli bank --config configs/hidden_4b_nogate_v2.json \
  --prepared data/prepared_longmemeval --split train --out runs/bank_r0
python -m admem.cli run --config configs/hidden_4b_nogate_v2.json \
  --prepared data/prepared_longmemeval --split train --mode build --bank runs/bank_r0 --out runs/build_r0
# 第1轮：针对M_build出题，并入旧题库；--defects-only只保留暴露当前M缺陷的新题
python -m admem.cli bank --config configs/hidden_4b_nogate_v2.json \
  --prepared data/prepared_longmemeval --split train \
  --memory-run runs/build_r0 --memory-snapshot M_build --base-bank runs/bank_r0 \
  --defects-only --out runs/bank_r1
```

- attacker 看得到 `referenced_by_memory` 与（预算允许时）`memory_entries`；旧题进入去重、novelty 计数与 `audit_context`。通过 gate 的题在 `bank_log.json` 记录 `defect`（`missing`/`lossy_or_reasoning`/`unretrievable`/`none`/`unresolved`），`--defects-only` 时不暴露缺陷的题记为 `accepted_no_defect`、不进 bank。
- `scripts/probe_longmemeval.py --bank-rounds 2 [--bank-defects-only]` 自动执行上述轮次，只在最后一轮 build 时收集状态；默认 `--bank-rounds 1` 与旧流程相同。
- `attacker-probe --memory-run runs/build_r0` 让 attacker 奖励里的 defect 项针对真实记忆计算；不传时 summary 会注明空 M 下 defect 项退化为常数。
- 每个 bank case 目录新增 `pack_pool.json`（pack 构成、事实句查询、被切分/跳过的 session 与 round，只含中性 ID）与 `bank_manifest.json`（题型策略、记忆快照与基础题库的 hash）。

### 13.5 划分与日志

- `prepare_longmemeval_cases.py`：`--manifest data/prepared_longmemeval/manifest.json` 直接沿用 admem 的划分，与 admem 产物混用时必须这样做；不传时，完全相同的历史按组内字典序最小的 question_id 一起 hash（单例组的划分与旧版相同），`_abs` 等共享历史的变体不会跨 split。`--select-sessions all` 不用官方证据标签挑选 session（默认 `evidence` 仍是旧行为，报告中标 `label_informed_selection: true`）；`--no-question-meta` 去掉 prompt 中的官方题型/日期；`--max-history-sessions K` 只内联最近 K 个 session。
- legacy pack 构建跳过的超长种子 session 写入 `pack_pool.json` 的 `skipped_sessions`；closed_loop 的 `ledger.json` 只记中性 `sN`，不再写原始 session ID。
- `scripts/attacker_report.py` 单独统计 `over_budget`、`accepted_no_defect` 与 defect 类别；`audit_bank_packs.py` 把 prompt 阶段溢出的 pack 标为 `not_sent`。

## 14. 奖励、提示词与其他修复（v2）

三个开关 `attacker_reward`、`builder_reward`、`builder_prompt` 默认 `v1`（旧实现逐字不变，旧实验可复现）；`configs/hidden_4b_nogate_v2.json` 与它的题型已知对照 `configs/type_aware_4b_nogate_v2.json` 全部设为 `v2`。修改后环境指纹会变，状态和题库要重新生成到新目录。纯算术部分在 `admem/rewards.py`（不依赖 torch）。

### 14.1 Attacker 奖励（`attacker_reward: v2`，`Environment._attacker_score_v2`）

| 情况 | v1 | v2 |
| --- | --- | --- |
| 被 gate 拒绝 | -0.1~0.45（4道 closed-book 被拒得 +0.35，高于一道合格题） | ∈[-`attacker_reject_penalty`, 0)，按 gate 检查顺序越晚失败扣得越少，总是低于空列表（0） |
| 整个输出非法 | -0.1 | -(penalty+0.1)，低于任何合法输出 |
| 重复题 | 字面重复 0 | 字面或语义重复都给小的负值 |
| novelty | 按 E 的平均使用次数衰减（往 E 里塞新 round 能拉回权重） | 按 E 中被问得最多的 round 衰减 |
| 证据填充 | 无检查 | 新的 evidence-audit judge 返回 `necessary_rids`，权重乘以必要比例 |
| “与用户无关”降权 | 一人称词表（"I bought…"判成0、"Fix my code"判成1） | 同一次 audit judge 的 `personal` 字段 |
| gate 修补过的 E | 全额 | 乘 `attacker_repair_credit`（默认0.5） |
| 空 M | 每题都算 missing（defect 项是常数），仍调用 reader | 跳过 defect 计算，value=1；M 非空时仍为 0.5+0.5×可修复 |
| 某题 judge 预算溢出 | 抛错 | 该题记 `over_budget`、计 0 |

v2 的 attacker 提示词末尾附加 `ATTACKER_SCORING_NOTE`，说明上面三条规则（少出题优于出错题、E 不加无关 round、换说法重复无分），使提示词与奖励一致。每道通过 gate 的题多一次 judge 调用（`EVIDENCE_AUDIT_SYSTEM`）。

### 14.2 Builder 奖励（`builder_reward: v2`，`Environment._builder_score_v2`）

- **跨 session 证据的前瞻题**：stream 窗口除旧条件（E 已读完且在窗口或旧条目 prov 内）外，E 与本窗口已读完 round 有交集的题也成为测试题，带 `context_rids`=E 中不在本窗口的 round。评分时这些 round 的原文在编辑前后都补进记忆，所以只衡量本窗口那部分证据是否被存下：之前漏存、或其余证据还在未来 session 里的题，也能给本窗口记分或扣分。SFT 示范的 `all_pass` 同样补 context。
- **归一化**：gain/loss 除以“目标题数”（与 x 有证据交集的测试题，至少1），不再被回放题数稀释，二者都截断到[0,1]。同一个正确写入在1题和12题状态下得分相同；回放题被改错照样按 `builder_loss_weight` 扣。
- **翻转复核**：k 下答案翻转时再用 2k 读取编辑前后各一次，取两次变化的平均；只是 top-k 排名被挤动的翻转只算一半。NOOP 仍不重新作答，合法 NOOP 仍为 0。
- **覆盖率项**（`builder_coverage_weight`，默认0.5）：judge 列出 x 中值得存的原子事实（`SOURCE_INVENTORY`），排除编辑前 Builder 可见条目已覆盖的，奖励新事实被存下的比例；需全部写入忠实。它不依赖题库，题库没问到的事实也有信号；没有测试题但有 x 的 stream/patch 状态因此也能评分（state 标 `coverage_scorable`，训练、导出、probe 不再丢弃这些窗口）。
- **冗余**：只对照 Builder 在 prompt 里看得到的条目（M_old 全文+目录摘要），最多计 `builder_redundancy_cap` 个操作。
- **非法输出与 GRPO 尺度**：v2 的合法下界有界（loss、faith、retention 各≤1×权重，growth≤0.2，冗余≤cap），非法输出取下界再减1。`admem/train.py` 在该角色使用 v2 奖励时默认：组内非法样本改为“最低合法奖励−合法std”（`--shape-invalid`），优势归一化 std 下限 0.1（`--std-floor`），避免一个非法样本压扁合法样本间的差别、或只差长度项的组被放大。VERL 按样本独立打分，做不到组内整形，只能靠 SFT 先降低非法率。

### 14.3 提示词（`builder_prompt: v2`）

- `BUILDER_V2`：约为旧版的一半长度，规则合并成一份规格。无论有没有 `task_type`，具体的 assistant 产出（推荐、列表项、数量、链接、创作细节）都存，并标明归属；题型提示改为一句话的 `BUILDER_TYPE_GUIDANCE_V2`。hidden 设定下 single-session-assistant 题不再因提示词不存 assistant 内容而必然失败。
- **卡片日期由 host 负责**：编辑时卡片按 prov 记录 `dates`（最早/最晚 session 日期），reader 看到 `[Source date: …] 文本`。Builder 只需把相对时间换成事件日期；v1 的 temporal 附加段与 patch 前言在 v2 下不再拼接。
- 去掉 `write_required` 与“第一个窗口禁止 NOOP”的硬规则；漏存由覆盖率项衡量。
- `stored_catalog` 先列与 x 相关的条目（不止 `old_entries` 个），其余最新优先，已在 M_old 里全文给出的不再重复（v1 是最旧的约30条）。
- `prompt_hashes()` 记录 gate 实际使用的 `prompts.SCREEN`（之前误记为未使用的 `GATE_SCREEN_SYSTEM`），并加入 Builder/reader/grade/faith/audit 提示词。
- 未改：attacker 题型指引是看过全部500道 LongMemEval 题后写的，生成的题会模仿测试集问法；需要新的、未参与设计的数据才能真正解决。

### 14.4 其他

- **配置配对**：`type_hidden.json` 现与 `type_aware.json` 只差 `hint_mode`（旧版在 embedding、Builder 预算、`old_entries`、`entry_tokens`、`max_ops` 上都不同，保留为 `type_hidden_legacy.json`）；`type_hidden_4b.json` 改用与 `type_aware_4b.json` 相同的本地 tokenizer 路径；新增 `type_hidden_4b_nogate.json`。测试检查每对配置只差 `hint_mode`。
- **训练**：`admem/train.py` 的 `--max-prompt-tokens` 默认取该角色的输入预算+256（旧默认8192会悄悄丢掉大记忆状态），被跳过比例超过 `--max-skip-fraction`（默认5%）就报错；单个 group 的 judge/服务失败只跳过该组，连续 `--max-env-errors`（默认3）组失败才中止，预算溢出的状态不再抽取。`scripts/train_verl.sh` 设 `CONFIG=<config>` 时由输入预算推导 `max_prompt_length`。`admem/verl_reward.py`：`ADMEM_REWARD_ON_UNKNOWN=neutral` 时环境失败返回 0（两个角色的合法 NOOP/空列表奖励），写入日志，超过 `ADMEM_MAX_UNKNOWN` 次仍报错；默认仍为抛错。
- **自己判自己**：BUILDER/ATTACKER 与 JUDGE/DEFENDER 解析到同一模型服务时，CLI 和 `probe_longmemeval.py` 打印警告，评测 summary 与 probe summary 记录 `self_judging`。
- **测试集不独立**：评测 summary 加 `benchmark_informed` 说明。真正的独立评估需要另一个未参与提示词设计的基准，代码无法补救。
- **对抗协同训练**：`--bank-rounds` 与针对 M 出题（第13.4节）实现了题库与记忆的交替，但 Builder 与 Attacker 的交替 RL 训练循环仍需另行编排。
