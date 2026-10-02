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

attacker 输入只有：原文pack、M provenance标记、题型、case允许使用的 `question_date`。不传目标问题/答案/当前case的目标题型/证据标签。`question_date`逐字复制；题型使用官方名称，并接受设计中的 `preference` / `temporal` 别名。

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
# advmem
# advmem
# advmem
