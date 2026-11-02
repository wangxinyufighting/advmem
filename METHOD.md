# 方法：对抗式记忆编辑（Adversarial Memory Editing）

本文档按代码实现拆解整体算法，供论文"方法"一节直接取用。所有符号、顺序、阈值、默认值均与仓库代码一致；模块映射见文末第 17 节。

---

## 0. 符号与总体流程

| 符号 | 含义 | 代码实体 |
| --- | --- | --- |
| $\mathcal{H}$ | 一个 case 的无损全历史（原文库） | `FullMemory`，`memory.py` |
| $r$ | round（一个 user 消息 + 紧随的 assistant 消息），稳定 ID `s{i}:r{j}` | `Round.rid` |
| $x_t$ | 第 $t$ 步的源窗口（$x_t\subseteq \mathcal{H}$） | `windows()` 产出 |
| $M_t$ | 第 $t$ 步的**可编辑记忆**（记忆卡片集合） | `memory` 列表，元素 `{id,text,prov,kind,dates}` |
| $P$ | pack：种子 session + 检索来的邻居 round | `Pack` |
| $Q$ | 自生成题库（经 gate 接受的题），$Q=\{(q,a,\text{type},E)\}$ | `bank.json` / `qbook` |
| $\pi_B,\pi_A$ | Builder（记忆编辑策略）、Attacker（出题策略） | `BUILDER` / `ATTACKER` role |
| $\mathcal{R}_B,\mathcal{R}_A$ | 两个角色的奖励 | `Environment.builder_score` / `attacker_score` |
| $D,G,J$ | Reader（答题）、Gate/Oracle、Judge，全部冻结 | `DEFENDER` / `JUDGE` |

整体是一个交替闭环：**Attacker 针对当前 $M$ 出题 → Gate 校验 → 用题测出 $M$ 的缺陷 → Builder 修补 → Refine 压缩**，两个策略用 GRPO 训练，奖励由冻结的 Reader/Judge 计算。

```
H ──(windows)──> x_t ──(Builder: stream)──> M_build
M_build ──(pack + Attacker)──> 候选题 ──(Gate)──> 接受题 Q
Q ──(defect 判定)──> 可修复缺陷 ──(Builder: patch)──> M_audit
Q 全解 ──(Builder: refine)──> M_final
```

---

## 1. 问题形式化

给定无损历史 $\mathcal{H}$ 与一个**永不暴露给策略**的目标问题集 $\mathcal{Q}^\star$（官方 q/a/证据标签），学习

$$
\pi_B^\star=\arg\max_{\pi_B}\ \mathbb{E}_{M\sim \pi_B(\mathcal{H})}\Big[\underbrace{\mathrm{Acc}\big(D(M),\,\mathcal{Q}^\star\big)}_{\text{可答性}}\Big]\quad
\text{s.t.}\quad M \text{ 忠实于 } \mathcal{H},\ \ |M|\ll|\mathcal{H}| .
$$

由于 $\mathcal{Q}^\star$ 不能进入训练回路，实际优化的是**自生成题**上的代理目标：

$$
\max_{\pi_B}\ \mathbb{E}_{M}\Big[\underbrace{\mathrm{Acc}\big(D(M),Q(M)\big)}_{\text{记忆可用}},\ -\underbrace{\mathrm{Loss}(M)}_{\text{遗忘}},\ -\underbrace{\mathrm{Unfaith}(M)}_{\text{不忠实}},\ -\underbrace{\mathrm{Redund}(M)}_{\text{冗余}}\Big],
$$

其中 $Q(M)$ 由 Attacker 针对 $M$ 生成并经 Gate 验证。Attacker 的目标是生成"$M$ 答不对但原文能答对"的题：

$$
\max_{\pi_A}\ \mathbb{E}\big[\ \text{accepted}\cdot\text{novel}\cdot\text{necessary}\cdot\text{repairable}\ \big].
$$

形成一个 minimax 式对抗：Attacker 找缺陷，Builder 修缺陷；题库是 Mistake Book，历史题通过 replay 防止遗忘。

---

## 2. 数据准备与标签隔离

`admem/data.py`

1. **历史白名单**：`history_for_builder(case)` 只保留 `haystack_session_ids / haystack_dates / haystack_sessions`；重复 session ID 仅对内部键加 `__duplicate_k` 后缀，文本不变。
2. **构建 $F$**：`FullMemory.build` 剥离消息级标签 `has_answer / answer_session_ids / question_type / question_id`（`TURN_LABELS`），并断言 `export_history() == history_only(case)`（结构 + 逐字文本）。
3. **私有标签分离**：官方 $q,a$、`answer_session_ids`、证据 round 写入 `private_eval.jsonl`，只由独立 `evaluate` / `longmemeval-eval` 读取；`context.json` 只含 `key/question_type/question_date/full_hash/split`（不含 `question_id`、`_abs`）。
4. **划分**：先按"完全相同历史"分组（`history_group`：session 日期 + `role/content` 的哈希），同组不拆；再按 `(300,50,150)` 逐组贪心分配，代价函数同时惩罚 split 总规模的偏差与题型分布的偏差（$\Delta$ 平方误差归一化）。跨 split 的 session-Jaccard $\ge 0.8$ 记入 `near_duplicate_cross_split` 供人工检查。分组不使用 `answer_*` 或证据标签。
5. **abstention**：由 `question_id` 后缀 `_abs` 判定；覆盖类指标对其置 `null`，不用空集子集规则伪造 100%。

---

## 3. 无损全历史 $F$ 的构建

`memory.py`

- **Token 化**：round = user 消息 + 紧随其后的 assistant 消息（异常排列保留为单消息 round，孤立消息不丢）。ID 由结构确定性生成：`s{i}:r{j}`，其中 `i` 为原始 session 序号、`j` 为 round 序号。
- **双序**：`original_index`（原始顺序）与 `sort_date = date_key(date)`（时间顺序）分开存储；渲染/检索一律用时间序。
- **边不存储**：`contains / next_round / next_session` 由顺序索引确定性导出（`structural_edges`），避免冗余图。
- **统一接口**：`Document(id,text,date,session_ids,prov,neighbors)` 同时表示原文 round 与 $M$ 条目，检索代码只有一套。
- **指纹**：`fingerprint = sha256(export_history())`。所有 pack/state/snapshot 都带指纹，跨版本（如 `longmemeval_s` vs `_cleaned`）拒绝混用。
- **不静默截断**：任何预算不足都抛 `BudgetExceeded`（`Unknown` 子类）或整块丢弃并记录，绝不裁剪原文。

---

## 4. 检索底座

`retrieve.py`

对文档集合（round 或 $M$ 条目）建两路召回：

1. **BM25**：文本为 `date + "\n" + text`，$k_1=1.5,b=0.75$。
2. **Dense**：长文本按 tokenizer（本地模型）或 1200 字符（API）切块；块向量检索后按 `max` 聚合回 round（max-sim）；round 平均向量仅用于可选主题聚类。
3. **融合**：RRF，$s(\text{id})=\sum_{\text{channel}}\frac{1}{60+\text{rank}}$，两路各取 $\max(k,\text{candidate\_k})$ 名。
4. **结构扩展**（可选 `expand`）：沿 round 前后邻接边扩展有限步，新节点得分 $0.5\times$ 父得分，通道记为 `adjacent_round`；只进候选集，是否入 top-$k$ 由融合/重排决定。
5. **可选重排**：cross-encoder，对候选按 query-document 打分重排；长 query 截前 192 token。
6. **多查询**：`search_many` 对每条用户发言分别检索（先批量编码，最多 32 条/组），再对命中做二次 RRF 并集排序，绝不把整个 session 拼成一条长 query。
7. **可见性**：`context()` 只按**完整 document** 计入预算，超预算整条跳过；指标使用实际可见命中，不用候选池。

> 只做 BM25 是显式消融（`--embedding none`），不冒充混合检索。

---

## 5. Pack 构造

`admem/packing.py`（`pack_builder="token"`，推荐）与 `packs.py`（legacy，字符预算）。

**种子**：按时间序遍历 session；用 attacker 真实 tokenizer 与**最长 system prompt** 估算 prompt 长度（`_SeedCost`：空 payload 基数 + 各 round JSON token 数），把 session 的 round 顺序装成不超过

$$
L_{\text{seed}} = (1-\rho)\cdot L_{\text{attacker}},\qquad \rho=\texttt{pack\_neighbor\_reserve}=0.3
$$

的连续片段；估算超限时用精确计数二分（`_verify`）。单个 round 超过 $L_{\text{seed}}$ 但不超过 $L_{\text{attacker}}$ 时独占一片段；超过硬上限则记入 `skipped_rounds`（不截断、不静默丢弃）。切片数 >1 的 session 记为 `session_part` 并写入 `split_sessions`。

**邻居**（`neighbor_mode`）：

| 模式 | 做法 |
| --- | --- |
| `topical` | 种子的每条 user 发言整段检索其他 session 的 round（`search_many`，排除本 session） |
| `facts` | 从种子 user 发言中抽一人称事实句（启发式打分：含一人称、数字/专名/事件动词加分，"can you/write/explain"等请求开头减分；超长无标点段落只取一人称词附近窗口），逐句检索其他 session 的**用户句**索引（文档 ID `rid#k`），RRF 合并后映射回 round |
| `hybrid` | 先 `facts`，名额不足再用 `topical` 补齐 |

**去集中**（`diversify`）：按排名取 $k$ 个 round，第一轮每 session 最多 1 个，之后逐轮放宽到 `neighbor_session_cap=2`，避免一个长 session 独占名额。

输出 `Pack(pack_id, full_hash, seed_id, kind, sweep, seed_rids, rids, dropped_rids, searches)`；`rids = seed_rids + 邻居`（渲染时仍按时间重排，故顺序只决定超预算时谁先被去掉）。`pack_pool.json` 记录构成、事实句查询、被切分/跳过的 session 与 round（仅中性 `sN`/`rid`）。

**必守规则**：种子原文不可截断（超预算直接报错）；额外 round 只能整条丢弃并记录；`check_pack` 校验指纹、rid 合法性与种子完整性。

---

## 6. Attacker：输入视图、题型策略、预算

`admem/prompts.py:attacker_messages` + `admem/audit_prompts.py:build_payload`

### 6.1 模型可见的 payload

```
type / question_date / n_questions / max_evidence_rounds
rounds: [{rid, session(sN), session_date, is_seed, referenced_by_memory,
          prior_questions, messages[], assistant_elided?}]         # 按时间排序
memory_entries: [{id, text, prov}]                                  # 可选
audit_context: {accepted_items[], omitted_relevant_items}           # 可选
slots: {qtype: 槽位数}                                              # 多题型计划时
```

- `view="compact"`：单 session 题型只给**种子**；跨 session 题型给种子 + 含 user 文本的邻居，非种子 round 隐去 assistant（`assistant_elided=true`）。被隐去的文本**不能被引用为证据**。
- `referenced_by_memory`：该 round 被哪些 $M$ 条目引用（仅提示"可能已压缩"，不证明事实保留）。
- `prior_questions`：该 round 已被多少已接受题引用（novelty 的局部提示）。
- `audit_context`：只含本 case **自生成且经 gate 接受**的题（与当前可见 round 有交集，最近 $k$ 条）；不含目标题元信息（严格键白名单）。
- `max_evidence_rounds`：单 session 题型 3，跨 session 题型（`multi-session / knowledge-update / temporal-reasoning`）8。

### 6.2 题型策略（`type_policy`）

| 取值 | 行为 |
| --- | --- |
| `auto` | `target_type`→`official`；`hidden`→`uniform`（兼容旧行为） |
| `official` | 用 case 官方题型（**标签泄漏**，仅诊断；`bank_manifest.json` 记 `label_informed_types: true`） |
| `uniform` | 按 `hash(seed, identity)` 在可行题型中均匀选一个 |
| `balanced`（主设定） | `TypeScheduler` 逐包分配多题型 |

`TypeScheduler.plan`：候选 = 结构可行 $\cap$ 权重>0 的题型；取 `types_per_pack=2` 个题型，`questions_per_pack=4` 个槽位平均分配（余数给先选者）。题型选择按缺口

$$
\text{priority}(t)=\underbrace{\big(w_t\,(\textstyle\sum \text{requested}+n)-c_t\big)}_{\text{配额缺口}},\ \text{tie-break}=\text{hash}(seed,identity,t)
$$

贪心取最大。计数只统计"请求的槽位"，与模型输出、gate 结果无关；同一 identity 重复调用返回同一计划（可重试且不重复计数），因此**从不读取官方题型**。

**结构可行性**（`structural_types`，不依赖人称词表）：`single-session-assistant` 需种子里有 assistant 文本；`multi-session` 需跨会话视图下 $\ge 2$ 个 session；其余需种子里有 user 文本。开启 `attacker_type_filter` 时再与人称词表预筛取交集。不可行 → 该 pack 记 `type_infeasible`，**不调用模型、不产生奖励**。

### 6.3 预算回退（确定性，不截断原文）

attacker prompt 超过 `attacker_input_tokens`（默认 12288）时依次：

1. 同时去掉 `memory_entries` 与 `audit_context`（保证"以前能装下的 pack 得到逐字相同 prompt"）；
2. 从"只含种子"开始，按 `pack.rids` 顺序（即检索排名）逐个加回邻居，装不下的**跳过**并记入 `budget_dropped_rids`；
3. 只含种子仍超限 → `BudgetExceeded(stage="attacker_prompt")`，该 pack 记 `over_budget`（不重试、不计连续失败）。

多题型计划下若预算裁掉了所有邻居导致可见 session < 2，则去掉 `multi-session` 槽位重新渲染。

---

## 7. Gate：五阶段校验 + 一次修复

`Environment.gate → _gate_once`，`gate_mode="full"`

按固定顺序执行，任一步失败即 `rejected` 并记录 `reason`：

| # | 阶段 | 判据 | 失败 reason |
| --- | --- | --- | --- |
| 1 | **Schema/证据合法性** | 键集严格等于 `{q,a,type,question_date,E}`；四个字符串字段非空；`type` 与请求一致且在 `TYPES` 内；`question_date` 逐字一致；`E` 非空、无重复、长度 $\le$ cap(3/8)、$\subseteq$ pack.rids；多题型计划下单 session 题的 `E` 必须 $\subseteq$ `seed_rids` | 各类 `InvalidAction` |
| 2 | **Oracle 独立作答** | Judge 只读 `q, question_date, type, E_history`（**不给候选答案**）独立作答，返回 `answerable / user_relevant / type_valid / no_answer_leak`，全部为真；`multi-session` 另加硬检查：$E$ 覆盖 $\ge 2$ 个不同 session | `oracle_or_type` |
| 3 | **答案支持性** | Judge 对齐 `(q,a,oracle.answer,E_history)`：候选 $a$ 必须回答所问，且每条历史断言都有 $E$ 支持；独立答案只需在必要事实上一致（不罚其省略候选多说的细节） | `unsupported_answer` |
| 4 | **宽检索筛查**（仅 `full` 且"宽范围"题型/措辞） | 触发条件：题型 $\in\{$multi-session, knowledge-update, temporal-reasoning$\}$ 或题面含 `all/total/latest/current/ever/so far`。取 $E\cup\text{top-}30$（`screen_k`）作为可见历史，Judge 判断 `stable`，并返回 `additional_rids`（应补入 $E$）与 `wrong_rids`（应删除） | `scope_or_completeness` |
| 5 | **闭卷可答** | 用空历史让 Reader 作答，若仍判对 → 题目无需记忆 | `closed_book` |

**超预算处理**：$E\cup$ top-30 装不进 judge 预算（默认 30000 token）时，$E$ **完整保留**，其余按检索排名逐个放入直到装满，装不下的跳过并记 `screen_budget{candidates,kept,omitted,limit}`；若 $E$ 本身装不下 → `BudgetExceeded`。

**一次修复尝试**（`gate` 外层）：若因 `scope_or_completeness` 被拒且 screen 返回了合法的 `wrong_rids/additional_rids`，则令

$$
E' = (E\setminus \text{wrong\_rids})\cup \text{additional\_rids}
$$

重跑整条校验链；只有修复后被接受才接受，并标 `repaired=true`（下游按 `repair_credit` 折减奖励，且不把 gate 修复结果当作策略的 SFT 输出）。修复不改变 $q$、不改变日期，也不截断证据。

**有效性附加检查**（`require_reader_answerable=true`，主设定）：通过 gate 的题还要满足"**Reader 仅凭 $E$ 原文就能答对**"（$E$ 原文全量入上下文；装不下才退化为检索）。这是"测试题必须可解"的类比约束：否则 Builder 永远修不好这道题，对抗训练会漂向不可解题。

---

## 8. 缺陷判定（Defect）

`Environment.defect(full, memory, q)`

1. $M$ 已答对 → `none`（无缺陷，不产生 patch）。
2. $M$ 答错，但把 $E$ 原文补进记忆后答对 → 是可修复缺陷，按证据是否在 $M$ 的 `prov` / 可见集合中细分：

| 类别 | 条件 | 含义 |
| --- | --- | --- |
| `missing` | $E \not\subseteq \mathrm{prov}(M)$ | 事实从未被存入 |
| `lossy_or_reasoning` | $E\subseteq\mathrm{prov}(M)$ 但不在被检索到的条目里 | 存了但被压缩丢失，或需组合推理 |
| `unretrievable` | $E\subseteq\mathrm{prov}(M)$ 且可见，仍答错 | 检索/组合层面失败 |
| `retrieval_composition_candidate` | $M$+原文都答错，但**只给原文**能答对（`defect_mode="direct"` 及 validity 路径） | $M$ 中的噪声挤掉了证据 |
| `unresolved` | 两者都答错 | 题目无效或超出能力，**不可修复** |

3. `REPAIRABLE_DEFECTS = {missing, lossy_or_reasoning, unretrievable, retrieval_composition_candidate}`；只有这些题进入 Mistake Book 并触发 patch；其余只记录。

`validity_defect`（`require_reader_answerable` 下）：$M$ 非空时先算 defect；若为 `unresolved` 再看"只给原文能否答对"——能答对则升级为 `retrieval_composition_candidate`（有效、可修），不能答对则**题目无效**，在 v2 奖励中以 `reader_unanswerable` 阶段记负分，bank 不收。

---

## 9. Builder：动作空间、输入与执行语义

### 9.1 动作空间

`admem/store.py:apply`，输出必须是**恰好一个 JSON 对象** `{"ops":[...]}`，`len(ops) ≤ max_ops=32`：

| 操作 | 必填字段 | 语义 |
| --- | --- | --- |
| `ADD` | `text, prov` | 新建卡片，ID 由 host 分配 `m_<hash>` |
| `UPDATE` | `id, text, prov` | 替换可见旧条目 |
| `MERGE` | `ids(≥2), text, prov` | 合并多个旧条目为一个 |
| `DELETE` | `id` | 删除 |

**并行语义与硬约束**（先全部校验，再返回新副本；模型不能直接改文件）：

- 一次输出内**同一旧 ID 最多被触碰一次**，不能引用本次新建的 ID（禁止链式依赖）；
- `id` 必须属于 `editable_entry_ids`；`ADD` 的 `prov` 非空、无重复，且 $\subseteq$ `allowed_source_rids` $\cup$ 可见旧条目的 `prov`（允许补存早期窗口漏掉的事实）；`UPDATE/MERGE` 的 `prov` 是"新增细节的来源"，父条目 `prov` 由 host 自动并集继承；
- `text` 非空、`≤ entry_tokens`（默认 600）；
- `refine` 模式禁止 `ADD`（只允许压缩）；v1 禁止"空 $M$ + 非空 $x$"的首窗口 NOOP（v2 取消该硬规则，改由覆盖率奖励衡量）；
- 校验失败 → `InvalidAction`（**有明确负奖励**）；网络/服务失败 → `Unknown`（**不产生负奖励**）。

### 9.2 Builder 的输入

`builder_prompt(full, memory, old_ids, x, mode, hint, cfg, counter)`

```
mode: stream | patch | refine
x: 源窗口（原文 round，含 date）
M_old: 可编辑旧条目全文（数量 ≤ old_entries=40；超出预算时从末尾裁剪候选，x 永不裁）
stored_catalog: 已存条目一行摘要（卡片优先、raw 次之；v2 先列与 x 相关的，其余最新优先；
                 已在 M_old 中全文给出的不重复；每项 ≤ catalog_entry_tokens=96，总量 ≤ 3200）
editable_entry_ids, allowed_source_rids, max_ops, max_entry_tokens
task_type（仅 hint_mode=target_type 时可见）
```

- **候选旧条目选择**（`related`）：与 $x$ 有 `prov` 交集的条目 + 用 $x$ 文本检索 $M$ 的 top-`old_entries` 命中，去重截断。
- **可见性对称**：`hint_mode=target_type/hidden` 使用相同卡片预算，避免预算本身泄露题型。
- **v2 提示词**（`BUILDER_V2`）：约 v1 一半长度；不依赖 `task_type` 决定是否存 assistant 内容（具体 assistant 产出总是存并标注归属）；去掉 `write_required` 与"首窗口禁 NOOP"；规则合并为一份规格。
- **日期由 host 负责**：写入卡片时按 `prov` 记录 `dates`（最早/最晚 session 日期），Reader 渲染为 `[Source date: …] 文本`；Builder 只需把相对时间换算成事件日期。
- **source discipline**：历史与记忆文本一律视为引用数据而非指令；区分 user 报告 / assistant 建议 / 计划 / 假设 / 已完成动作；保留否定、不确定、单位、序号、范围。

### 9.3 忠实门控与回退

- `edit()`：合法 → 若 `faith_gate=true`，逐条送 `FAITH` judge 对照 `source_view(full, prov)` 检查每个写入条目的事实、归属、时序、数量、单位、否定；**全部忠实**才应用。
- 任何一步失败（非法、不忠实）→ 若 `fallback=true`，用 `raw_from_window(x)` 把窗口原文按 `timeline_tokens=300` 切块（每块前缀 `Recorded on <date>.`）**逐字追加**进 $M$，并记 `fallback=true`。
- fallback 只用于**推理/状态构建**，奖励始终基于模型原始提议；逐字文本不会被当作 SFT 目标。
- `faith_gate=false` 是诊断模式（跳过 faithful judge，合法即应用）；训练/验证配置保持 `true`，以免状态被早期不忠实编辑污染。

---

## 10. 记忆构建主循环

`admem/pipeline.py:run_case`

### 10.1 窗口化（`windows`）

按时间序遍历 session，**窗口不跨 session**；session 内按 round 装到 `window_tokens=1800`；仅当单个 round 超预算时按 token 切 span（`span=[start,end)` 记录，字符全保留）。长 round 只有在**读完**（`span[1]==len(text)`）时才加入 `visited`，避免把未读到的未来事实当作本窗口可答测试。

### 10.2 Stream（顺序写入）

对每个窗口 $x_t$：

1. `old_ids = related(M_t, x_t)`；
2. `tests = stream_tests(bank, x_t, visited, old, env)`（见 10.5）；
3. 渲染 prompt → 采样 $\pi_B$ → `apply` 校验 → faith gate → 应用或 fallback；
4. 记录 `checkpoint`（`M/Q/accepted/events`，含每个窗口的 `checks/fallback/ops/M_tokens`）；
5. 收集 SFT 状态：仅当"合法 + 全部测试答对"（无测试时至少 1 个 op）才把原始输出记为示范。

### 10.3 Audit（出题）+ Patch（修补）

每个 sweep 遍历 pack 池（`sweeps=2`）：

1. **Attacker 出题**：`attacker_state`（题型计划、视图、预算回退）→ 采样 $\pi_A$ → `attacker_score`（内含逐题 gate）；
2. **入库**：仅 `status=accepted` 的题；按 `question_key = hash(normalized q, question_date, type)` 去重（**不按 $E$ 去重**，同一 round 的不同属性可出多题）；
3. **缺陷判定**：对每题在当前 $M$ 上重算 `defect`（因为同包前一题可能已改动 $M$）；
4. **可修复才补**：`defect.kind ∈ REPAIRABLE_DEFECTS` → 加入 `qbook`（`fails+1`，`last_retry_sweep`）；
5. **Patch**：$x = \mathrm{source\_view}(full, E)$，`tests = [q] + replay(qbook, 12)`；Builder 以 `mode=patch` 编辑；接受条件 `patch_ok = 编辑合法 ∧ all_pass(proposal, protected ∪ {q})`（`protected` 为已解决题，防止修新忘旧）。不满足则 fallback（原文追加，且同样需 `all_pass`）；仍失败则不改动；
6. **每轮把已解/未解状态写回 qbook**；旧 active 题每轮最多重试一次（不因去重永久失去修复机会）。

### 10.4 Refine（压缩）

条件：`qbook` 非空 **且全部 resolved**（禁止 vacuous 全通过）。对每个 $M$ 条目作为种子（连同其 top-`old_entries` 邻居作为可编辑集），`mode=refine`，接受条件：

$$
\text{keep}= \big(\text{token}(M')<\text{token}(M)\big)\ \wedge\ \text{all\_pass}(M',Q)
$$

即"更短且不丢任何已解题"。否则不改。每 sweep 执行 `refine_rounds=1` 遍。

### 10.5 测试题选择（$\Delta$ 可答性）

- **v1**：$E$ 已读完，且 $E\subseteq \text{visited}$ 且 $E\subseteq$（本窗口可见 $\cup$ 待编辑旧条目的 `prov`）。
- **v1 额外**：跨 session 题只有在"最后一块证据出现且之前证据已存进 $M$"时才成为测试题。
- **v2 前瞻题**（`stream_tests`）：$E$ 与本窗口已读完 round 有交集但不满足上述条件也可成为测试题，带 `context_rids = E \setminus 可见`。评分时把 `context_rids` 对应原文**在编辑前后都补进**记忆（`with_context`），因此只衡量"本窗口那部分证据有没有被存下"——之前漏存、或其余证据还在未来 session 的题也能给本窗口记分/扣分。
- 排序：与窗口有交集的题优先；上限 `replay_questions=12`。
- **replay 采样**：一半取 `fails` 最高的题（防忘已解决题），一半按 `hash(identity, question_key)` 确定性随机（保多样性）。

### 10.6 终止

每 sweep 结束时若 `digest(M)` 未变则提前结束；最多 `sweeps=2`。输出 `M_build / M_audit_k / M_refine_k / M_final / Q.json / ledger.json`。

### 10.7 状态收集与可恢复性

- `Collector.save`：`id = digest(state \ full_path)`，记录 `{id, state, prompt, completion?, demo_quality?}`；奖励元数据（tests/full_hash 等）只在 `state` 里，导出的 actor prompt 不含 tests。
- 每个 case 先写 `run_config.json` 签名（full 指纹、context、config 指纹、模式、四个 role 的模型 tag、bank 哈希）；不一致则拒绝复用输出目录。
- **建记忆的 nonce 不含 bank**（`build_namespace`），因此换题库重跑可命中 API 缓存：记忆不变、只重算测试题。
- 每个事件写 `checkpoint.json`；中断后重放确定性缓存请求即可恢复（绝不手工编辑快照）。

---

## 11. Builder 奖励

`Environment.builder_score`（v2 为 `_builder_score_v2`），probe、在线 GRPO、VERL 共用同一入口。

### 11.1 公共中间量

| 量 | 定义 |
| --- | --- |
| `gain` | 错→对 的题数 / 分母（v1：总题数；v2：与 $x$ 有证据交集的题数，至少 1），截断到 $[0,1]$ |
| `loss` | 对→错 的题数 / 同一分母，截断到 $[0,1]$ |
| `faith_ratio` | 逐条 faithful 判定的通过比例（`all(faith_scores)` 为忠实） |
| `information_loss` | 对被 DELETE/UPDATE/MERGE 的旧条目：用 JUDGE 列其**必须保留的原子事实**（`preserve/corrected/unsupported`），再在编辑后记忆中分批寻找文本证据；缺失事实数 / 需检查事实总数 |
| `growth_ratio` | $\min(1,\max(0,(\text{after}-\text{before})/\max(1,\text{source},\text{editable})))$ |
| `shrink_ratio` | $\max(0,(\text{before}-\text{after})/\text{before})$ |
| `redundant_ops` | 重复 ADD、仅改来源的 UPDATE、无信息增益且不缩短的等价改写、整动作无文本效果等（见 11.4） |
| `coverage.ratio` | v2 专有：$x$ 中"Builder 可见范围内尚未存过"的原子事实，在编辑后被存下的比例 |

### 11.2 v2 目标（`configs/rl_*_v2.json` 默认）

$$
\begin{aligned}
R_B ={}& \mathbb{1}[\text{faithful}]\cdot w_g\,\text{gain}
      - w_\ell\,\text{loss}
      - w_f\,(1-\text{faith\_ratio})
      - w_r\,\text{information\_loss}\\
     &+ \mathbb{1}[\text{faithful}]\cdot w_c\,\text{coverage.ratio}
      + \mathbb{1}[\text{safe\_compression}]\,w_s\,\text{shrink\_ratio}
      - w_s\,\text{growth\_ratio}
      - w_d\,\min(|\text{redundant\_ops}|,\ \text{cap}).
\end{aligned}
$$

默认 $w_g=1.0,\ w_\ell=2.0,\ w_f=2.0,\ w_r=2.0,\ w_c=0.5,\ w_s=0.2,\ w_d=0.1,\ \text{cap}=10$。
`safe_compression = faithful ∧ loss=0 ∧ retention_pass ∧ shrink>0`。

关键性质：

1. **遗忘独立受罚**：即使平均正确率不变，对→错仍按 $w_\ell$ 扣分。
2. **合法 NOOP 恒为 0**：NOOP 时 `after=before`、$\text{delta}=0$，且**不重复调用**有随机性的 Reader/Judge（不靠环境噪声取奖励）。若 $x$ 有新事实，NOOP 会因 coverage 项为负而低于 0。
3. **覆盖率项不依赖题库**：`SOURCE_INVENTORY` 列出 $x$ 中值得存的原子事实（≤30 条，可空），先扣除编辑前 Builder 可见条目已覆盖的，再看编辑后（写入条目 + 仍存在的可见旧条目）覆盖比例；无新事实时该项不计分也不扣分。因此"没有测试题但有 $x$"的 stream/patch 状态也可评分（`coverage_scorable`）。
4. **翻转复核**：$k$ 下答案翻转时，用 $2k$ 再读编辑前后各一次，取两次变化的平均（`flip_delta`）；只是 top-$k$ 排名被挤动的翻转只算一半。
5. **冗余只对照 Builder 看得到的条目**（`M_old` 全文 + `catalog_ids` 摘要），避免罚它看不见的重复；最多计 `cap` 个操作，靠增加无害操作无法稀释罚分。
6. **非法输出严格低于合法下界**：合法项各自有界，故

$$
R_B^{\text{invalid}} = -(w_\ell+w_f+w_r+w_s+w_d\cdot\text{cap}+1)\quad(-8.2\ \text{于默认值}).
$$

v1 公式相同结构但：分母为总题数、无 coverage、冗余按 `max_ops` 计、`invalid = -(w_\ell+w_f+w_r+w_s+w_d\cdot\text{max\_ops}+1)`。

### 11.3 事实保留检查（`assess_edits`）

- 若旧条目文本在编辑后**原样存在**（含存活副本）→ 直接通过（`method="exact_text"`），不调用 judge。
- 否则：`_inventory` 产出原子事实清单（`preserve` 必留；`corrected` 需给出原文支持的替换事实；`unsupported` 不必保留但须标出原断言）→ `_coverage` 在**完整条目**批次中寻找证据，批次按"改动条目优先、prov 相关条目次之"排序，只有**一个批次联合蕴含**该事实才计覆盖；缺失判定会遍历全部候选记忆，不把检索未命中当作丢失。
- 不可修复/判官失败 → `Unknown`，不转成惩罚或通过。
- 明确声明：这是**诊断性语义校验**，不是无损性的形式化证明。

### 11.4 冗余判定

对每个写入操作（DELETE 不计入分母）：

1. `UPDATE` 且文本与父条目完全相同 → `unchanged_text`（仅改来源也算）；
2. 非"安全缩短"（父条目全部保留且 token 减少）且文本与基线（未被触碰的旧条目 + 本次先前写入）完全相同 → `duplicate_text`；
3. 非"安全缩短"时，把写入条目的事实清单在基线上做覆盖，若全部覆盖 → `equivalent_write_without_token_saving`（`semantic=true`）；
4. 整动作前后条目文本多重集相同（如 DELETE+ADD 改名、互换 ID）→ `no_final_text_effect`。

---

## 12. Attacker 奖励

### 12.1 v2（`_attacker_score_v2`）

单题：

$$
r_i=\begin{cases}
-(p)\,(1-\text{progress(stage)}) & \text{被 gate 拒绝 / duplicate}\\[2pt]
-(p+0.1) & \text{整个输出非法（低于任何合法输出）}\\[2pt]
0 & \text{该题 judge 预算溢出（环境原因，不惩罚策略）}\\[2pt]
v(M)\cdot w_{\text{nov}}\cdot \dfrac{|\text{necessary\_rids}|}{|E|}\cdot w_{\text{pers}}\cdot w_{\text{repair}} & \text{accepted}
\end{cases}
$$

$$
R_A=\frac{1}{\texttt{questions\_per\_pack}}\sum_i r_i,\qquad p=\texttt{attacker\_reject\_penalty}=0.1 .
$$

- **拒绝阶梯**（越晚失败扣得越少，且恒 $<0$，故低于"空列表"=0）：`schema 0.0 < oracle_type 0.2 < oracle_answerable 0.35 < unsupported_answer 0.5 < scope_or_completeness 0.65 < closed_book 0.8 = duplicate 0.8 < reader_unanswerable 0.85`。
- **价值**：$v(M)=1$ 若 $M$ 为空（此时跳过 defect 计算，省 reader 调用）；否则 $v=0.5+0.5\cdot\mathbb{1}[\text{defect 可修复}]$。
- **novelty**：$w_{\text{nov}}=1/(1+\max_{r\in E}\text{usage}(r))$。用 **max** 而非 mean：往 $E$ 里塞没用过的 round 无法把权重拉回（抑制"围着一个 session 刷题"）。
- **必要性**：一次 judge（`EVIDENCE_AUDIT_SYSTEM`）返回 `necessary_rids / personal / duplicate_of`。$|\text{necessary}|/|E|$ 惩罚无关证据填充。
- **个人相关**：非 SSA 题且 `personal=false` → $\times w_{\text{pers}}=0.25$（替代不可靠的一人称词表）。
- **gate 修补折减**：`repaired=true` → $\times w_{\text{repair}}=0.5$（不把 gate 的修复算作策略能力）。
- **语义去重**：audit 的 `duplicate_of` 指向 `prior_questions`（只取与当前 $E$ 有交集的最近 8 条）中同实体/属性/时间范围/操作的问题；字面去重按 `question_key`。
- 提示词末尾拼接 `ATTACKER_SCORING_NOTE`（少出题优于出错题、$E$ 不加无关 round、换说法重复无分），使**提示词与奖励一致**。

v1 差异：拒绝分是固定阶梯（schema −0.1 / 形状 0 / 内容空 0.15 / 答案错 0.25 / 太简单 0.35 / scope 0.45），重复题给 0，novelty 用**平均**已用次数，无必要证据比例、无 audit、空 $M$ 仍调用 reader。v1 的缺陷：4 道 closed-book 被拒可得 $+0.35$，高于一道合格题，属于奖励漏洞。

### 12.2 题库生成（bank）

`generate_bank`：对每个 pack 独立"提议 + gate"，纯函数式（不修改调用方的 `accepted`），失败可只重试该 pack。

- 每题记录 `status ∈ {accepted, accepted_no_defect, rejected, duplicate, invalid, empty, over_budget, type_infeasible, transport_error}`；`defects_only` 时只保留暴露可修复缺陷的新题。
- `--memory-run`：针对某次构建的 $M$ 出题（attacker 可见 `referenced_by_memory`/`memory_entries`）；旧题作为 `base_bank` 参与去重、novelty 与 `audit_context`。
- `--memory-pool`：用一个**记忆池**（最新 $M$ 占 `latest_share=0.5`，其余在更早轮次间均匀），每 pack 用哪份由 `hash(seed, case, pack)` 决定并记 `memory_pool_index`，防止 Attacker 只过拟合最新 Builder。
- 逐次尝试的输入/输出/gate 结果写入 `pack_traces/`，`bank_log.json` 用 `trace_file` 关联；瞬时 API 失败只重试该 pack（`PACK_RETRIES`），连续 `PACK_FAIL_LIMIT` 个 pack 失败才放弃 case。

---

## 13. 训练

### 13.1 训练数据

状态 → JSONL/Parquet：`data_source=admem_{role}`、`prompt`、`reward_model.ground_truth=""`、`extra_info={state_path,config,split}`。只有 `train` split 可训练；导入时校验 role/prompt/环境指纹一致。Builder 的 GRPO 只接受 `builder_state_scorable`（有测试题，或 v2 下 `coverage_scorable`）。

### 13.2 SFT 预热（可选但推荐）

只学**已验证示范**：stream 需合法 + 忠实 + 全部测试答对（无测试时至少 1 个 op）；patch 需 `patch_ok`；refine 需"更短 + 全部已解题仍答对"。损失只在实际 completion token 上：$\mathcal{L}=-\sum_t\log \pi(w_t\mid w_{<t})$。GRPO 只能锐化已存在的行为，SFT 先降低非法/不忠实率。

### 13.3 GRPO

`admem/learning.py`

- **Rollout**：`temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0`，使采样分布与 teacher-forcing logp 一致；截断（达到 `max_new_tokens` 且无 EOS）追加 `[TRUNCATED_OUTPUT]` → 非法输出 → 负奖励。
- **只对 completion 计 logp**（prompt/padding 不参与）；用 `logits_to_keep` 避免整段 prompt 的词表 logits。
- **组内优势**：

$$
A_i=\frac{r_i-\bar r}{\max(\text{std}(r),\,\sigma_0)+\epsilon},\qquad \sigma_0=\texttt{std\_floor}=0.1\ (\text{v2}).
$$

$\sigma_0>0$ 防止"奖励只差几个百分点（例如只差长度项）被放大成满幅优势"。$\text{std}\le 10^{-8}$ 的组直接跳过（无信号）。

- **损失**（逐 token PPO clip + k3 KL 锚定冻结参考策略）：

$$
\mathcal{L}=\frac{1}{T}\sum_t\Big[-\min\big(\rho_t A,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A\big)+\beta\big(e^{\Delta_t}-\Delta_t-1\big)\Big],
$$

$\rho_t=\exp(\log\pi_\theta-\log\pi_{\text{old}})$，$\Delta_t=\log\pi_{\text{ref}}-\log\pi_\theta$，$\epsilon=0.2,\ \beta=0.02$，梯度范数裁剪 1.0；非有限损失直接报错（不更新参数）。

- **组内非法样本整形**（`shape_group`）：一个非法样本（−8.2）会把组内 std 撑大、压扁合法样本间约 0.05 量级的差别。故把非法样本奖励改为

$$
r_{\text{invalid}}\leftarrow \min_{i\in\text{legal}} r_i-\max(\text{std}_{\text{legal}},\,0.05),
$$

保持它排在所有合法样本之下，但不再主导归一化尺度。合法性阈值：v2 Builder 取"合法下界 + 0.5"，v2 Attacker 取"$-(p+0.1)+0.05$"；全部合法或全部非法时原样返回。

- **更新调度**（`admem/train.py`）：每个 update 采 $G=8$ 个 prompt × $g=8$ 个样本，只选组内 std 最大的 `topvar` 个组做一次 `step`；`groups_per_update` 个候选组。单个 group 评分失败（`Unknown`）只跳过该 group，连续 `max_env_errors=3` 组失败才中止（不烧完 API 预算）；确定性 `BudgetExceeded` 的状态标记为不可评分并不再抽取。
- **环境指纹**：`Config` 值 + 奖励相关源码哈希（`_FINGERPRINT_EXCLUDED` 排除 train/learning/validation/verl_* 工具模块）；状态与训练配置指纹不一致时拒绝训练。
- **API 缓存**：键包含服务、请求、模型名与 nonce；同一组内共享长 prompt 前缀。

### 13.4 VERL 集成

- `admem/verl_reward.py:compute_score(data_source, solution_str, ground_truth, extra_info)`：从 `state_path` 载入状态，校验 split/role/配置指纹/`full_hash`，调用同一个 `Environment.score`，返回 `effective_reward` 与诊断字段（`legal/noop/faith_ratio/gain/loss/information_loss/growth_ratio/component_*`；attacker 为 `accepted/rejected/duplicates/over_budget`）。**不使用** `ground_truth` 中的官方答案。
- `admem/verl_adv.py`：注册优势估计器 `admem_grpo`，在组内做 shape + `max(std, floor)` 归一化，再乘 `response_mask`（只在有效 token 上非零）。VERL 按样本独立打分，无法做组内整形，故同时用 SFT 降低非法率。
- 环境失败默认 fail-fast；`ADMEM_REWARD_ON_UNKNOWN=neutral` 返回 0（等于 NOOP/空列表奖励，是有偏折中），写日志，累计超过 `ADMEM_MAX_UNKNOWN` 仍报错。

### 13.5 交替训练

`scripts/alternate.py`：每轮冻结对手，按"构建记忆 → 针对记忆池出题 → 训练 Attacker → 用新 Attacker 生成对抗题库（`defects_only`，并入累计题库）→ 重建记忆并收集 Builder 状态 → 训练 Builder"推进。因为 Attacker 攻击的是"数百个 Builder 步骤构建出的整份记忆"、奖励又依赖慢速 judge，所以必须整阶段冻结对手，而不是像 Code-A1 那样每步同时更新。

四个风险的对应机制：

| 风险 | 机制 |
| --- | --- |
| Builder 遗忘 | 累计题库即 Mistake Book，历史题永久 replay |
| Attacker 过拟合单个 Builder | 记忆池（多轮旧记忆 + 最新记忆） |
| 不可修的"攻击" | `require_reader_answerable`：题必须能被 Reader 用原文答对 |
| 过拟合冻结对手 | 阶段短（`phase_steps`、`patience=2`），每阶段从上一最优继续 |

---

## 14. 验证与早停（只用 val split）

`admem/validation.py`

- **不用的信号**：测试集、以及 case 的官方目标题（单题方差 ±7 分且 Builder 可通过它同时丢掉其他一切；部署时也不存在目标题）。训练奖励也不用于选点（它会通过"少写"上升）。
- **Builder 指标**：`bank_accuracy`（冻结 `val_bank`，**主指标**，逐题结果用于配对 bootstrap）、`official_accuracy`（报告但不决策）、护栏 `legal_rate / faith_rate / noop_rate / fallback_rate / mean_m_tokens`、可选 `kl_loss`。
- **Attacker 指标**：`defects_per_pack`（**主指标**：每个 pack 中"通过 gate 且暴露参考记忆可修复缺陷"的题数），护栏 `invalid_rate / accept_rate`。
- **EarlyStopper**：配对 bootstrap $P(\text{new}>\text{best})\ge 0.9$ 且平均增益 $>$ `min_delta` 且无护栏违反才成为新最优；连续 `patience=3` 次无显著提升停止；护栏相对 step-0 基线（Builder：legal −0.05、faith −0.05、NOOP +0.15、fallback +0.10、记忆规模 ×1.5；Attacker：invalid +0.10、accept −0.10）违反则不得成为最优，连续 `guard_patience=1` 违反即停。**NOOP 护栏是崩塌探测器**：策略学会"不写"时奖励升到 0 而 `noop_rate` 跳升。

---

## 15. 评测协议

- `longmemeval-eval`：先用选定 Builder 在 **test split** 上以 `mode=build` 构建 $M$（`M_build`/`M_final`），**之后**才从 `private_eval.jsonl` / `--labels` 读取官方 $q,a$；`--cases` 只用于构建无损历史。
- **答题**：Reader 只见 $M$ 的 `text`（检索 top-`reader_k=10`，或 `--all-memory` 全量；全量装不进预算直接报错，不把截断后的 $M$ 报成 all-M）；`[Source date: …]` 由 host 渲染；题型提示仅在 `hint_mode=target_type` 时可见。Reader **不能通过 `prov` 回取原文**。
- **判分**：项目内 LLM judge 仅作诊断；`predictions.jsonl` 交 LongMemEval 官方 `evaluate_qa.py` 得到正式分数。
- **未知值语义**：`status=error/unknown` 的样本不计入 `accuracy_completed` 分母，同时报告 `lower_all`（全当错）与 `upper_all`（全当对）；abstention 单独分组。
- summary 记录 `self_judging`（策略与判官同模型时告警）与 `benchmark_informed`（提示词在看过全部 500 题后写成，任何 split 都不是 held-out）。

---

## 16. 不变量与已知边界

**必须保持的不变量**

1. **标签隔离**：官方 $q,a$、`answer_session_ids`、消息级 `has_answer` 永不进入 $F$、sampler、Builder、Attacker 的输入；只在评测侧读取。Attacker 的 `audit_context` 只接受自生成题；`type_policy=official` 属显式诊断泄漏并留痕。
2. **不静默截断**：超预算抛 `BudgetExceeded` 或整块丢弃并记录；`prompt`/`entry`/`reader` 预算都按精确 tokenizer 计数；`raw_chunks` 保留原文每个字符，只在块前加日期头。
3. **未知 ≠ 错误**：API/服务失败、判官截断（`[TRUNCATED_OUTPUT]`）、abstention、缺标签一律记 `null/unknown`，不当作答错或成功；`Unknown` 不产生负奖励。
4. **每个产物可追溯**：full 指纹、pack 指纹、`run_config.json` 签名、环境指纹、prompt 哈希、`pack_traces/`、`checkpoint.json`、`ledger.json`。
5. **奖励只在模型原始输出上计算**：fallback 与 gate 修复只影响状态/后续 $M$，不改变该样本的奖励。

**已知边界**

- 事实保留/覆盖率/忠实度都是 **LLM 判官估计**，不是形式化无损证明；跨批次才能联合推出的事实会被保守判为缺失。
- Gate 的 top-30 宽检索**不是全历史穷尽检查**，无法证明"没有遗漏"；accepted 只表示"通过 gate"，不是 Mistake Book 的充分条件。
- 提示词由看过全部 500 道 LongMemEval 题后写成，生成的题会模仿测试问法；严格 held-out 结论需要另一个未参与设计的基准。
- 全部默认权重（`gain/loss/faith/retention/coverage/size/redundancy`）与阈值（`window/entry/timeline/screen_k`）**尚未在真实数据上调优**，应以扫描实验为准，不得把它们当作已调优结论。
- 假阴性/假阳性来源：判官误判、batch 边界、检索排名抖动（已用 $2k$ 复核折半缓解）、采样噪声（每题只测一条答案）。

---

## 17. 代码映射

| 文件 | 职责 |
| --- | --- |
| `memory.py` | 无损 $F$：round 化、双序、结构边、`Document`、`memory_documents` |
| `retrieve.py` | BM25 + dense、块 max-sim、RRF、邻接扩展、cross-encoder、补检索 |
| `packs.py` | legacy pack（字符预算）、`Sampler` 无放回 sweep、`fit_pack` |
| `admem/packing.py` | token 预算 pack、种子切分、`facts/hybrid` 邻居、`diversify` |
| `admem/data.py` | 数据准备、标签隔离、split 分配、`TypeScheduler`、`hint`、`policy_type` |
| `admem/prompts.py` | Attacker 视图/prompt、Builder v1/v2、Reader/Grade/Faith/Screen、目录与预算回退 |
| `admem/audit_prompts.py` | 六题型指引、gate oracle/support、`build_payload`、结构可行性、`EVIDENCE_AUDIT` |
| `admem/store.py` | `apply`（动作校验与执行）、`augment`、`raw_chunks`、`question_key` |
| `admem/edit_checks.py` | 事实清单、覆盖检查、冗余判定 |
| `admem/environment.py` | 冻结环境：index/reader/grade/gate/defect、`builder_score`、`attacker_score` |
| `admem/pipeline.py` | windows、stream/audit/patch/refine 主循环、状态收集、bank 生成、记忆池 |
| `admem/rewards.py` | v2 奖励的纯算术：拒绝阶梯、novelty、`flip_delta`、`shape_group`、组内优势 |
| `admem/learning.py` | GRPO 参考实现：rollout、logp、advantage、clip+k3 KL、优化 |
| `admem/train.py` | SFT/GRPO 单机入口、组选择、错误容忍、checkpoint |
| `admem/verl_reward.py` / `verl_adv.py` / `verl_launch.py` | VERL 奖励接口、`admem_grpo` 优势估计器、启动命令 |
| `admem/validation.py` | Builder/Attacker 验证指标、配对 bootstrap、`EarlyStopper` |
| `admem/probes.py` | 合成 probe 套件、Attacker 奖励方差诊断（`signal_group_rate` / `all_fail` / `same_reward_collapse` / `deterministic`） |
| `admem/bootstrap.py` | 从已构建 $M$ 制造 `missing/fragmented/duplicate` 状态，避免早期奖励全零 |
| `admem/cli.py` | prepare/bank/run/bootstrap/probe/export/relocate/evaluate/longmemeval-eval/merge |
