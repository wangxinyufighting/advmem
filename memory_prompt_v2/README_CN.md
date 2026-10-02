# Memory audit 英文 prompt v2

## 交付范围

本版依据你上传的 `longmemeval_questions.txt`（500条带答案的导出记录）、算法v1及case0123的实际日志修订。不是把500条原题放进attacker，也不读取某个case的target/gold来定向出题。

- `memory_prompts.py`：共享英文规则 + 六类英文规则、英文gate模板、原文白名单输入适配。
- `prompts/`：可直接阅读的完整英文prompt；每个attacker类型文件已经包含共享规则。
- `install_prompt_v2.py`：保守的AST源码安装器；默认只预览，接口不同就停止，不猜着覆盖。
- `corpus_profile.json`：由上传文本计算的分布、不可答题文本识别及待复核异常。
- `TRAINING_REVIEW.md`：训练前的设计/工程问题与待办。
- `tests/test_prompt_v2.py` / `VALIDATION.json`：离线验证和范围说明。

## 1. 数据中最需要对齐的问法

| question_type | 原字段计数 | 根据答案文字识别的不可答条数 | 新提示词重点 |
|---|---:|---:|---|
| single-session-user | 70 | 6 | 人名、地点、金额、单位、时间、过去状态和一次性事件，不只稳定偏好 |
| single-session-assistant | 56 | 0 | 回忆先前推荐、长列表第k项、表格单元格、链接/标识、创作内容；不是问外部世界真相 |
| single-session-preference | 30 | 0 | 应用历史约束给建议；a是个性化rubric，而不是“用户喜欢什么”的事实问答 |
| multi-session | 133 | 12 | 计数、求和、差值、比例、平均、集合和跨事件解析；每个前提都必须在E中 |
| knowledge-update | 78 | 6 | 新值、旧值、初值、前后对比、指定时点；不把它缩窄成latest-only |
| temporal-reasoning | 133 | 6 | 排序、间隔、相对时间、时间定位事件；提及日期不等于事件日期 |
| 合计 | 500 | 30 | 30条不可答记录嵌在上述字段中，不额外加到500上 |

上传文件没有question_id、_abs标记、session日期或原文。不可答条数是按答案明确写“未提到/信息不足”推断，不是重新验证官方ID。

Index123的目标是高中到本科结束的累计教育年数，参考10年；Index131是到硕士结束的不可答变体。它们进一步说明case0123的高中事实与评测相关，但题库文本不能恢复gold round IDs。

需要复核的文本：Index318同时写14天和包含末日8天，时间计数说明自相矛盾；Index127把“未提到12月参观”写成0。原文不足以核实或改正它们，运行prompt不模仿这种“没看到=零”的规则，也不自动改官方标签。

## 2. 安装：先只改生成器，保留gate做对照

解压本包到任意目录，假设解压后的文件放在 `prompt_patch/`，项目是当前目录：

```bash
# 默认只预览，不写源码。
python prompt_patch/install_prompt_v2.py --project .

# 确认兼容后安装英文生成器。
python prompt_patch/install_prompt_v2.py --project . --apply
```

安装器会：

1. 替换已知接口的 `agents.Attacker.generate(self, pack, qtype, date, n_questions=4, nonce="")`。
2. 复制 `memory_prompts.py` 到项目根目录。
3. 将 `llm.Client.json()` 中已知的“仅返回一个有效 JSON 对象”附加指令换成英文。
4. 为被改写文件保存按旧内容哈希命名的 `.bak`，不更改模型、代理、重试或API密钥。

没有收到你本地最新的完整 `agents.py/llm.py`。安装器只适配此前已知的接口；签名或gate结构不同会明确停止，所有计划在写盘前先解析/编译。它不是任意版本代码的万能迁移器。

验证实际入口：

```bash
python - <<'PY'
import inspect
import agents
from memory_prompts import attacker_prompt, PROMPT_VERSION, prompt_hashes
print(agents.__file__)
print(inspect.getsource(agents.Attacker.generate))
print(PROMPT_VERSION)
print(attacker_prompt('multi-session'))
print(prompt_hashes())
PY
```

本版生成器的system指令和已知JSON追加指令是英文，生成q/a也要求英文。原文不翻译，专有名词原样保留。若你自己的客户端还追加了其他中文指令，需要检查客户端；安装器不猜改未知代码。

### 重新运行

保持原pack、模型和预算，改用新的输出目录：

```bash
python benchmark_attack.py \
  --data "$DATA" \
  --packs-dir runs/pack_types_5 \
  --embedding local \
  --questions-per-pack 4 \
  --gate full \
  --pack-chars 48000 \
  --gate-chars 120000 \
  --n-grid 5 10 20 40 \
  --seed 0 \
  --out runs/attacker_types_5_prompt_en_v2 \
  --resume
```

不要拿旧prompt的检查点直接续跑新prompt。`--resume`只用于新目录这一次配置内的中断恢复。模型API缓存仍可共用，因为prompt/payload变化会进入请求键；生成nonce也带PROMPT_VERSION。实验manifest还应记录 `prompt_hashes()`。

这不是纯“翻译语言”的单变量实验：v2同时改了题型规则和模型输入视图（结构化round、is_seed、中性session ID）。与旧版比较时应记为“生成器v2”，不能把效果全部归因于英文。

## 3. 可选：同时修订gate的三个提示词

在独立实验中执行：

```bash
python prompt_patch/install_prompt_v2.py --project . --with-gate
python prompt_patch/install_prompt_v2.py --project . --with-gate --apply
```

只在已知gate代码形状下更新：

- `screen = oracle.json(...)`：宽检索筛查prompt。
- `decision = oracle.json(...)`：独立oracle回答及题型检查prompt。
- `support = grade(...)`：改为 `verify_support(...)`，同时读取真实E原文，按q真正要求比较两种答案。
- gate中的 `full.render(...)` 模型输入改成中性ID原文视图，避免 `answer_*` 原session ID泄露给gate模型。

不会修改全局 `agents.grade()`，因此target benchmark的评分口径不因这次修改自动变动。gate调用的通用closed-book reader / grade仍沿用原实现；本包不声称整个工程每个模型调用都已改成英文。

第10包的“问两个店名，却因为oracle没复述优惠细节被拒绝”是新support规则的回归目标。新规则仍拒绝a中缺乏原文支持的额外陈述，不能只因q问得少就放过伪造事实。

重新跑时使用另一个out，例如 `runs/attacker_types_5_prompt_gate_en_v2`。不能把题型合规、答案支持、closed-book、完备性筛查融合成“只要两模型答案相同就接受”。

## 4. 这次实际改变了哪些输入

新的动态payload白名单为：

```text
type, question_date, n_questions, max_evidence_rounds,
rounds: [rid, session, session_date, is_seed, referenced_by_memory, messages(role,content)]
```

- 不传官方q/a/has_answer/answer_session_ids，也不传原始 `answer_*` session ID。
- 不修改F，只建立模型视图。`s39:r2`等rid不变，评测映射仍可用。
- 输入原文的完整role/content逐字保留。序列化会增加字符开销，48k原文字符不等于16k训练token，必须重新核实模型实际输入预算。
- `referenced_by_memory`只表示provenance引用，不代表事实覆盖。

可选的跨pack去重提示接口已提供，但旧benchmark**不会自动填它**：

```python
attacker.audit_context = {
    "full_hash": full.fingerprint,
    "accepted_items": previously_accepted_generated_items,
}
```

只允许本case已通过gate的自生成题，不能把target QA改名塞入此字段。适配器仅取与当前pack证据有重叠的最近24条，并报告省略数量；这不是全量事实账本，也不保证语义去重。正式维护、恢复、更新这个状态仍需调用端实现。

## 5. 没有偷偷改变的机制

当前generate调用与gate通常要求所有item.type等于本次qtype。因此本版仍然“一包指定一种题型”；不能只在prompt里说混合题型而不改验证器。

这意味着：第27包如果仍指定preference，新prompt会要求生成真正的偏好应用题，**不保证它一定询问高中年份**。要解决这个机会分配问题，需另测混合槽位/轮换调度：例如总计4题中保留一题给未覆盖事实、一题给可成立的跨会话组合，其余做类型轮换；同步修改逐题expected_type，不读取当前case的官方题型。该调度不是本补丁已完成的功能。

也没有实现：实体知识图谱、算术执行器、全文完备性证明、可靠不可答题生成、全局问题去重、builder/defect/GRPO或token级上下文管理。

## 6. 正式训练的评测边界

本版已经参考全部500条问题及答案进行设计。即使运行时不注入目标题，也不能再把这500条称为prompt设计完全未见的测试集；事后重新随机划分同一500条也不能消除这次人工暴露。

它们仍可用于开发对比，但应如实记录benchmark-informed prompt tuning。严格泛化评估需额外未参与设计的数据/任务，或者单独保留在查看这些目标前已经冻结的基线。不要把本文件的答案直接当训练reward或case-specific示范。

后续训练拆分还需检查正/负变体和共享原文；同一历史的变体不应跨训练/验证集。单凭这份导出没有question_id和历史，不能完成该去重。

## 7. 验证结果

- 29项离线测试通过：英文模板、六种类型、输入白名单、中性ID、schema、别名、预算、局部context隔离、gate支持接口、安装器预览/备份/重入。
- 对你实际上传的0123.zip，重放45个pack×6个题型，共270次payload构建，验证原文未丢、顺序和rid一致、种子标记正确、原session ID不暴露。
- 第27包仍包含14个round，6个seed round，s39:r2明确标成seed。
- **没有调用真实LLM重新出题，没有测到覆盖率提升。**安装器测试使用旧代码的已知结构夹具，不是你本地最新源码。
- 所有Python文件通过3.10语法解析。

```bash
python -m pytest -q tests/test_prompt_v2.py
```

在本补丁目录运行上述测试，避免与原项目tests目录混淆。
