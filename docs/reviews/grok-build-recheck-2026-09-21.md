# Grok Build 修复复验

日期：2026-09-21。对象：当前 `feat/cloud-research-rag` 工作区中相对 `d546804` 的未提交修复；对照首轮 `grok-build-review-2026-09-21.md` 的 R1–R16。没有修改生产源码、提交或发布。

**结论：修复有实质进展，但仍不能整体验收。** 离线测试已全绿，原有数个关键失败路径得到修复；独立探针仍复现分块越界、来源丢失、取消后发布、损坏配置被覆盖，以及 HTTP 配置缓存失效。评估已增加 BM25 执行路径，但原计划的三组评估仍未实现，部分指标口径也不正确。

## 本次实际运行

| 检查 | 结果 |
| --- | --- |
| `npm run typecheck` | 通过 |
| 隔离存储、`SKIP_EMBEDDING_TESTS=1 npm test` | **177 passed / 4 skipped，10 个测试文件通过、1 个跳过** |
| 两个 `smoke:voyage-*`，显式移除 Key | 均正常启动、exit 0，输出 `UNVERIFIED` |
| `npm pack --dry-run --ignore-scripts --json` | 35 个文件，包含新增 `abort.ts` |
| `npm run eval:retrieval`，无 Key | 能运行并生成 BM25 报告；打印 hit@5、recall、MRR 均为 0.22727；指标限制见 F5/F6 |
| 独立临时 SQLite + 确定性向量探针 | 复现下列 F1–F4、F7 |
| `git diff --check` | 通过 |

全部索引探针只操作临时合成文档，模型函数以确定性向量替换，真实网络被阻断。未运行真实 ONNX 推理、Voyage 请求、Pi 完整会话、干净安装或论文效果验收。此次未重新执行打包后的 Pi loader。

## 剩余问题

### F1 · P1：overlap 拼接后仍突破 maxTokens（R4 未关闭）

位置：`chunking.ts:210–221`、`187–195`。

`appendPiece` 在拼接前发现超限后调用 `flush(true)`，但 flush 可能保留 overlap；随后直接拼接新 piece，没有重新检查剩余容量。默认参数下，输入 `a.repeat(400) + "\n\n" + b.repeat(960)`，输出估算 token 数为 **[100, 256]**，配置上限为 **240**。这在实现自身的估算规则下已违反上限，无需真实 tokenizer 就能确认。

修复：按 overlap、分隔符和新内容的总预算分配空间，在每个输出边界验证上限。新增“短于 target 的首段 + 接近 max 的后段”用例。真实模型 tokenizer 上限仍须单独满足；CJK-aware 字符估算不能自动成为模型硬上限。报告 E2 的“hard cap / done”应撤回。

### F2 · P2：正文去重误删不同来源的有效内容（R11/R12 的新回归）

位置：`chunking.ts:172–175`。

为去除纯 overlap 尾块，`emit` 直接丢弃与前块相等或为其后缀的正文，未检查是否有新增内容或来源是否相同。两个 SourceBlock 分别来自 PDF 第 1、2 页，正文均为 `Important repeated evidence`，最终只保留第 1 页的一个 chunk。第二页没有进入索引，无法按该页引用；短内容恰好是前块后缀时也会被误删。

修复：用新增内容状态判断纯 overlap，不能按全局正文相等删除有独立来源的块；增加相同正文不同页、不同章节及短后缀正文测试。Context 的去重键也应检查 page/chunk identity，避免再次合并不同页来源。

### F3 · P2：最后一批 embedding 期间取消，仍写入并发布（R8 未关闭）

位置：`indexing.ts:337–362`，`rebuildWithSwitch` 发布之前也缺检查。

信号在进入 batch 前检查，但最后一次 `await provider.embedDocuments` 返回后直到事务及 manifest 发布均没有再检查。LocalEmbeddingProvider 在最后一次推理后也没有取消检查，因此这是实际可达的时序。探针让最后一次 embedding 在返回向量前触发 abort，结果仍为 `indexed=1, failed=0`，生成活动 manifest，原正文被替换。

修复：异步操作返回后、写入前和发布前重新检查 signal，保证取消不表现为成功。当前测试只覆盖“调用之前已取消”，不能验证运行中取消。自动注入的 deadline 使用合作式取消；未证明解析/ONNX 期间能及时结束整个 hook。

### F4 · P2：损坏配置诊断未约束业务入口，普通工具会覆盖原配置（R15 未关闭）

位置：`config.ts:191–209,212–225`；`index.ts` 的 `rag_index` 配置保存路径。

`loadConfigDetailed` 已返回 issues，但 `loadConfig` 丢弃它们；`saveConfig` 同样忽略文件 invalid 状态。已有兼容本地索引时，把 config.json 写为 `{BROKEN`，`rag_query` 仍正常返回结果且不报告配置问题；再调用 `rag_index`，工具将默认配置写回，文件状态变为 `ok`。原文件被覆盖，恢复诊断依据消失。自动 hook 虽取 detailed 结果，也未处理其 issues。

修复：只读 status 可以使用诊断默认值；业务入口须显式处理配置错误，普通索引/toggle 不应承担隐式恢复。明确修复或重置入口，并保留原损坏文件。仅在 status 中显示 issues 不足以关闭此问题。

### F5 · P2：三组评估仍缺实现（R14 部分修复）

位置：`scripts/eval-retrieval.ts:171–187`。

脚本现在能索引固定代码语料并执行 BM25，这比首轮的纯模板有进展。但本地 MiniLM 混合检索没有运行路径；云 embedding 分支使用 `hasKey ? "not-run" : "not-run"`，云 reranker 也恒为 not-run。所谓 opt-in 没有对应 CLI 参数或执行分支。语料及标注路径仍固定，没有计划要求的外部固定语料输入。

修复：实现可显式选择并执行的 local-hybrid / cloud-embedding / cloud-embedding+reranker 三组，允许缺 Key 时跳过云组；BM25 可作为额外基线。文档应写“云评估执行路径未实现”，不能把缺口仅归因于没有 Key。

### F6 · P2：候选召回率和来源准确率计算口径不成立

位置：`scripts/eval-retrieval.ts:119,135–159`。

`retrieve` 最终只返回 5 条，NoneReranker 路径不会使用传入的 candidateTopK=30 扩召回；`recallAtCandidates` 与 hitAt5 因而都只检查这 5 条，不能表示候选阶段召回。此外，`hitMatches` 已要求页码一致，只有 matched 才进入 `pageChecked`，再检验相同页码会让单页标注的 sourceAccuracy 有值时必然为 1；错误页码从分母中消失。

修复：分别保留候选集合和最终 top5；明确定义按问题命中或按相关证据召回的分母；先按内容/文件判断相关性，再独立评估来源位置是否准确，并纳入错误来源。此次打印的 0.22727 不可当作三组模型效果或候选召回验收数据。

### F7 · P2：缓存 provider 导致 HTTP 配置更新不生效（R16 修复引入）

位置：`providers/embedding/factory.ts:13–15,55–70`。

缓存键只含 provider/model/dimensions/key，但实例固定持有 timeoutMs/maxRetries。先用 30000ms/3 构造 Voyage provider，再以相同模型和 Key 请求 1000ms/0，返回同一对象，其参数仍是 30000ms/3。无需真实请求即可确认；状态展示的新配置与后续请求行为不一致。

修复：将相关 HTTP 参数加入云 provider 缓存键，或将传输策略按请求传入；本地模型 pipeline 缓存可继续复用。增加修改 timeout/retries 后生效的测试。

## 首轮问题的复验状态

| 首轮编号 | 本次判断 |
| --- | --- |
| R1 | 已修复已扫描目标的解析错误发布路径；回归用例通过。扫描失败与合法空集合的区分仍未被此次验收覆盖。 |
| R2 | force rebuild 独立 generation 和失败不删旧记录的路径已修复；新增回归直接测 helper，完整命令/重启失败恢复仍需补证据。 |
| R3 | 云刷新关闭逻辑已接入 hook；用例通过。 |
| R4 | 未关闭，见 F1；tokenizer 合同仍不足。 |
| R5 | 非空索引查询已有统一兼容检查，旧库无 fingerprint 拒绝查询用例通过。 |
| R6 | 真实向量表维度检查已前移至空库判断前，代码路径正确；新增名为 empty 的用例复用非空库，不能单独证明空库重建成功。 |
| R7 | unique generation 替代删除同合同旧目录；路径测试通过。 |
| R8 | 部分完成，signal 已传递；运行中取消仍失败，见 F3。 |
| R9 | 手动命令/tool 已补 failed、errors、rerank、degraded；hook 仍忽略刷新返回的 failed/errors，context 未呈现检索降级原因。未全部关闭。 |
| R10 | 临时 embedding 故障 BM25 降级用例通过；自动查询异常有跳过处理。 |
| R11 | 页码、section、id、chunkIndex 已贯通主要输出；仍有 F2。转换后的 HTML/DOCX 文本行号也不等于原始源位置，相关分支仍需校正。 |
| R12 | Markdown 解析保留短章节的用例通过；chunker 的后缀去重仍可删有效短内容，见 F2。 |
| R13 | 已关闭启动缺陷；无 Key 路径实跑通过，真实 API 仍未验证。 |
| R14 | 部分完成，见 F5/F6。 |
| R15 | 部分完成，诊断信息已有，但入口仍有 F4。 |
| R16 | 顺序查询复用 provider 已实现；HTTP 策略缓存引入 F7。未测并发首次初始化及真实 ONNX 耗时。 |

建议先修 F1–F4，再补 F5–F7；E2/E3/D3/B2 等 done 状态应按剩余缺陷调整，避免把“代码存在 + 部分离线测试通过”等同于已满足原计划。

## 复现材料

- `grok-build-recheck-probe.mjs`：离线探针，不发送用户数据，不依赖真实模型。
- `grok-build-recheck-evidence.jsonl`：本轮探针结果。
- 测试、pack、eval 原始日志：`/private/tmp/rag-recheck/`。

```bash
node --experimental-strip-types docs/reviews/grok-build-recheck-probe.mjs
```

探针保留其新建临时目录以便查看，未修改现有用户索引。运行评估脚本按其既有行为刷新了忽略目录 `eval/runs/` 中的当日报告。
