# Grok Build 修复验收说明（供 Astra 复验）

日期：2026-09-21。  
对象：`feat/cloud-research-rag` 工作区相对 `d546804` 的**未提交**改动。  
对照：`grok-build-review-2026-09-21.md`（R1–R16）、`grok-build-recheck-2026-09-21.md`（F1–F7）。

本文件不是实现本身的“已通过”证明。它列出本次声称修了什么、用哪条命令复现、期望看到什么、以及仍然不能当验收关闭的项。请用独立临时目录和阻断网络的探针复核；不要在用户全局 `~/.pi/rag` 上跑。

**实现方自测（不能替代 Astra 探针）：**

| 检查 | 实现方结果 |
| --- | --- |
| `npm run typecheck` | 通过 |
| 隔离 `PI_RAG_DIR`、`SKIP_EMBEDDING_TESTS=1 npm test` | **188 passed / 4 skipped** |
| `npm run smoke:voyage-embed` / `smoke:voyage-rerank`，无 Key | exit 0，`UNVERIFIED` |
| `SKIP_EMBEDDING_TESTS=1 npm run eval:retrieval` | 四组都有状态：`local-bm25:ok`，其余 `not-run`（ONNX skip / 无 Key） |
| F1/F2 手跑 `chunkBlocks` | `a×400+\n\n+b×960` → 估算 token `[100, 240]`；同文两页 → 2 chunks |

未做：真实 ONNX、真实 Voyage、Pi 完整会话、干净安装、三页 PDF、论文效果。

---

## 1. 工作区范围

- 分支：`feat/cloud-research-rag`
- 基线 HEAD：`d546804`
- 状态：工作区脏，**未 commit、未 push、未 publish**
- 相对 `d546804`：31 个已跟踪文件改动，+1596 / −343；未跟踪 `abort.ts`、`__tests__/review-fixes.test.ts`、`docs/reviews/`

建议复验前：

```bash
cd /Users/chouchao/Documents/RAG_Knowledge_Repository/pi-local-rag
git rev-parse HEAD   # 期望 d546804
git status           # 应看到未提交修复，而不是干净树
```

隔离环境：

```bash
export PI_RAG_DIR=$(mktemp -d /tmp/pi-rag-astra-XXXX)
export PI_RAG_LEGACY_DIR="$PI_RAG_DIR/legacy"
unset VOYAGE_API_KEY
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck
npm run smoke:voyage-embed
npm run smoke:voyage-rerank
SKIP_EMBEDDING_TESTS=1 npm run eval:retrieval
```

复用复验探针（实现方已按 F1–F7 改过生产代码，期望这些 case 不再复现原失败）：

```bash
node --experimental-strip-types docs/reviews/grok-build-recheck-probe.mjs
```

首轮探针仍可作回归，但其中若干断言针对 `d546804` 的旧行为：

```bash
node --experimental-transform-types docs/reviews/grok-build-review-probe.mjs
```

---

## 2. 针对 F1–F7 的修复与验收点

### F1 · overlap 拼接后突破 maxTokens（原 R4）

**改动：** `chunking.ts` `appendPiece` / `flush`。`flush` 保留 overlap 后，再拼新 piece 前重算预算；超限则从 overlap 头部缩短或清空。`emit` 若仍超 `maxTokens` 会走 `hardSplitByTokens`。chunker fingerprint 升为 `token-v3`。

**期望：** 默认 180/240/30 下，输入 `"a".repeat(400)+"\n\n"+"b".repeat(960)`，各 chunk 的 `estimateTokens` **均 ≤ 240**。实现方见到 `[100, 240]`。

**用例：** `__tests__/parsing.test.ts`  
`keeps a short first paragraph plus a near-max second paragraph under maxTokens`

**仍不能关闭：** MiniLM 真实 tokenizer 上限。报告 E2 标 **partial**（估算硬上限，不是模型硬上限）。

### F2 · 正文去重误删不同来源（原 R11/R12 回归）

**改动：** `chunking.ts` 不再按“全文相等 / 前块后缀”丢 chunk；overlap 尾块只看 `addedSinceFlush`。页边界 `resetBuf()`。`context.ts` 去重键含 `id` / `pageStart` / `pageEnd` / `chunkIndex` / `hash`。

**期望：** 两页均为 `Important repeated evidence` 时得到 2 个 chunk，`pageStart` 为 1 和 2。Context 同时出现 `page 1` 与 `page 2`。

**用例：**  
- `parsing.test.ts`：`keeps identical text from distinct PDF pages`  
- `parsing.test.ts`：`keeps a short suffix that is new content in a different section`  
- `retrieval.test.ts`：`does not merge distinct pages that share content`

**仍开放：** HTML/DOCX 行号不是原始源位置（已改为不输出行号）。三页真实 PDF 抽查未做。

### F3 · 最后一批 embedding 期间取消仍发布（原 R8）

**改动：** `indexing.ts` 在 `embedDocuments` 返回后、Phase 3 写库前、stamp/`last_build` 前、`finalizeStaging` 前调用 `throwIfAborted`。`LocalEmbeddingProvider` 在推理返回后也检查。取消抛 `AbortError`，不记为 `indexed>0, failed=0`。

**期望：** 最后一次 `embedDocuments` 在返回向量前 `abort()`，`rebuildWithSwitch(..., signal)` reject `AbortError`；活动库正文仍是取消前的内容；不新写 `active.json` 指向这次 staging。

**用例：** `__tests__/review-fixes.test.ts`  
`F3: abort during the last embed batch does not write or publish`

复验探针 `abort_during_last_embed` 应对齐上述行为。

**仍开放：** 解析/ONNX 阻塞期间的抢占式取消；hook 12s deadline 仍是合作式。

### F4 · 损坏配置被业务入口覆盖（原 R15）

**改动：** `config.ts`：`saveConfig` 在 `fileStatus==="invalid"` 时抛 `ConfigFileInvalidError`；`requireWritableConfig()`；`resetBrokenConfig()` 把原文件拷到 `config.json.broken-<ts>` 再写默认值。  
`index.ts`：`/rag index|rebuild|refresh|on|off|ext|exclude` 与 `rag_index` 走可写检查。`rag_query` 在 invalid 时返回错误说明、不静默搜。只读 `status` / `/rag config` 仍显示诊断。自动注入在 invalid 时跳过。新命令：`/rag config reset`。

**期望：** `config.json` 为 `{BROKEN` 时：

1. `loadConfigDetailed().fileStatus === "invalid"`
2. `saveConfig(defaultConfig())` 抛错，文件内容仍是 `{BROKEN`
3. `rag_index` 不把 defaults 写回
4. `rag_query` 文本含 invalid / repair / `config reset`
5. `/rag config reset` 之后存在 `.broken-*` 备份，新文件为合法 JSON

**用例：**  
- `config.test.ts`：`saveConfig refuses to overwrite`、`resetBrokenConfig keeps the original`  
- `review-fixes.test.ts`：`F4: rag_index does not overwrite a broken config.json`

### F5 · 三组评估执行路径（原 R14）

**改动：** `scripts/eval-retrieval.ts` 四组均可选择执行：

| 组 | 何时跑 | 何时 not-run |
| --- | --- | --- |
| `local-bm25` | 默认 | — |
| `local-hybrid` | 未设 `SKIP_EMBEDDING_TESTS` | `SKIP_EMBEDDING_TESTS=1`（路径已实现） |
| `cloud-embedding` | 有 `VOYAGE_API_KEY` 且未设 `EVAL_SKIP_CLOUD` | 无 Key 或 `EVAL_SKIP_CLOUD=1` |
| `cloud-embedding+reranker` | 同上 | 同上 |

```bash
npm run eval:retrieval
npm run eval:retrieval -- --groups=local-bm25,local-hybrid
npm run eval:retrieval -- --corpus=/path/to/docs
```

无 Key 时云组 reason 为 `VOYAGE_API_KEY is not set`，有 Key 时应真正建库+查询，而不是恒 `not-run`。

**仍开放：** 本环境未跑 MiniLM hybrid 与真实 Voyage 组。有 Key 后请 Astra 实跑云组。local-bm25 数字是 FTS 管道指标，不是模型质量。

### F6 · 指标口径

**改动：** 候选池 `hybridSearch(..., 30)`；最终 top5 另算。`recallAtCandidates` 看候选池的**文件/snippet**命中；`hitAt5` / `mrrAt5` 看 top5 的内容命中。`sourceAccuracy`：仅对「有页标注且 top5 已内容命中」的问题计分；页码错误进分母（不为 1）。

报告 JSON 含 `metricDefinitions`。此次打印的 local-bm25 ≈ 0.227 仍**不能**当作三组模型效果或验收分数。

### F7 · Voyage HTTP 配置被缓存吞掉（原 R16 引入）

**改动：** `providers/embedding/factory.ts` 云 provider 缓存键含 `timeoutMs`、`maxRetries`。本地 pipeline 仍按 model/dim 复用。`VoyageEmbeddingProvider.timeoutMs` / `maxRetries` 只读可见。

**期望：** 先 30000ms/3 再 1000ms/0，两个实例不同；后者 `timeoutMs===1000`、`maxRetries===0`。

**用例：** `review-fixes.test.ts`  
`F7: Voyage HTTP timeout/retry changes produce a new provider`

复验探针 `http_config_cache` 期望 `sameInstance: false`。

---

## 3. 首轮 R1–R16 在本树中的状态（实现方声明）

请以探针为准，下表只说明代码意图。

| 编号 | 实现方声明 | 主要落点 / 用例 |
| --- | --- | --- |
| R1 | 解析失败计入 `failed`，阻止 publish | `review-fixes` R1 |
| R2 | force 用独立 generation；失败不 prune 旧记录 | `review-fixes` R2 |
| R3 | 云模式默认不自动刷新 | `review-fixes` R3 |
| R4 | 估算上限见 F1；tokenizer 未关 | `parsing.test.ts` |
| R5 | 无 fingerprint 拒绝查询 | `review-fixes` R5 |
| R6 | 空库也查表维度；新增真正空库用例 | `review-fixes` F6 |
| R7 | unique staging，不删旧 generation | `review-fixes` R7 |
| R8 | 信号贯通；运行中取消见 F3 | `review-fixes` R8 + F3 |
| R9 | 命令/tool 输出 failed、degraded、rerank；hook 打印 refresh failed；context header 可含 Retrieval note | `retrieval.test.ts` degraded note |
| R10 | 临时 embed 失败 → BM25 + `degraded` | `review-fixes` R10 |
| R11 | page/section/id/chunkIndex 贯通；未知行号不输出 | `review-fixes` R11 |
| R12 | Markdown 短章节保留 | `parsing.test.ts` short heading |
| R13 | strip-types 可启动 smoke | 无 Key smoke |
| R14 | 见 F5/F6 | `eval:retrieval` |
| R15 | 诊断 + 可写入口拒绝损坏文件，见 F4 | `config.test.ts` |
| R16 | 本地复用 pipeline；云 HTTP 键见 F7 | R16 + F7 |

其他一并改过、方便顺带看的点：

- 自动注入 12s deadline（`AUTO_INJECT_DEADLINE_MS`）；invalid config 时跳过注入。
- HTML/DOCX 不把转换后文本行号当成原始位置。
- `files.document_id` / `title` 列。
- HTTP 退避响应 abort（`abort.ts` + `providers/http.ts`）。

---

## 4. 建议 Astra 的复验顺序

1. **F1 / F2**（纯函数，最快）：跑 `parsing.test.ts` 或复验探针的 `overlap_overflow`、`same_text_distinct_pages`。
2. **F3 / F4 / F7**：复验探针后半段 + `review-fixes.test.ts` 对应项。
3. **F5 / F6**：读 `eval/runs/eval-2026-09-21.json` 的 `runs[].status/reason` 与 `metricDefinitions`；有 Key 时去掉 `EVAL_SKIP_CLOUD` 跑云组。
4. **回归：** `SKIP_EMBEDDING_TESTS=1 npm test`、两个 smoke、`git diff --check`。
5. 若时间够：临时目录跑 `/rag rebuild --force` 失败恢复（R2 命令级仍弱于 helper 级）。

---

## 5. 实现方认为仍不能整体验收的原因

- 无真实 ONNX / Voyage / Pi 会话证据。
- E1 三页 PDF、论文抽查未做。
- E2 没有模型 tokenizer 合同。
- E4 云组与 local-hybrid 在本环境未实跑。
- R2 完整命令+进程重启失败恢复证据仍偏 helper 测试。
- 扫描失败 vs 合法空文件集合仍无单独验收用例（复验 R1 备注）。

A1–E4 建议按 `docs/implementation-report.md` 第二节阅读：C1/C2/D\* 为 **done (offline)**，E1/E2/E4 为 **partial**。不要把「代码存在 + 离线全绿」写成计划已满足。

---

## 6. 现有索引

processing fingerprint `chunker` 现为 `token-v3`。旧库会 `needsRebuild`。复验请用临时 `PI_RAG_DIR`，不要在用户已有库上 `--force`。
