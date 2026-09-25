# Grok Build U1 修复验收说明（供 Astra 复验）

日期：2026-09-22。  
对象：`feat/cloud-research-rag` 工作区相对 `d546804` 的**未提交**改动。  
对照：`grok-build-t1-t4-review-2026-09-22.md` 的 **U1**。

第四轮已关闭 T1/T2/T3 以及 T4 的重排成功/部分失败/全部失败。本轮只针对 U1（空候选时丢掉 query embedding 失败）。请不要把已通过项退回。

本文件不是“已验收通过”证明。请用独立临时目录和阻断网络的探针复核；不要在用户全局 `~/.pi/rag` 上跑。

**实现方自测（不能替代 Astra 探针）：**

| 检查 | 实现方结果 |
| --- | --- |
| `npm run typecheck` | 通过 |
| `SKIP_EMBEDDING_TESTS=1 npm test` | **195 passed / 4 skipped** |
| 新增用例 | `__tests__/retrieval.test.ts`：`keeps embedding failure when FTS has zero hits` |

未做：真实 ONNX、真实 Voyage、Pi 完整会话、干净安装、三页 PDF、tokenizer、论文效果。本轮未重跑无 Key smoke / pack。

---

## 1. 工作区范围

- 分支：`feat/cloud-research-rag`
- 基线 HEAD：`d546804`
- 状态：工作区脏，**未 commit、未 push、未 publish**

```bash
cd /Users/chouchao/Documents/RAG_Knowledge_Repository/pi-local-rag
git rev-parse HEAD   # 期望 d546804
git status
```

建议复验：

```bash
export PI_RAG_DIR=$(mktemp -d /tmp/pi-rag-astra-u1-XXXX)
export PI_RAG_LEGACY_DIR="$PI_RAG_DIR/legacy"
unset VOYAGE_API_KEY
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck

node --experimental-strip-types docs/reviews/grok-build-t1-t4-entry-probe.mjs
PROBE_MODE=empty node --experimental-strip-types docs/reviews/grok-build-t1-t4-eval-probe.mjs
```

entry-probe 末段 `query_failure_no_fts` 是 U1 的原反例。eval-probe `PROBE_MODE=empty` 是整组评估反例。重排三态仍可用上一轮 eval-probe 的 success/partial 以及 `grok-build-acceptance-eval-probe.mjs`（全失败）。

---

## 2. U1：空候选时丢弃 embedding 失败原因

**原反例：** 已有向量的索引 + query embedding 抛 `TypeError('synthetic network outage')` + FTS 零命中，得到：

```json
{"hits":0,"candidates":0,"degraded":null,"method":"hybrid"}
```

评估侧：云组 23 题全部 query embedding 失败，却只统计 6/23 degraded（有 FTS 命中的题），组状态 `degraded` 而非全题失败的 `error`。无答案题还可能把故障当成正常零命中。

**改动：**

| 文件 | 行为 |
| --- | --- |
| `search.ts` | 新增 `hybridSearchDetailed()` → `{ hits, degraded }`。`degraded` 与 `hits.length` 独立。无 FTS/无向量命中时仍返回该字段。`hybridSearch()` 仍返回数组，供旧测试。 |
| `retrieval.ts` | `retrieveWithCandidates` 读 `searched.degraded`，不再用 `hits[0]?.degraded`。空 hits 且有 embedding 故障时 `method` 为 `bm25-fallback`。 |
| `index.ts` | `/rag search`、`rag_query` 空结果仍输出 `degraded` / `method`。 |
| `scripts/eval-retrieval.ts` | 每题记录 `bundle.degraded`；无答案题在 `degraded` 时不计入 unanswerable 成功。组状态规则不变：全题 degraded → `error`。 |

**期望（bundle / entry-probe `query_failure_no_fts`）：**

```json
{"hits":0,"candidates":0,"degraded":"<含 query embedding failed>", "method":"bm25-fallback"}
```

`degraded` 不得为 `null`；`method` 不得为无诊断的 `"hybrid"`。

**期望（eval-probe `PROBE_MODE=empty`）：** 文档 embedding 成功、每个 query embedding 均失败时：

- 两组（或该模式下的云组）`degraded` 计数 = 问题数（23，含无答案题）
- `status` 为 `error`
- `reason` 含 `all 23 questions degraded`（或等价的全量失败描述）
- `perQuestion` 每条都有非空 `degraded` 和 `method: "bm25-fallback"`
- 合成分数不可当作模型质量

**期望（工具空结果）：** `rag_query` / `/rag search` 在零命中 + embedding 故障时，文本或 notify 含 `degraded`，与“正常无结果”可区分。

**用例：** `__tests__/retrieval.test.ts`  
`keeps embedding failure when FTS has zero hits`

---

## 3. 第四轮已关闭项（本轮未回退）

| 项 | 第四轮结论 | 本轮 |
| --- | --- | --- |
| T1 根不存在 / 不可读 / 子树不可读 | 通过 | 未改生产语义 |
| T2 字符串 false | 通过 | 未改 |
| T3 损坏 JSON 的 `/rag search` | 通过 | 未改（search 现多用 `retrieveWithCandidates`，invalid JSON 仍先拦截） |
| T4 重排全失败 / 部分失败 / 全成功 | 通过 | 未改组状态规则 |
| F1–F3、F7、R2 命令失败保护 | 通过 | 未改 |

---

## 4. 仍不能整体验收的原因

与前几轮披露一致：

- 无真实 ONNX / Voyage / Pi 会话 / 三页 PDF / tokenizer 合同。
- 合成分数不是质量评估。
- E1/E2/E4 仍为 partial；C/D 为 done (offline)。

不要把「U1 代码存在 + 195 passed」写成 A1–E4 已全部验收。
