# Grok Build T1–T4 修复验收说明（供 Astra 复验）

日期：2026-09-22。  
对象：`feat/cloud-research-rag` 工作区相对 `d546804` 的**未提交**改动（含前两轮 R1–R16 / F1–F7 修复，以及本轮 T1–T4）。  
对照：`grok-build-acceptance-review-2026-09-21.md`。

本文件列出本轮声称修了什么、用哪条命令复现、期望看到什么。请用独立临时目录和阻断网络的探针复核；不要在用户全局 `~/.pi/rag` 上跑。离线全绿不能替代探针。

**实现方自测（不能替代 Astra 探针）：**

| 检查 | 实现方结果 |
| --- | --- |
| `npm run typecheck` | 通过 |
| 隔离 `PI_RAG_DIR`、`SKIP_EMBEDDING_TESTS=1 npm test` | **194 passed / 4 skipped** |
| 前两轮 F1/F2/F3/F7 用例 | 仍绿 |
| 本轮新增 | `review-fixes` T1/T3；`config.test` 字符串/数字 boolean；`retrieval.test` rerank-fallback |

未做：真实 ONNX、真实 Voyage、Pi 完整会话、干净安装、三页 PDF、论文效果、不可读目录的 OS 级 chmod 验收。

---

## 1. 工作区范围

- 分支：`feat/cloud-research-rag`
- 基线 HEAD：`d546804`
- 状态：工作区脏，**未 commit、未 push、未 publish**

```bash
cd /Users/chouchao/Documents/RAG_Knowledge_Repository/pi-local-rag
git rev-parse HEAD   # 期望 d546804
git status           # 应看到未提交修复
```

建议复验命令：

```bash
export PI_RAG_DIR=$(mktemp -d /tmp/pi-rag-astra-t-XXXX)
export PI_RAG_LEGACY_DIR="$PI_RAG_DIR/legacy"
unset VOYAGE_API_KEY
SKIP_EMBEDDING_TESTS=1 npm test
npm run typecheck

node --experimental-strip-types docs/reviews/grok-build-acceptance-entry-probe.mjs
node --experimental-strip-types docs/reviews/grok-build-acceptance-eval-probe.mjs
```

`entry-probe` 覆盖 T1/T2/T3（以及既有 R2 命令失败保护）。`eval-probe` 覆盖 T4。

---

## 2. T1–T4：改动、期望、用例

### T1 · P1：跟踪根目录暂时不可用被当成删除

**改动：** `chunking.ts` 新增 `collectFromTrackedDetailed` / `collectFromTrackedDetailedAsync`，返回 `{ files, errors, unavailableRoots }`。根路径 `stat`/`readdir` 失败记入 errors，不再 `continue` 成空扫描。  
`index.ts` 的 `/rag rebuild --force` 与 `/rag refresh`：`scan.errors.length > 0` 时通知并 **return**，不构造 droppedFiles、不调用 `rebuildWithSwitch`。自动注入刷新同样跳过。

**期望（命令级，对齐第三轮复现）：** 为跟踪目录中两个文件建库 → 将目录 rename 模拟挂载点消失 → `/rag rebuild --force`：

- 通知含 `unavailable` / `refusing`
- **不**出现 `2 deleted` / `0 failed` 的成功摘要
- 不发布指向空库的 `active.json`（或至少活动查询 chunks 仍为 2）
- `closeDbConn()` 后再 `getIndexStats().totalChunks` 与重建前相同

成功扫描后的真实删除（文件确实没了、根目录仍在且可读）仍允许进入 droppedFiles。

**用例：**  
- `__tests__/index.test.ts`：`reports unavailable tracked roots separately from a successful empty scan`  
- `__tests__/review-fixes.test.ts`：`T1: a missing tracked root does not publish an empty index`

**仍未单独验收：** 不可读目录（EACCES）、子目录局部 `readdir` 失败的命令级 chmod 场景。代码路径会把这类错误写入 `errors` 并同样拒绝；请探针补一项若需要关闭该备注。

### T2 · P2：字符串 `"false"` 误开云自动刷新

**改动：** `validateConfig` 要求 `cloudAutoRefresh`、`ragEnabled` 为 `typeof === "boolean"`。`shouldAutoRefresh` 仅当 `cloudAutoRefresh === true` 时对非 local embedding 开放刷新。`requireWritableConfig` 把类型错误视为 blocking（合法 JSON 但类型错也会拒绝写入入口）。不把 `"false"` 静默coerce 成 `false`。

**期望：**

```js
validateConfig({ ...defaultConfig(), cloudAutoRefresh: "false" })
// 含 cloudAutoRefresh must be a boolean

shouldAutoRefresh(voyageCfgWithStringFalse, staleStats) === false

loadConfigDetailed().issues 非空
requireWritableConfig() 抛 ConfigFileInvalidError
```

hook：Voyage 配置 + `"cloudAutoRefresh":"false"` + 过期 lastBuild + 改文档后 `before_agent_start`，HTTP stub **不应**出现 `input_type=document`。

**用例：** `__tests__/config.test.ts`  
`string false is not a valid cloudAutoRefresh value`  
`numeric cloudAutoRefresh is a type issue`

entry-probe 的 `invalid_boolean_cloud_refresh`、`string_false_cloud_hook` 应对齐。

### T3 · P2：`/rag search` 遗漏损坏配置保护

**改动：** `/rag search` 改为 `loadConfigDetailed()`；`fileStatus === "invalid"` 时 `notify` 错误并 return，不调用 `retrieve`、不 `setWidget`。与 `rag_query` 同一套文案（repair / `config reset`）。

**期望：** 已有兼容本地索引时写入 `{BROKEN`，`/rag search alpha`：

- 有配置错误通知
- 无结果 widget
- `config.json` 仍为 `{BROKEN`

`rag_index` 不覆盖损坏文件（第三轮已通过，应保持）。

**用例：** `__tests__/review-fixes.test.ts`  
`T3: /rag search refuses a broken config.json`

entry-probe `broken_config_command_search`：`notes` 含 invalid，不应再出现两条查询命中 widget。

### T4 · P2：评估把重排失败报成成功

**改动：** `retrieveWithCandidates()` 一次 hybridSearch（必要时一次 rerank），返回 `{ hits, candidates, degraded, method }`。`method` 为 `hybrid` | `bm25-fallback` | `rerank` | `rerank-fallback`。  
eval 每题只调这一次；`perQuestion` 含 `degraded`、`method`。组状态：

| 条件 | `status` |
| --- | --- |
| 无降级 | `ok` |
| 部分题目降级 | `degraded`，`reason` 含比例，`degraded` 为计数 |
| 全部题目降级 | `error`，`reason` 含 `all N questions degraded` |

配置里的 reranker 写在 `configuredReranker`，不再假装实际跑了 rerank。

**期望（eval-probe）：** embedding stub 成功、全部 rerank HTTP 400 时：

- `cloud-embedding+reranker.status` 为 `error`（或至少不是无字段的 `ok`）
- 有 `reason` / `degraded`
- `perQuestion` 带 `degraded` 与 `method: "rerank-fallback"`
- 合成分数不可当作模型质量

**用例：** `__tests__/retrieval.test.ts`  
`records rerank-fallback when the reranker HTTP call fails`

**仍弱：** 没有单独的「部分失败 → status=degraded」集成探针；规则已实现，请 eval-probe 或临时 stub 只让半数 rerank 失败来补。

---

## 3. 第三轮对其它项的态度（实现方不重开）

第三轮已判定 F1/F2/F3/F7 原反例通过；R2 命令路径（embedding 抛错 + `--force`）已通过。本轮未回退这些修复。F5 执行路径仍在；本环境无 Key / `SKIP_EMBEDDING_TESTS=1` 时云组与 local-hybrid 仍为 `not-run`。F6 口径保持候选 30 + top5、内容与页码分离。

---

## 4. 仍不能整体验收的原因

与前两轮披露一致，外加：

- T1 的 EACCES / 局部子树失败未做 chmod 命令级实锤（ENOENT 根消失已有命令级测试）。
- T4 全失败路径有单测 + 应对齐 eval-probe；部分失败组状态缺专用探针。
- 无真实 ONNX / Voyage / Pi 会话 / 三页 PDF / tokenizer 合同。

C1/C2/D\* 仍是 **done (offline)**。E1/E2/E4 仍是 **partial**。不要把「代码存在 + 离线 194 passed」写成计划已满足。
