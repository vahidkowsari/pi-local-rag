# T1–T4 第四轮复验

日期：2026-09-22。审查当前未提交工作区，对照实现方 `grok-build-acceptance-t1-t4-2026-09-22.md`。生产源码、测试和实现方报告未修改。

**结论：T1/T2/T3 的原问题可关闭；T4 的重排失败分支已修复，但空候选时的 embedding 故障仍被隐去，暂不能关闭整个降级可观察性问题。** 本轮没有复现前几轮的分块越界、跨页去重、取消后发布、HTTP 配置缓存问题。不应把已通过的修复再次退回。

## 剩余问题：U1 · P2 空候选时丢弃 embedding 失败原因

位置：`search.ts:108–116`，`retrieval.ts:74,84–94`。

`hybridSearch` 捕获临时 embedding 故障后保存 `embedDegraded`，但没有 FTS 候选时直接返回 `[]`。新 `retrieveWithCandidates` 只从 `hits[0]?.degraded` 恢复错误状态，空数组无法携带该信息，最终返回 `degraded: undefined, method: "hybrid"`。这让“服务失败且没有词法候选”和“检索正常完成但无结果”无法区分。

**独立复现：** 用已建好向量的临时索引查询不存在的词，并使 query embedding 抛 `TypeError('synthetic network outage')`。实际输出：

```json
{"hits":0,"candidates":0,"degraded":null,"method":"hybrid"}
```

**评估影响也已实跑：** 文档 embedding 成功，云组每个 query embedding 均抛网络异常；两组各有 23 个问题，所有查询向量请求均失败，但报告各只统计 **6/23 degraded**，其余 17 个无 FTS 候选的问题未标失败。组状态为 degraded，而实现方约定的全题降级应为 error。无答案题也可能把该故障当作正常的无命中成功，污染指标。

修复要求：让搜索层返回独立于命中数组的结果合同，例如 `{ hits, degraded, method }`，或在无可用降级结果时抛出可识别故障。不要再从第一条 hit 推断整个请求是否失败。让空结果的工具输出和评估也保留诊断；补“已有向量 + query 故障 + FTS 零命中”的 bundle 与整组评估用例。

## T1–T4 的独立复验

| 项 | 结果 | 实际证据 |
| --- | --- | --- |
| T1 根目录不存在 | 通过 | rename 临时跟踪目录后执行 `/rag rebuild --force`，报 unavailable/refusing；活动 chunks 保持 2，未发布 manifest |
| T1 根目录不可读 | 通过 | 对临时目录实际 chmod 000，命令报 not readable，旧 chunks 保持 2 |
| T1 子目录局部不可读 | 通过 | 对临时子目录 chmod 000，命令拒绝发布，旧 chunks 保持 2 |
| T2 字符串 false | 通过原问题 | 先用合法配置建库，再把 cloudAutoRefresh 改成字符串 false；配置有 boolean issue，hook 的 HTTP stub 仅收到 query，没有 document |
| T3 损坏 JSON 的命令搜索 | 通过 | `{BROKEN` 时有 invalid/repair 通知，无结果 widget |
| T4 重排全部失败 | 通过 | 23 次 rerank 全 HTTP 400 → status=error，reason=all 23 questions degraded，degraded=23 |
| T4 重排部分失败 | 通过 | 交替失败 12 次、成功 11 次 → status=degraded，degraded=12 |
| T4 重排全部成功 | 通过 | 23 次 stub 重排成功 → status=ok |
| T4 query embedding 故障、无 FTS | 不通过 | 见 U1，空结果丢失故障状态 |

chmod 只涉及探针新建临时目录，并在 finally 恢复为 0700。根目录和子目录权限错误的声明缺口已通过真实文件系统权限检查补齐，不只是 mock。没有模拟整机崩溃或多进程并发。

原 entry-probe 的云配置准备段会在“拿字符串 false 配置建库”时被新增校验提前拒绝，这属于旧探针准备方式失效。本轮调整为先合法建库、再注入错误配置，以检验真正的 hook 行为，没有修改生产逻辑绕过校验。

## 其他实际检查

- 全套测试：**194 passed / 4 skipped**，10 个文件通过、1 个跳过；存储、legacy 和 npm cache 均隔离在 `/private/tmp/rag-fourth/`。
- `npm run typecheck`：通过。
- `git diff --check`：通过。
- 旧 `grok-build-recheck-probe.mjs` 复跑：F1 输出估算 [100,240]，F2 保留两页，F3 取消抛 AbortError 且保留旧正文，F4 索引工具不覆盖损坏文件，F7 新 HTTP 参数生效。
- R2 命令失败保护仍通过：embedding 抛错后不删旧记录，连接重开后两个 chunks 保留。
- 新评估候选/最终排名复用单次查询：全失败重排探针中云 query 总请求从上轮 69 降至 46，即两云组各 23 次。

模型与 HTTP 都使用离线替身，合成分数不可作为质量评估。本轮没有重新运行无 Key smoke 或打包，没有执行真实 ONNX、Voyage、Pi 完整会话、干净安装、三页 PDF 或论文检索效果；也没有补齐真实 tokenizer 合同。实现方对这些边界的披露仍成立，故不能宣布原 A1–E4 全部验收通过。

## 材料

- `grok-build-t1-t4-review-evidence.jsonl`：正向复验及 U1 的证据。
- `grok-build-t1-t4-entry-probe.mjs`：临时索引、命令、权限、hook 和空候选故障探针。
- `grok-build-t1-t4-eval-probe.mjs`：success / partial / empty 三种场景；执行后恢复原当日 eval 报告。
- 原始日志：`/private/tmp/rag-fourth/`。

在仓库根目录运行：

```bash
node --experimental-strip-types docs/reviews/grok-build-t1-t4-entry-probe.mjs
PROBE_MODE=success node --experimental-strip-types docs/reviews/grok-build-t1-t4-eval-probe.mjs
PROBE_MODE=partial node --experimental-strip-types docs/reviews/grok-build-t1-t4-eval-probe.mjs
PROBE_MODE=empty node --experimental-strip-types docs/reviews/grok-build-t1-t4-eval-probe.mjs
```

全重排失败场景仍用上一轮 `grok-build-acceptance-eval-probe.mjs` 复现。建议仅针对 U1 补修复，然后进入已列明的真实环境验收，不必重新返工本轮已关闭项。
