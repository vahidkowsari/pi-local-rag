# 第三轮代码复验

对象：`grok-build-acceptance-2026-09-21.md` 所描述的未提交工作区；基线 HEAD 仍为 `d546804`。本次只审查和生成证据，没有修改生产源码或实现方报告。

**结论：上轮针对性修复多数已经成立；仍不能整体验收。** F1/F2/F3/F7 的原反例已通过，F4 的工具路径已修复但命令路径遗漏。F5 不再是恒定 not-run，实际存在模型与云执行路径；F6 的候选池及页码分母逻辑已有修正。不过本轮另有 4 项已复现问题，其中跟踪根目录不可用时发布空库应优先修复。

## 当前问题

### T1 · P1：跟踪根目录暂时不可用会被当作删除，发布空活动索引

位置：`chunking.ts:402–408`，`index.ts:368–384`。

`collectFromTrackedAsync` 对不存在的根路径直接 continue，扫描过程没有结构化错误结果；重建再通过 `existsSync` 去掉对应旧文件，将它们全部列入 droppedFiles。已有索引时，空 targetFiles 不触发提前返回，最终发布空 staging。

**命令级复现：** 先为一个跟踪目录中的两个合成文件建库，再将目录改名模拟挂载点暂时消失，调用 `/rag rebuild --force`。输出 `0 re-indexed / 2 deleted / 0 failed`，生成 active.json，关闭并重新打开连接后 **chunks 从 2 变为 0**。旧数据库文件仍在磁盘，但活动查询已无内容。不是“目录为空且已成功扫描”的情形。

实现方报告已承认“扫描失败 vs 合法空集合”未验收；本轮把该缺口复现为实际缺陷。

修复：扫描返回 files 与 errors/不可用根目录信息；必要根目录不存在、无法 stat/readdir 时禁止删除其旧记录及发布替代索引。明确区分成功扫描后的真实删除与暂时不可访问。加入根目录消失、不可读目录、局部扫描失败的命令级测试。

### T2 · P2：字符串 `"false"` 被当作开启云自动刷新，配置校验无报错

位置：`config.ts` 的 `mergeSaved` / `validateConfig`；`indexing.ts` 的 `shouldAutoRefresh`；`index.ts:119–154`。

合法 JSON 中写入 `"cloudAutoRefresh": "false"`，当前类型断言及校验没有拒绝字符串；运行时 `!config.cloudAutoRefresh` 为 false，因此允许自动刷新。给合成 Voyage 配置设置该值、把索引时间置旧并修改文档，执行真实 hook handler，HTTP stub 收到 **[document, query]**，同时 `loadConfigDetailed().issues=[]`。

这会让用户的配置错误表现成普通对话触发文档上传/计费。`requireWritableConfig` 及 hook 目前主要检查 JSON 解析状态，不能替代运行时类型及语义校验。

修复：校验布尔字段真实类型，云刷新仅对 `=== true` 开放；业务入口明确处理相关配置 issues。补字符串 false、数字及非法环境值测试。需要保留只读诊断能力，不能静默把错误值视为有效配置。

### T3 · P2：`/rag search` 遗漏损坏配置保护（F4 未全部关闭）

位置：`index.ts:295–301`。

`rag_query`、`rag_index`、hook 已处理 invalid JSON，但 `/rag search` 仍直接使用 `loadConfig()`。在已有兼容本地索引上写入 `{BROKEN`，调用 `/rag search alpha`，命令正常展示两条查询结果，**没有任何配置错误通知**。

修复：命令查询和工具查询复用同一配置检查，损坏配置时提示修复/显式 reset；不应由入口不同决定是否静默使用默认值。此次没有再复现普通索引工具覆盖损坏文件的问题，该部分可关闭。

### T4 · P2：评估丢弃降级标记，把失败的重排实验报告为成功

位置：`scripts/eval-retrieval.ts:139–145,169–178,274–282`。

在线检索允许降级是合理的，但实验报告必须记录实际运行的方法。`scoreGroup` 仅提取排名/命中数据，丢弃 `RetrievedChunk.degraded`；`runGroup` 直接输出 `status:ok` 和配置指定的 reranker。向量查询失败后退回 BM25 同样可能发生此问题。

**离线执行四组的探针：** 用确定性向量替换本地模型，HTTP stub 为云 embedding 返回合法向量、为所有 rerank 请求返回 HTTP 400。实际收到 **23 次 rerank 请求且全部失败**，最终 `cloud-embedding+reranker` 仍为 **ok**，没有 reason/degraded 字段，逐问题记录也没有降级信息。这证明云执行分支已存在，同时暴露其失败报告不可信；这些合成数值不是模型质量结果。

修复：逐问题保存降级原因及实际检索方式；汇总降级次数，并将实验组标为 degraded/error 或按明确规则排除失败样本。候选集合与最终排名最好复用同一次查询，避免当前 reranker 组的两次 embedding 查询在故障时对应不同候选池。用 stub 验证全失败、部分失败、全部成功三种状态。

## 对上轮 F1–F7 的判断

| 项 | 当前结论 | 本次依据 |
| --- | --- | --- |
| F1 | 原反例已修复 | token 估算从 [100,256] 变为 [100,240]；真实模型 tokenizer 合同仍未完成 |
| F2 | 原反例已修复 | 相同正文的第 1、2 页各保留一个 chunk，context 去重包含来源身份 |
| F3 | 原反例已修复 | 最后一批 embedding 期间 abort 后抛 AbortError，未发布 manifest，旧正文保留 |
| F4 | 部分修复 | rag_query 提示 invalid，rag_index 不覆盖；命令 search 遗漏，见 T3 |
| F5 | 执行路径已实现，未获真实模型验收 | 离线替身真正跑过四组；真实 ONNX/Voyage 没有执行。外部 corpus 已支持，但问题标注仍固定为仓库 questions.json，外部语料评估还需匹配标注入口 |
| F6 | 原两个指标缺陷已在代码中修正 | 使用候选 30 与最终 5，内容匹配与页码判断分离；仍需有针对性的指标 fixture，且 T4 阻止把组结果当模型验收 |
| F7 | 原反例已修复 | 30000ms/3 改为 1000ms/0 后实例不同，新配置值生效 |

## 实际验证及边界

- TypeScript typecheck：通过。
- 第一次全套测试：187 passed / 1 failed / 4 skipped；失败是测试中的 npm pack 使用默认缓存遇权限问题。
- 将 npm cache 一并隔离到 `/private/tmp/rag-third/npm` 后：**188 passed / 4 skipped**，10 个测试文件通过、1 个跳过。未修改测试或放宽全局权限。
- 两个无 Key smoke：正常启动，exit 0，明确 UNVERIFIED。
- `SKIP_EMBEDDING_TESTS=1` 且无 Key 的 eval：local-bm25=ok，其余三组 not-run；这与实现方报告一致。
- 额外命令级失败保护：删除一个已索引文件、使剩余文件 embedding 抛错，再执行 `/rag rebuild --force`；命令输出 1 failed，旧两个 chunks 在连接重开后仍在，未创建 manifest。**这条 R2 命令路径已通过，不能再称只有 helper 证据。** 本次重开的是连接，不等同于整个 Pi 进程重启。
- `git diff --check`：通过。

实现方报告对真实 ONNX、Voyage、Pi 完整会话、干净安装、三页 PDF、tokenizer 和论文效果等未完成项的披露基本准确；无需把已经修好的 F1/F2/F3/F7 重新退回。应优先解决 T1–T4，再做其列明的集成与效果验收。

未验证真实云请求、真实本地模型、三页 PDF 或论文效果；本次云路径验证完全使用 HTTP stub，没有发送用户文档。

## 证据与重现

- `grok-build-acceptance-review-evidence.jsonl`：原探针复跑及本轮追加证据。
- `grok-build-acceptance-entry-probe.mjs`：T1/T2/T3 及 R2 命令失败保护，临时合成库。
- `grok-build-acceptance-eval-probe.mjs`：T4，合成模型/HTTP，执行后恢复原当日评估文件，避免混入质量数据。
- 原始测试、smoke/探针日志在 `/private/tmp/rag-third/`；忽略目录 eval/runs 的当日报告已通过无 Key、跳过 ONNX 的真实 BM25 运行更新。

在仓库根目录执行：

```bash
node --experimental-strip-types docs/reviews/grok-build-recheck-probe.mjs
node --experimental-strip-types docs/reviews/grok-build-acceptance-entry-probe.mjs
node --experimental-strip-types docs/reviews/grok-build-acceptance-eval-probe.mjs
```
