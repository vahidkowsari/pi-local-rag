# U1 第五轮复验

日期：2026-09-22。对象：当前未提交代码及 `grok-build-acceptance-u1-2026-09-22.md`，沿用前轮审查范围。

**结论：U1 可以关闭。本轮针对 U1 及关联回归未发现新增阻塞问题。** 搜索层现在独立返回降级状态，零命中不再丢失 embedding 故障。工具、命令和评估均已接入该结果。

这只代表本轮缺陷修复通过离线复验，不代表原 A1–E4 项目已整体验收。

## 独立实跑结果

| 检查 | 结果 |
| --- | --- |
| 已有向量、query embedding 故障、FTS 零命中 | hits=0、candidates=0，仍保留 `query embedding failed`，method=`bm25-fallback` |
| `rag_query` 零命中故障输出 | 文本含 degraded、故障原因和 method，能够区别正常无结果 |
| `/rag search` 零命中故障输出 | error 级通知含 degraded 原因，无结果 widget |
| 两云组所有 query embedding 失败 | 各 23/23 degraded、status=error、reason=`all 23 questions degraded` |
| 每题诊断 | 两云组所有题均有非空 degraded，method 均为 `bm25-fallback` |
| 无答案题 | 两云组 unanswerable=0，不再把请求故障当作正常无命中成功 |
| 重排部分失败 | 12/23 degraded，组状态 degraded |
| 重排全部成功 | 组状态 ok |
| 重排全部失败 | 23/23 degraded，组状态 error |
| 全套测试 | **195 passed / 4 skipped**；10 个测试文件通过、1 个跳过 |
| `npm run typecheck` | 通过 |
| `git diff --check` | 通过 |

回归探针还确认：跟踪根目录消失、根目录或子目录 chmod 000 时，重建拒绝发布且旧 chunks 保持 2；字符串 false 配置只触发 query、不触发 document；损坏 JSON 的命令搜索被拒绝；命令级 embedding 失败仍保留旧索引。无需重新退回这些已关闭项。

## 核对代码

- `search.ts`：`hybridSearchDetailed` 在空结果及非空结果路径均独立返回 degraded；旧数组接口保留为包装器。
- `retrieval.ts`：读取 `searched.degraded`，不再依赖第一条 hit；空候选的 embedding 故障标为 bm25-fallback。
- `index.ts`：命令及工具使用 bundle，零结果时保留故障提示。
- `scripts/eval-retrieval.ts`：按 bundle 记录每题诊断，无答案题要求没有 degraded 才能计为成功。

## 验证边界与交付状态

所有索引与查询探针使用临时合成文档、确定性模型替身和 HTTP stub，没有发送用户文档或请求真实 Voyage。测试存储、legacy 路径和 npm cache 隔离于 `/private/tmp/rag-u1/`。评估探针执行后恢复原当日 eval 文件，合成数值不留作模型质量报告。

本轮没有验证真实 ONNX、真实 Voyage、Pi 完整会话、干净安装、三页 PDF、模型 tokenizer 合同或论文检索效果，也未重新运行 smoke/pack。实现方报告对这些限制的披露准确。应转入这些尚未完成的集成及质量验收，E1/E2/E4 仍不能仅凭本轮通过改为整体完成。

生产源码、测试及实现方报告未修改；没有 commit、push 或发布。

## 证据

- `grok-build-u1-review-evidence.jsonl`：U1、工具输出、整组评估及关联回归结果。
- `grok-build-u1-entry-probe.mjs`：在上轮入口探针上补充工具与命令零命中故障检查。
- `grok-build-u1-eval-probe.mjs`：额外输出每题诊断覆盖、实际 method 集合及无答案题指标。
- 原始日志：`/private/tmp/rag-u1/`。

在仓库根目录运行：

```bash
node --experimental-strip-types docs/reviews/grok-build-u1-entry-probe.mjs
PROBE_MODE=empty node --experimental-strip-types docs/reviews/grok-build-u1-eval-probe.mjs
PROBE_MODE=partial node --experimental-strip-types docs/reviews/grok-build-u1-eval-probe.mjs
PROBE_MODE=success node --experimental-strip-types docs/reviews/grok-build-u1-eval-probe.mjs
```

重排全部失败仍用 `grok-build-acceptance-eval-probe.mjs` 验证。
