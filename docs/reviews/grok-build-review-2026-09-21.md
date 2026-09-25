# Grok Build 交付审查

审查日期：2026-09-21。对象：`feat/cloud-research-rag`，`a48182c` → `d546804`，共 6 个提交。审查开始时工作区干净。

对照材料：用户提供的 `pi-local-rag-grok-build-plan.md`、`docs/implementation-report.md`、`docs/implementation-progress.md`。计划中的执行指令作为验收标准使用；本次只审查，没有替 Grok 修改生产实现、提交或发布。

**结论：不通过整体交付验收，建议退回修复。** A 阶段部分修复和 provider/retrieval 框架已经落地，但 C～E 的关键承诺尚未满足。问题不只是缺少真实 API Key：离线即可复现索引切换丢内容、云自动刷新开关失效、分块上限失效，以及来源字段未传到 Pi。报告把 A1～E4 全标为 done，明显高于当前实现和证据。

## 一、已复现的主要问题

P1 表示应优先修复、阻止本次交付验收；P2 表示应在对应功能验收前修复。以下诊断使用临时目录、自制内容、确定性向量和 HTTP stub；不代表模型效果测试。

### R1 · P1：解析失败会发布空的或不完整的新索引

位置：`indexing.ts:210–213,266`，`indexing.ts:129–143`。

读取和解析异常只增加 `readErrorCount`，最后计入 `skipped`，没有进入 `failed/errors`。重建仅凭 `result.failed === 0` 发布 manifest。旧库需要重建时，只要目标文件不可读、PDF 解析抛错或在扫描后消失，就可能把缺内容的新库切成活动索引。

复现：旧库有 1 个 chunk，对一个不存在的目标文件调用 `rebuildWithSwitch(..., true)`；结果为 `indexed=0, skipped=1, failed=0, errors=[]`，仍生成 `active.json`，活动库变成 0 chunks。旧数据库文件尚在，但正常查询已失去原内容。

修复要求：解析错误必须计入失败，保留具体文件和原因；任何必要输入失败时禁止发布 staging，验证文件集合和向量覆盖后才能切换。

### R2 · P1：重建仍在成功之前修改活动库，而且操作在写锁之外

位置：`index.ts:299–307`，`indexing.ts:122–124`。

命令先对 dropped files 调用 `pruneIndexedFile`；非 force 模式还预先把文件标为 unembedded，然后才调用受锁保护的重建。后续 embedding 失败无法回滚这些修改。同合同 force rebuild 走原库的逐文件提交，发生后续文件写入错误时也没有整个重建的切换边界。

复现：活动库含保留文件和已删除文件，注入 embedding 异常后运行 `/rag rebuild --force`。旧库中的 deleted-file 记录已经消失，命令却输出 `✅ Rebuilt: 0 re-indexed ... 1 deleted`。

修复要求：把删除、文件状态变更和写入任务统一纳入提交边界及写锁；全量重建使用独立 generation，成功后发布。失败时旧活动索引及文件状态应可原样使用。

### R3 · P1：cloudAutoRefresh=false 没有阻止云端自动刷新

位置：`index.ts:117–128`。

自动注入 hook 只检查索引年龄，未检查 embedding provider 或 `cloudAutoRefresh`，随后直接 await `indexFiles`。云模式默认关闭自动刷新的配置实际未生效，会在普通对话时重算发生变化的文档、消耗额度，并阻塞对话。

复现：配置为 Voyage、`cloudAutoRefresh=false`，旧索引超过 24 小时，修改一个自制文档后调用 hook。HTTP stub 收到 `document` 和 `query` 两类请求；其中 document 请求本应被关闭。

修复要求：本地保留刷新行为，云模式必须显式开启才刷新；对整条自动注入链设置时间预算。

### R4 · P1：chunkBlocks 的 maxTokens 不是上限，超长段落从未被切开

位置：`chunking.ts:107–125`。

代码只在添加新段落前 flush 旧 buffer，随后仍把整个段落追加进去。单个超长段落、无换行中文或长代码行会原样成为超大 chunk。PDF 解析把每页文字压成一行，使该路径尤其常见。

复现：10,000 字符的单段输入，默认 max=240，输出 chunk 长度为 `[10000,120]`，按实现自身估算为 `[2500,30]` tokens。最后 120 字符还单独形成纯 overlap 尾块。

修复要求：按明确 tokenizer/模型约束切开超长段落和单行；overlap 不得产生只有重复内容的尾块。仅把字符数除以 4，无法保证 MiniLM 的输入预算，中英文和公式必须补测试。

### R5 · P1：查询绕过 fingerprint 兼容检查，并对无 fingerprint 的旧库默认使用本地模型

位置：`search.ts:75–77`，`providers/embedding/factory.ts:34–49`，`index.ts:114–115,133–138`。

索引写入会检查兼容性，但查询直接按 stored fingerprint 构建 provider；缺少 fingerprint 时返回默认 local provider。自动注入虽然取得 `needsRebuild`，却未处理该状态。因此状态显示需要重建，查询和注入仍继续运行；旧库向量来源未知时甚至可能用错模型。

复现：非空旧库无 fingerprint 时，兼容检查返回 false，`hybridSearch` 仍返回 1 条结果；将配置改为同维度其他模型，同样提示不兼容却继续返回结果。

修复要求：所有查询、写入和自动刷新入口使用同一兼容检查。旧库无 fingerprint 不得推定模型；手动操作提示重建，自动注入跳过并限频通知。

### R6 · P2：空库跳过实际向量表维度检查，rebuild --force 也无法恢复

位置：`index-manager.ts:75–79`，`indexing.ts:122–127`。

先以本地默认打开空库会创建 384 维表。之后改成 Voyage 1024 维，`chunks===0` 直接判兼容，强制重建仍走旧表。

复现：实际 rebuild 返回 `Expected 384 dimensions but received 1024`，`failed=1`，状态却仍显示 `needsRebuild=false`。

修复要求：即使没有 chunks，也必须核对已存在的 schema；维度变化时创建新索引或安全重建空 schema。

### R7 · P2：回到旧配置时会自动删除以前的索引

位置：`index-manager.ts:104–120`。

staging 目录只由合同哈希决定；如果目录存在且不是当前 active，就递归删除。A 配置切到 B 后，再尝试重建 A，会先删除之前保留的 A 索引。若本次重建失败，旧 A 回滚副本也丢失，与计划“不自动删除旧索引”的要求相悖。

复现：准备 A 的旧数据库文件，活动 manifest 指向 B，再执行 `prepareStagingDir(A)`；旧 A 文件消失。

修复要求：每次构建使用唯一 generation；已发布索引与未完成 staging 分开管理，不能把旧版本当 staging 清理。

### R8 · P2：取消信号未贯通查询和索引，自动注入缺少总时间预算

位置：`retrieval.ts:39,48–56`，`search.ts:77`，`indexing.ts:278`，`index.ts:640,666`。

`retrieve.signal` 没有传到 query embedding；none reranker 分支在检查取消前直接返回。索引函数未接收 signal，Pi 工具 execute 也未接收或转发取消参数。HTTP 有单请求超时，但没有覆盖刷新、多 batch、查询和重排的统一截止时间，退避等待也不响应取消。

复现：传入已经 abort 的 signal，查询 embedding 仍调用一次并返回 1 条结果。

修复要求：从 Pi 工具/hook 到 retrieval/indexing/provider 贯通 signal 和总 deadline；取消不得转换成成功、重试或降级。

### R9 · P2：索引失败和重排降级没有完整呈现给调用者

位置：`index.ts:228,355,654,672–678`，`retrieval.ts:28–30,72–75`。

索引结果的 `failed/errors` 被命令和工具输出丢弃，部分甚至带成功勾号。创建 reranker 失败时静默换成 NoneReranker；实际 rerank 请求失败虽然设置 `degraded`，三个入口也不展示该原因，tool 输出同时丢弃 rerank score。

复现：R2 的失败重建仍报成功；配置 Voyage reranker 但无 Key 时，查询返回结果，`degraded=null`、`rerank=null`，没有解释重排为何未运行。

修复要求：成功、部分失败、完全失败采用明确结果合同；保留并输出降级原因及各类分数，配置错误不能静默换模型。

### R10 · P2：查询 embedding 临时失败没有 BM25 降级

位置：`search.ts:74–79`，`retrieval.ts:37–45`。

即使已经拿到 FTS 候选，embedding 异常仍被直接抛出。代码注释提及“无向量时 BM25”，但这是正常返回空向量结果的路径，不能处理实际网络故障。

复现：FTS 能命中 1 条，HTTP stub 抛网络异常后，retrieve 整体失败，没有返回 BM25 结果或降级标记。自动注入 hook 也没有故障跳过处理。

修复要求：把临时服务故障、配置/合同不匹配和用户取消区分开；只对允许降级的故障使用已有 BM25 结果并显式标记。

### R11 · P2：PDF 页码和章节没有贯通到 Pi，文本行号也不是原始来源范围

位置：`context.ts:38–43`，`index.ts:672–678`，`chunking.ts:107–121`，`repository.ts:59–66`。

数据库和检索 chunk 已带 page/section，但 Context Builder 和 rag_query 只输出 file/lines，丢弃已知页码、章节和 chunk identity。分块中的 linePos 是按 trim 后段落长度重算，无法还原原文件空行、heading 被删除后的行号；甚至单行正文也会输出多行范围。files 表还缺少计划要求的 document ID 和可空 title，chunkIndex 也未贯通查询结果。

复现：结果 chunk 的 `pageStart=2, section=Method`，注入文本只有 `paper.pdf (lines 1-2)`，工具 JSON 同样没有 page/section/id。

修复要求：解析阶段保存真实来源范围；存储、查询、工具和 context 逐层保留字段。未知的行号不能输出为准确位置。

### R12 · P2：Markdown 的短章节会被直接丢弃

位置：`parsing.ts:39–45`。

按标题分段后过滤所有不超过 20 字符的章节。如果文档还有较长章节，就不会回退到全文，短参数、缩写或结论会永久消失。

复现：文档包含 `# Short\n42` 和较长的 `# Long` 部分，解析结果只有 Long，数字 42 完全丢失。

修复要求：保留非空短块并在分块时合并，不能用字符数过滤掉有效证据。

### R13 · P2：两个云 smoke 命令实际无法启动

位置：`providers/http.ts:2–5`，`package.json` 的 `smoke:voyage-*` scripts。

脚本使用 Node `--experimental-strip-types`，但 HttpError 构造器使用 TypeScript parameter properties，strip-only 模式不支持。

复现：Node v22.23.2 下两个命令均退出 1，报 `ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX`，尚未进入无 Key 的 UNVERIFIED 分支。这个问题与 API Key 或服务可用性无关。

修复要求：使用与代码兼容的运行器/编译产物，或改成可擦除的语法；分别验证无 Key 启动路径和有条件的真实 smoke。

### R14 · P2：E4 是结果模板，不是可运行的检索评估

位置：`scripts/eval-retrieval.ts:27–41`，`eval/questions.json`。

脚本没有建索引、查询、相关性匹配、去重、计时、计量或计算指标，只写出说明和固定 not-run。即使 Key 存在、语料准备好也没有执行入口。本地组同样永远 not-run；20 条问题没有中文问题或带页码的科研证据标注。

修复要求：实现可接收固定语料和标注的三组评估；无 Key 时允许云组未运行，但本地组和指标计算必须可执行。当前应标“未完成”，不能仅标“未验证”。

### R15 · P2：配置解析失败静默恢复默认值，不支持的模型/参数也未完整校验

位置：`config.ts:65–81,90–102,174–224`。

破损 JSON 或非对象配置静默回到默认 local；无法解析的数字/布尔环境值被忽略。模型只校验非空，local 模型与维度配对、Voyage 支持模型集合及 ragAlpha/threshold 等合同缺少校验。这会把操作失误表现成不同配置，状态也不能准确报告配置问题。

复现：config.json 内容为 `{BROKEN`，loadConfig 返回 local，validateConfig 返回空问题列表。

修复要求：区分“没有配置”和“配置损坏”；错误必须可见，同时保留只读状态诊断及明确的恢复入口。

### R16 · P2：正常本地查询不再复用已加载的 embedding pipeline

位置：`providers/embedding/factory.ts:37–38`，`providers/embedding/local.ts:19,27–31`。

每次查询有 fingerprint 的本地索引，都 new 一个 LocalEmbeddingProvider，而 pipeline 缓存只存在 provider 实例内部。原 facade 的默认单例缓存没有被这条生产路径使用。结构上导致每次查询重新调用 pipeline 初始化；没有进行真实 ONNX 耗时量测，因此不推断具体延迟数字。

诊断确认同一索引连续取出的 provider 不是同一对象。

修复要求：按已验证合同缓存 provider/pipeline，并在合同变化时正确失效；增加同一模型多次查询只初始化一次的测试。

## 二、验收缺口和报告应如何改写

| 步骤 | 审查状态 | 依据 |
| --- | --- | --- |
| A1 | 部分完成 | 入口与打包内容改善；本机 Pi loader 能加载。全套测试隔离及干净安装证据不足。 |
| A2 | 离线用例通过 | BM25 方向、单候选/同分处理和冻结排序用例存在；不代表真实检索质量提升。 |
| A3 | 部分完成 | 单文件 embedding/事务失败保留旧数据有实现和测试；命令级清理及全量切换仍有 R1/R2 问题。 |
| B1 | 部分完成 | Provider/facade 已实现，但生产查询的缓存退化见 R16；ONNX 未实测。 |
| B2 | 部分完成 | 合并/环境覆盖基本实现；非法配置和模型边界未满足。 |
| B3 | 部分完成 | HTTP/index 映射等 mock 用例存在；smoke 不可启动，模型/token 上限校验不完整。 |
| C1 | 不通过 | 无法保证失败恢复、空库维度和旧版本保留；R1/R2/R6/R7。 |
| C2 | 不通过 | 查询绕过兼容检查，失败输出及云刷新不满足合同。 |
| D1 | 基本结构完成 | 三个入口确实复用 retrieve，NoneReranker 保序截取有测试。 |
| D2 | 部分完成 | 正常映射、请求失败回退有实现；配置/取消/降级可观察性不足。 |
| D3 | 部分完成 | 有 context builder 和估算声明；来源、总预算、取消和降级缺失。 |
| E1 | 部分完成 | PDF 提取阶段有物理页字段；缺少实际生成的三页 PDF 第 2 页标记、跨页链路验收证据。 |
| E2 | 不通过 | 超长块不切分、纯 overlap 尾块、非真实 tokenizer、短内容丢失。 |
| E3 | 未完成 | 部分页字段入库；document 元数据、原始范围和输出贯通不足，真实论文抽查未做。 |
| E4 | 未完成 | 只有问题与结果模板，没有可运行指标计算。 |

其他应补验收的具体点：

- Processing fingerprint 只含 parser/chunker/maxLines=50；实际分块使用 target/max/overlap，而这些参数及计数合同没有进入 fingerprint（`fingerprint.ts:41–47`）。修改参数后存在错误复用旧 hash 的风险。
- Context 的计数器明确标为 estimated 是正确的，但“with slack”注释没有对应安全余量；中文估算不能据此宣称符合真实 Pi token 上限。所有正文被预算丢弃时，hook 仍发送仅有 header 的空资料消息。
- 候选上限可被 `limit` 绕过：`Math.max(limit, min(candidateTopK,200))`；tool 参数仅 Type.Number，没有整数、正数或上限约束。需验证不合理 limit 不触发巨量召回。
- Voyage reranker 未校验输入 token 总量，且请求未设 `truncation:false`，服务端默认会截断。官方接口明确规定 query、单 query+document、总 token 及文档数限制，参见 [Voyage Rerank API](https://docs.voyageai.com/reference/reranker-api)。
- Embedding 的 query/document、float、output_dimension 字段与当前 [Voyage Embedding API](https://docs.voyageai.com/reference/embeddings-api) 相符；选择 `voyage-4-lite`/`rerank-2.5-lite` 本身不是本次否决理由。账户可用性、真实配额、效果和时延仍未实测。
- 现有 fingerprint 测试主要测 helper；“切换”用例向路径写入字符串 `new`，没有真正执行 SQLite 重建、连接重开或失败重启流程，因此无法支持完整 C1 验收。
- 进度文档 B～E 多处 commit 仍是 “this commit”，最终报告未写明结束 SHA；需按实际状态补充支持模型/维度、恢复/回滚步骤和可复现命令，不应保留全部 done。

## 三、本次实际验证

环境：Node **v22.23.2**，项目 TypeScript **5.7.3**，当前 HEAD **d546804**。

| 验证 | 实际结果 | 解释 |
| --- | --- | --- |
| `npm run typecheck` | 通过 | 使用项目锁文件安装的 tsc；没有把 `build --noCheck` 算成类型验证。 |
| 离线全套测试，临时 PI_RAG_DIR/PI_RAG_LEGACY_DIR/npm cache | **157 passed / 4 failed / 4 skipped** | 4 个失败都是 getFreshDbConn dispose 用例，见下文。 |
| 单独隔离运行上述 4 项 | **4 passed / 105 按名称未选中** | 证明连接功能在隔离环境可用；不能据此把整套运行改报全绿。 |
| `npm pack --dry-run --ignore-scripts --json` | 通过，34 个文件 | repository、providers、retrieval、context、parsing 等运行文件在包中。 |
| 生成 tarball → 临时解包 → 本机 Pi `loadExtensions` | **1 extension / 0 errors** | rag_index、rag_query、rag_status、rag 命令、before_agent_start 均注册。复用现有 node_modules，不等同干净环境重新安装。 |
| `npm run smoke:voyage-embed`，显式移除 Key | **失败，exit 1** | TypeScript parameter property 无法 strip，见 R13。 |
| `npm run smoke:voyage-rerank`，显式移除 Key | **失败，exit 1** | 同上。 |
| reviewer 定向诊断 | 见附带 JSONL | 使用真实 SQLite/FTS/vec 和实现函数，替换模型推理与 HTTP，复现上述边界问题。 |

**全套测试的隔离问题：** `__tests__/index.test.ts:1027–1031` 清理时直接 delete 外部传入的 PI_RAG_DIR/PI_RAG_LEGACY_DIR，后续 dispose 用例又没有自己的临时存储。最终回落到用户全局目录；在本次 sandbox 中 SQLite WAL 写入被拒绝。最初 npm 默认 cache 也遇到写权限限制，改用临时 cache 后 pack 和对应测试通过。未修改全局权限或测试逻辑来获得通过。

这属于测试 harness 和交付复现的问题，不能据此断言生产连接功能坏了；也不能接受交付报告的 **161 passed** 作为本次整套复验结论。现有用例没有覆盖本审查复现的多数业务缺陷。

未进行：真实 ONNX 推理、真实 Voyage 请求、真实 Pi 会话工具/隐藏上下文交互、用户论文发送、真实论文人工抽查、检索效果/成本/延迟评估、全新依赖安装。Pi loader 加载通过与完整会话端到端通过是不同证据。

## 四、复现材料与修复顺序

附带 `grok-build-review-probe.mjs` 和 `grok-build-review-evidence.jsonl`。前者是观察脚本，使用临时合成库并阻断真实网络；结果用于诊断，不是模型评分或“验收通过”。后者还包含额外空库重建、静默重排降级及 provider 缓存观察。

```bash
node --experimental-transform-types docs/reviews/grok-build-review-probe.mjs
```

本次完整原始日志和额外探针保存在临时目录 `/private/tmp/pi-rag-review-FD0Ym9`；长期结论以本报告及随附证据为准。

建议按以下顺序返工，每批完成后重新验收：

1. **索引安全与云刷新**：R1/R2/R3/R5/R6/R7。增加真正经过命令、staging 发布和重启的失败恢复测试。
2. **查询可控性**：R8/R9/R10/R15/R16。取消、deadline、配置校验、可观察降级及模型缓存先闭环。
3. **科研内容与来源**：R4/R11/R12。真实 tokenizer、长段切分、原始来源范围、三页 PDF 和输出链路一起验证。
4. **交付证据**：R13/R14。修复 smoke、实现真正评估器、修复测试隔离，再重写报告状态。真实服务不具备条件的项继续列“未验证”，未实现的项列“未完成”。

本次新增的只是审查文档、诊断脚本和观察记录，生产源码保持审查时版本。
