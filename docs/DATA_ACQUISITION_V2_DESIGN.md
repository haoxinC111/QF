# QF 全量数据获取与长期归档设计

本文基于项目当前 V2 PIT 研究需求，以及用户提供的第三方“Tushare 15000 积分版本”
调用文档制定。目标是尽可能取得完整历史数据，同时保证来源可替换、任务可恢复、原始
数据可追溯、PIT 处理正确，并能方便地生成 QF 所需研究缓存。

## 1. 对提供文档的技术判断

文档描述的不是本人官方 Tushare Token 的标准连接方式，而是：

- 使用管理员发放的 Token；
- 修改 Tushare SDK 私有属性 `_DataApi__http_url`，把请求发往第三方兼容网关；
- 或向第三方网关根路径发送兼容 Tushare 的 HTTP POST；
- 标称 15000 积分、100 次/分钟、总量不限；
- 标称覆盖常规接口与部分特色接口，不包含官方历史分钟和 `rt_k` 实时权限；
- 另有一个修改 SDK 内部常量的“实时爬虫”方案。

因此，本项目不得把该来源标记为 `provider=tushare_official`，也不能把本人官方
`TUSHARE_TOKEN` 发送到第三方域名。第三方管理员发放的 Token 必须使用独立变量，并
把来源记录为 `tushare_compatible_proxy`。

在批量调用前必须取得/确认：

1. 服务提供方有权提供这些数据；
2. 允许在本地长期保存并用于个人研究；
3. 允许批量历史归档，且100次/分钟与总量口径真实有效；
4. 数据字段、历史范围、复权口径和修订规则与其声称的 Tushare 兼容口径一致。

任一项不能确认时，可以做少量技术探针，但不得将其作为正式生产数据源。

## 2. 获取方式选择

### 2.1 采用直接 HTTP 适配器

归档程序应使用文档给出的兼容 HTTP 协议：

```json
{
  "api_name": "daily",
  "token": "<从独立环境变量读取>",
  "params": {
    "trade_date": "20260110"
  },
  "fields": ""
}
```

选择 HTTP 而不是修改 SDK 私有属性，原因是：

- SDK 私有属性可能随 Tushare 版本变化；
- `ts.set_token()` 使用进程全局状态，容易把官方 Token 误发给第三方；
- HTTP 请求参数、响应和重试更容易完整记录；
- 可在不改变上层归档逻辑的情况下替换成官方 Tushare 或其他合法数据源。

请求必须设置 `Accept-Encoding: gzip`、连接/读取超时、TLS 证书验证，并禁止自动跳转
到不在允许列表中的域名。项目配置和日志不得出现 Token。

### 2.2 不采用文档中的实时爬虫方案

修改 `tushare.stock.cons` 内部常量的实时爬虫：

- 依赖未公开、可能随时变化的内部实现；
- 不能补齐历史数据；
- 来源、许可和稳定性难以证明；
- 与当前日频 PIT 研究无关。

因此它不进入本次归档范围。未来若正式立项实时/盘中交易，应使用有明确授权和 SLA
的独立实时数据源。

## 3. 数据源隔离

必须实现统一 Provider 协议：

```text
ArchiveProvider
  ├── OfficialTushareProvider
  ├── TushareCompatibleHttpProvider
  └── PublicFallbackProvider
```

每一行或每个 Parquet 分区至少带有以下血缘：

- `source_provider`
- `source_endpoint`
- `source_snapshot_id`
- `source_request_id`
- `fetched_at_utc`
- `raw_payload_sha256`
- `normalizer_version`

不同来源的数据不能静默拼接。相同主键发生冲突时，保留双方记录并输出差异报告，由
明确的来源优先级生成 Silver 层。

## 4. 四层存储结构

```text
data_lake/
  raw/
    tushare_compatible_proxy/<snapshot>/<endpoint>/request-*.json.zst
  bronze/
    <provider>/<endpoint>/year=YYYY/part-*.parquet
  silver/
    security_master/
    market_daily/
    financial_pit/
    corporate_events/
    research_features/
  gold/
    qf/base_cache/
    qf/pit_cache/
  catalog/
    archive.duckdb
    endpoint_inventory.yaml
    download_manifest.jsonl
    schema_registry/
    checksums.sha256
```

- Raw：保存每个请求的原始 JSON，Zstandard 压缩，只追加不覆盖。
- Bronze：把 `fields + items` 无损转为 Parquet，保留全部返回列。
- Silver：统一证券身份、日期、单位、修订版本和 PIT 有效时间。
- Gold：由固定代码版本从 Silver 生成当前 QF 可直接读取的缓存。

Raw 与 Bronze 不进入 Git，并至少保留一份独立备份。

## 5. 任务注册表

每个 endpoint 不能写成散落的下载脚本，而应登记为数据任务：

```yaml
- api_name: daily
  priority: P0
  dataset: market_daily
  start: earliest
  end: latest_complete_trade_date
  primary_key: [ts_code, trade_date]
  primary_split: trade_date
  fallback_split: ts_code
  expected_row_cap: probe
  pit: true
  all_fields: true
```

任务唯一 ID：

```text
sha256(provider + api_name + canonical_params + fields + snapshot_id)
```

状态机：

```text
pending -> running -> success | confirmed_empty
                   -> retryable_error -> pending
                   -> denied | invalid_params | suspect_truncated | quarantined
```

只有 `success` 和经过复核的 `confirmed_empty` 才算完成。

## 6. 接口切分策略

不能简单相信“一次请求跨度尽可能大”。如果上游存在行数上限，大跨度请求可能成功
返回但静默截断。应按下表切分：

| 数据类型 | 首选切分 | 备用切分 | 说明 |
|---|---|---|---|
| `trade_cal`、证券/指数基本信息 | 全量快照 | 交易所/状态 | 到期前再取一次最终快照 |
| `daily`、`adj_factor`、`daily_basic` | 单交易日全市场 | 单证券/年度 | 约5000多行/日，最适合横截面获取 |
| `stk_limit`、`suspend_d`、`moneyflow` | 单交易日 | 单证券/月或年 | 保证可交易状态与行情同日对齐 |
| 指数日线 | 指数/年度 | 指数/月度 | 价格与全收益指数分开保存 |
| `index_weight` | 指数/月度 | 指数/单日 | 保存真实历史成分和权重 |
| 财务 `*_vip` | 报告期/季度 | 公告月 | 保存所有报告类型和修订版本 |
| 普通财务接口 | 单证券/多年 | 单证券/年度 | 用 VIP 批量结果做抽样交叉核验 |
| 分红、回购、解禁、质押 | 公告年/月 | 单证券/年度 | 保留预案、实施、取消等全部状态 |
| 十大股东、股东户数、增减持 | 单证券/年度 | 单证券/季度 | 主键必须包含报告期/公告日/股东名 |
| `report_rc`、机构调研、券商推荐 | 发布月 | 发布周/机构 | 不得只保留最新预测 |
| `cyq_perf` | 单证券/年度 | 单证券/月 | 单日全市场可能贴近接口行数上限 |
| `cyq_chips` | 单证券/月 | 单证券/周或日 | 每股每日有多个价格档，先做容量试验 |
| ETF/基金日线与份额 | 单交易日 | 单基金/年度 | 持仓披露按报告期单独切分 |
| 宏观数据 | 单接口全量 | 年度 | 体量小，保存全部字段 |

每个接口的准确行数上限必须由权限探针实测，不能照搬官方 Tushare 文档作为第三方
网关的保证。

## 7. 防截断算法

对每个任务执行以下规则：

1. 小范围探针取得返回字段、错误格式和疑似行数上限。
2. 请求一个正常分区。
3. 若 `rows >= observed_cap`、返回日期未覆盖边界、最后主键异常，标记
   `suspect_truncated`。
4. 对日期范围二分；已缩到单日仍触顶时改用备用维度（证券代码）。
5. 子分区全部成功后才封存父任务完成状态。
6. 反向再请求随机分区，与已保存主键集合和 SHA256 对照。

不要通过“HTTP 200”判断完整，也不要把空数组自动视为真实无数据。

## 8. 限速、并发和重试

文档标称100次/分钟，但建议留出20%--25%余量：

- 全局令牌桶：默认75次/分钟；
- 启动阶段：单 worker；连续1000次无429/服务错误后最多提升到2 workers；
- 所有 worker 共享同一个令牌桶，不能每线程各限100次；
- 请求间隔加入小幅随机抖动；
- 连接超时10秒，读取超时120秒；
- 429、502、503、504：指数退避并带抖动，最多5次；
- 401/403、字段错误：不自动重试，进入失败队列；
- 响应不是预期 JSON、字段缺失或服务端 HTML：隔离原始响应并暂停该接口。

文档建议0.5秒间隔，但连续运行相当于最高120次/分钟，可能超过其同时标称的100次
权限。因此不能直接采用0.5秒固定间隔。

## 9. PIT 时间规则

- 日行情/估值：收盘后生成的记录最早用于下一交易日交易。
- 财务：使用 `ann_date/f_ann_date`；没有可靠时分秒时，默认公告后的下一交易日可用。
- 业绩预告、快报、分析师预测、调研：以首次公开时间为可见时间，修订另建版本。
- 分红、解禁、回购等事件：同时保存公告日、登记日、除权日、实施日，按研究问题选择
  正确可见时间。
- 概念/行业成员没有 `in_date/out_date` 时，只能保存为下载日快照，不得回填历史。
- Raw/Bronze 不做“只留最新一条”的去重；PIT 版本选择只发生在 Silver/Gold 层。

## 10. 权限与接口探针

批量下载前，生成 `permission_report.json`，每个 endpoint 记录：

- `success / denied / not_found / incompatible / transient_error`
- 返回字段、最小样本日期、最大样本日期；
- 单次最大行数的实测值；
- 是否支持 `fields`、日期、证券、报告期等参数；
- 与官方字段定义的差异；
- 响应耗时和连续请求稳定性。

尤其要确认文档声称的特色接口：卖方预测、券商推荐、`cyq_perf`、`cyq_chips`、
技术因子等。没有探针成功的接口不得提前标为“已拥有权限”。

## 11. 数据真实性与跨源校验

第三方兼容网关需要比官方源更严格的验收：

1. 随机选100只股票、20个交易日，与交易所、合法官方源或已冻结公开缓存核对。
2. 检查 OHLC 关系、成交量/金额单位、复权因子跳变和涨跌停价格。
3. 财务表抽样核对公告日、报告期、数值单位和修订记录。
4. 对相同请求隔日重取；若历史无说明地变化，生成 drift 报告，不覆盖旧快照。
5. 记录 SDK/HTTP 返回 schema 指纹；字段增删立即阻断该接口的 Silver 构建。
6. 对重要表输出证券覆盖、日期覆盖、退市股覆盖、重复主键和空值热图。

在这些检查通过前，数据只能标记为 `research_unverified`，不能支撑“严格复现”或实盘
结论。

## 12. 下载批次

### 批次 A：安全和容量试验

- 权限探针；
- 任选5个交易日、50只股票、4个财务季度；
- 估算吞吐、错误率、压缩比和总容量；
- 完成跨源抽样核验。

### 批次 B：P0 全市场底座

- 证券身份、交易日历、日线、复权、估值、涨跌停、停牌；
- 指数、历史成分、行业；
- 全部财务版本、业绩预告/快报和公司行动。

### 批次 C：P1 难替代数据

- 股东/治理事件、资金流、两融、龙虎榜；
- 分析师、调研、券商推荐；
- `cyq_perf`、ETF/基金、宏观和股指衍生品。

### 批次 D：大表

- `cyq_chips`；
- `stk_factor_pro`；
- 异源概念/资金流和其他 P2 数据。

### 批次 E：冻结与补增量

- 权限到期前7天执行全量缺口扫描；
- 到期前1天补最后增量；
- 生成最终 manifest、SHA256、覆盖报告和离线恢复演练。

## 13. 增量更新

- 交易日表：每个收盘后补最近5个交易日，允许上游更正。
- 财务/事件：每日补最近90天，并按新公告触发历史报告期重取。
- 证券/指数/行业成员：每周快照；每月生成差异。
- 每次增量形成新 Raw snapshot；Silver 通过版本规则合并，不覆盖历史血缘。

## 14. 安全规则

- 官方 Token：仅允许发送到官方 Tushare 域名。
- 兼容网关 Token：使用 `QF_ARCHIVE_API_TOKEN`，不得复用 `TUSHARE_TOKEN`。
- 网关地址：使用 `QF_ARCHIVE_API_URL`，不硬编码进仓库。
- Token 不进入 YAML、命令行参数、日志、异常、manifest 或 Git。
- 日志中的请求体必须把 `token` 替换成 `<redacted>`。
- 禁止通过未知 MCP 转发 Token；归档器只使用明确允许列表中的 HTTPS 主机。

## 15. 验收门槛

全部满足后才可称为“完成归档”：

- P0 endpoint 100% 成功或有书面、可解释的不可用记录；
- 所有分区都有原始响应 SHA256；
- 无未解决的 `suspect_truncated`；
- 关键表主键重复率为0，或每个重复均能由修订版本解释；
- 退市股、历史 ST 和北交所覆盖单独通过；
- PIT 财务公告日前不可见；
- 与独立来源的抽样差异在预先定义的容忍范围内；
- 从空目录仅用 manifest 能恢复 Bronze、Silver 和 QF Gold；
- `download_manifest.jsonl`、`coverage_report`、`checksums.sha256` 完整通过。

## 16. 实施顺序

1. 实现 Provider 接口和直接 HTTP 适配器。
2. 实现 endpoint registry、任务数据库和全局限速器。
3. 实现 Raw JSON 封存、Parquet 转换和 schema 指纹。
4. 实现断点续传、递归分片及截断检测。
5. 实现权限、覆盖、漂移和跨源核验报告。
6. 只跑批次 A；人工确认授权、质量和容量。
7. 再依次执行 B、C、D、E。

本设计能最大化保留可用历史数据，但不能在未验证第三方服务授权、接口范围和返回行数
上限的情况下承诺“绝对完整”。完整性必须由任务清单、原始响应、覆盖报告和跨源核验
共同证明，而不能依靠销售文档中的“15000积分”标签。
