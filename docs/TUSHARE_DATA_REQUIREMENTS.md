# Tushare 数据获取与验收清单（V2 Alpha2/Alpha3）

本文档是本项目真实数据复现的唯一数据交接清单。目标是让本地 Agent 在取得
Tushare Token 后，按照固定接口、字段、区间和顺序构建可校验的数据缓存，避免
未来函数、幸存者偏差、复权错误和数据源混用。

> 本文只描述“当前 V2 Alpha2/Alpha3 可运行的最小闭环”，不是账号到期前的完整
> 数据归档范围。若希望一次性保留未来研究可能使用的数据，请同时执行
> `TUSHARE_DATA_ARCHIVE_PLAN.md`。本文第 5 节的“不需要”仅表示当前策略不读取，
> 不代表没有长期归档价值。

## 1. 固定研究口径

- 研究市场：A 股。
- 股票池：沪深 300 历史成分，不使用当前成分回填历史。
- 成分指数：`399300.SZ`。
- 风险开关价格指数：`000300.SH`。
- 业绩比较全收益指数：`H00300.CSI`。
- 正式回测区间：`2012-01-01` 至 `2025-12-31`。
- 行情预热：回测日前至少 500 个自然日；建议从 `2010-01-01` 开始保留。
- PIT 财务历史：回测日前 6 年，即至少从 `2006-01-01` 开始。
- 2026 年及以后数据：可以下载和封存，但不得用于重新选因子、调权或调参；只作
  真正的前向观察期。
- 频率：日频足够。当前版本不需要分钟、Tick 或 Level-2 数据。

## 2. Token 与权限要求

- 推荐：本人官方 Tushare 账户的 5000 积分或以上常规数据权限。
- 15000 积分完全满足，但当前版本不使用其额外的特色数据权限。
- 不需要另购历史分钟、实时分钟、实时日线、新闻、公告 PDF 等独立权限。
- Token 只通过环境变量传入：

  ```bash
  export TUSHARE_TOKEN="<YOUR_TOKEN>"
  ```

- 禁止把 Token 写入 `config.yaml`、源码、日志、结果文件或 Git 历史。
- 禁止提交原始账户密码、Cookie 或第三方共享账户信息。

## 3. 必须获取的基础市场数据

### 3.1 交易日历与证券身份

- [ ] `trade_cal`
  - 参数：`exchange=SSE`、`is_open=1`。
  - 字段：`cal_date,is_open`。
  - 区间：至少 `2006-01-01` 至 `2026-01-14`，为 PIT 可用日映射留出余量。
- [ ] `stock_basic`
  - 分别请求 `list_status=L,D,P`，必须包含退市和暂停上市证券。
  - 字段：
    `ts_code,name,industry,market,list_status,list_date,delist_date`。
- [ ] `namechange`
  - 每只历史成分股请求。
  - 字段：`ts_code,name,start_date,end_date`。
  - 用途：历史 ST/退市名称识别，禁止使用当前名称回填历史。

### 3.2 历史股票池与行业

- [ ] `index_weight`
  - `index_code=399300.SZ`。
  - 从 `2011-11-01` 至 `2025-12-31` 按月请求。
  - 必须保留：`trade_date,con_code,weight`。
  - 不得只保存当前 300 只股票。
- [ ] `index_classify`
  - 参数：`level=L1,src=SW2021`。
  - 必须保留 `index_code` 和行业名称/层级字段。
- [ ] `index_member_all`
  - 对所有申万一级行业分别请求 `is_new=Y` 和 `is_new=N`。
  - 必须保留：
    `l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new`。
  - 所有沪深 300 历史成分必须能匹配行业区间；缺失项必须记录并阻断严格回测。

### 3.3 每只历史成分股的日频行情

以下接口按 `ts_code` 请求。保存范围至少覆盖 `2010-01-01` 至 `2025-12-31`；
程序允许根据预热期精确裁剪，但不得晚于策略所需首日。

- [ ] `daily`
  - 字段：
    `ts_code,trade_date,open,high,low,close,pre_close,vol,amount`。
- [ ] `adj_factor`
  - 字段：`ts_code,trade_date,adj_factor`。
- [ ] `stk_limit`
  - 字段：`ts_code,trade_date,up_limit,down_limit`。
- [ ] `daily_basic`（基础市场缓存部分）
  - 字段：`ts_code,trade_date,total_mv,circ_mv`。
- [ ] `dividend`
  - 字段：
    `ts_code,div_proc,record_date,ex_date,pay_date,div_listdate,cash_div,stk_div,imp_ann_date`。
  - 只把 `div_proc=实施` 的记录送入成交/公司行动账本，但原始响应应先封存。

### 3.4 基准与风险指数

- [ ] `index_daily`：`000300.SH`
  - 用途：200 日均线风险开关。
- [ ] `index_daily`：`H00300.CSI`
  - 用途：全收益比较基准。
- [ ] 两者字段：
  `ts_code,trade_date,open,high,low,close,pre_close,pct_chg`。
- [ ] 区间：至少 `2010-01-01` 至 `2025-12-31`。
- [ ] 如果 `H00300.CSI` 无返回，不得静默用价格指数替代全收益指数；必须报告并
  暂停正式超额收益结论。

## 4. 必须获取的 PIT 财务与估值数据

PIT 数据按基础市场缓存中出现过的全部历史成分股请求。请求区间至少为
`2006-01-01` 至 `2025-12-31`。财务数据以公告日期为可见性基础，默认在公告后的
下一交易日可用；日终估值只允许用于收盘后产生、下一交易日执行的信号。

### 4.1 利润表 `income`

- [ ] 参数：每只股票，`report_type=1`。
- [ ] 身份与时点字段：
  `ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag`。
- [ ] 数值字段：
  `basic_eps,total_revenue,revenue,operate_profit,total_profit,n_income,n_income_attr_p`。

### 4.2 资产负债表 `balancesheet`

- [ ] 参数：每只股票，`report_type=1`。
- [ ] 身份与时点字段：
  `ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag`。
- [ ] 数值字段：
  `money_cap,accounts_receiv,inventories,total_cur_assets,fix_assets,total_assets,total_cur_liab,total_liab,total_hldr_eqy_exc_min_int`。

### 4.3 现金流量表 `cashflow`

- [ ] 参数：每只股票，`report_type=1`。
- [ ] 身份与时点字段：
  `ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag`。
- [ ] 数值字段：
  `n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,c_pay_acq_const_fiolta`。

### 4.4 财务指标 `fina_indicator`

- [ ] 身份与时点字段：`ts_code,ann_date,end_date,update_flag`。
- [ ] 数值字段：
  `roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets,current_ratio,quick_ratio,assets_turn,ocf_to_or,ocf_to_opincome,profit_dedt,or_yoy,netprofit_yoy`。
- [ ] 单只股票一次返回超过接口上限时必须分页或按日期分段，禁止截断后继续研究。

### 4.5 日频估值 `daily_basic`

- [ ] 字段：
  `ts_code,trade_date,turnover_rate,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv`。
- [ ] 区间：`2006-01-01` 至 `2025-12-31`；至少必须完整覆盖正式回测区间。
- [ ] 非正 PE/PB/PS 不能伪装成“极度便宜”，因子层应按现有规则转为缺失。

## 5. 当前不需要下载的数据

- [ ] 不需要 A 股历史分钟、实时分钟、Tick、Level-2。
- [ ] 不需要实时日线或集合竞价。
- [ ] 不需要新闻、公告 PDF、互动问答。
- [ ] 不需要筹码分布、筹码胜率、券商金股、分析师盈利预测。
- [ ] 不需要北向资金、融资融券、龙虎榜、基金持仓。
- [ ] 不需要宏观数据作为当前 Alpha2/Alpha3 的输入。

如后续增加这些因子，必须作为新研究阶段单独立项，并重新定义冻结开发期和保留期；
不得在看过 2022--2025 保留期结果后随意加入因子。

## 6. 本地配置

从 `config.example.yaml` 复制配置，不要手写旧版 schema：

```bash
cp config.example.yaml config.yaml
```

至少确认以下配置值：

```yaml
data:
  provider: tushare
  cache_dir: data/cache
  token_env: TUSHARE_TOKEN
  universe_index: 399300.SZ
  regime_index: 000300.SH
  benchmark_index: H00300.CSI
  benchmark_is_total_return: true
  calls_per_minute: 400
  retries: 5
  warmup_calendar_days: 500
  strict_validation: true
  industry_standard: SW2021
  industry_level: L1

point_in_time:
  enabled: true
  provider: tushare
  cache_dir: data/pit_cache
  history_years: 6
  fundamental_lag_trading_days: 1
  valuation_lag_trading_days: 0
  maximum_fundamental_age_days: 550
  maximum_valuation_age_days: 10
  minimum_symbol_coverage: 0.90

backtest:
  start_date: 2012-01-01
  end_date: 2025-12-31
```

`calls_per_minute: 400` 只适用于确认账户具备 500 次/分钟权限后；否则必须按账户
实际权限下调。不要把请求频率设为权限上限，以便给重试和人工探针留余量。

## 7. 强制下载顺序

先同步依赖，避免 `uv run` 隐式改写锁文件：

```bash
env -u UV_DEFAULT_INDEX uv sync --locked --extra public \
  --default-index https://pypi.org/simple
```

然后依次执行：

```bash
uv run --no-sync python run.py download --config config.yaml
uv run --no-sync python run.py validate-data --config config.yaml

uv run --no-sync python run.py pit-download --config config.yaml
uv run --no-sync python run.py pit-verify --config config.yaml

uv run --no-sync python run.py pit-research \
  --config config.yaml \
  --output results/pit_factor_research_v2_alpha2
uv run --no-sync python run.py result-verify \
  --output results/pit_factor_research_v2_alpha2 --strict

uv run --no-sync python run.py pit-shadow \
  --config config.yaml \
  --alpha2-research results/pit_factor_research_v2_alpha2 \
  --output results/pit_shadow_v2_alpha3
uv run --no-sync python run.py result-verify \
  --output results/pit_shadow_v2_alpha3 --strict
```

`download -> validate-data -> pit-download -> pit-verify` 的顺序不能调换。PIT manifest
必须绑定已经封存的基础市场 manifest 和数据指纹。

## 8. 数据验收标准

### 8.1 基础市场缓存

- [ ] `validate-data` 正常退出，且无严格校验警告。
- [ ] `manifest.json` 的 `provider=tushare`。
- [ ] `requested_start/requested_end` 覆盖配置要求。
- [ ] manifest 文件清单 SHA256 全部通过。
- [ ] 历史成分股票数应约为 700--850；明显只有约 300 只通常意味着误用了当前成分。
- [ ] 每个成分证券都有独立行情分区；确实无行情的证券有明确失败记录。
- [ ] OHLC、前收盘、复权因子无非法缺口。
- [ ] `total_mv`、`circ_mv` 在有行情交易日不得静默缺失。
- [ ] 成交量为 0/停牌日不允许被当成正常可成交日。
- [ ] 涨跌停、ST、退市日期及公司行动校验通过。
- [ ] `H00300.CSI` 明确标记为全收益基准。

### 8.2 PIT 缓存

- [ ] `pit-verify` 正常退出。
- [ ] PIT manifest 的基础 manifest SHA256 与当前 `data/cache/manifest.json` 一致。
- [ ] 每只预期证券都有 fundamentals 与 valuations 分区，包括合法空分区。
- [ ] `fundamental_symbol_coverage >= 90%`。
- [ ] `valuation_symbol_coverage >= 90%`。
- [ ] 所有 `available_date >= announcement_date`。
- [ ] 财务默认至少延迟 1 个交易日可见。
- [ ] 估值日期不晚于其可用日期，且不会用于同一收盘价成交。
- [ ] 同一证券、报告期、指标的修订顺序确定且源行 SHA256 唯一可追踪。
- [ ] 修改未来公告或未来估值不会改变过去信号（未来数据扰动测试通过）。

### 8.3 研究结果

- [ ] Alpha2 输出因子覆盖率、IC、分位组收益、行业/规模暴露、消融、成本敏感性和
  滚动评价。
- [ ] Alpha3 输出四臂账本、覆盖匹配、固定 25% 混合归因和治理结论。
- [ ] 开发期、验证期、保留期分别报告，不只报告 2012--2025 全期值。
- [ ] 2022--2025 保留期只允许在冻结参数后查看。
- [ ] 所有结果目录通过 `result-verify --strict`。
- [ ] 结果报告同时记录 Git commit、配置 SHA256、基础数据指纹和 PIT 数据指纹。

## 9. 下载完成后需要交回的非敏感文件

不要提交 Token 或整套原始缓存。先提交/提供以下小文件供复核：

- [ ] `data/cache/manifest.json`
- [ ] `data/pit_cache/manifest.json`
- [ ] `results/pit_factor_research_v2_alpha2/reproducibility.json`
- [ ] `results/pit_factor_research_v2_alpha2/pit_factor_governance.json`
- [ ] `results/pit_shadow_v2_alpha3/reproducibility.json`
- [ ] `results/pit_shadow_v2_alpha3/pit_shadow_governance.json`
- [ ] `results/pit_shadow_v2_alpha3/pit_shadow_comparison.csv`
- [ ] `results/pit_shadow_v2_alpha3/pit_shadow_coverage.csv`
- [ ] 两个结果目录的 artifact manifest / SHA256 清单。
- [ ] 下载失败、空响应、重试和覆盖不足的完整摘要；日志须先检查不含 Token。

只有上述清单全部通过，才能把结果称为“绑定 Tushare 数据指纹的真实 PIT 研究”。
即使通过，也不代表未来收益保证，更不能把目标年化收益写成承诺。
