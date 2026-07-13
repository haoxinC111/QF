# A股动态股票池多因子量化回测

这是一个研究/回测级的 Python 项目，目标不是展示一段“看起来能跑”的选股代码，而是把容易让 A 股回测失真的环节显式建模：历史成分股、复权、信号与成交错位、涨跌停、停牌、T+1、100 股整手、费用、滑点、成交容量和风险敞口。

> 重要：本项目不构成投资建议，不承诺收益，也没有连接券商下单。先用离线演示确认环境，再用自己的数据权限回测；任何真实资金使用前，都应做样本外检验、压力测试、人工复核和小资金仿真。

## 一、策略概览

默认策略是“历史沪深 300 成分股内的月度截面多因子策略”。它只做多，不加杠杆，不做日内交易。

每个自然月最后一个交易日收盘后：

1. 取得当时已生效的指数成分，而不是今天的成分股。
2. 剔除 ST、历史不足、成交额不足、价格过低以及长期趋势为负的股票。
3. 计算五个价格/成交因子，做截面缩尾和标准化。
4. 按综合得分选前 20 只，用逆波动率分配权重，单股不超过 8%。
5. 用沪深 300 的 200 日均线决定总股票仓位：风险开启 95%，风险关闭 30%。
6. 下一交易日开盘执行；遇到停牌、涨跌停或容量限制，最多重试 3 个交易日。

### 因子定义

所有价格因子都使用 `原始价格 × 复权因子` 的总回报序列。令该序列收盘价为 \(P_t\)：

| 因子 | 定义 | 默认权重 | 意图 |
| --- | --- | ---: | --- |
| 12-1 动量 | \(P_{t-21}/P_{t-252}-1\) | 35% | 捕捉中长期趋势，跳过最近约一个月 |
| 6-1 动量 | \(P_{t-21}/P_{t-126}-1\) | 20% | 补充中期趋势 |
| 长期趋势 | \(P_t/MA_{200}(P)-1\) | 15% | 偏向处于长期上升趋势的股票 |
| 低波动 | 60 日年化波动率的相反数 | 20% | 避免组合被高波动个股主导 |
| 流动性 | \(\log(1+20日平均成交额)\) | 10% | 偏向更容易成交的标的 |

对每个因子在当日股票池内进行 5%/95% 分位缩尾，再转为 z-score。综合分数为：

\[
Score_i=0.35Z(MOM12\_1)+0.20Z(MOM6\_1)+0.15Z(Trend)+0.20Z(-Vol)+0.10Z(Liquidity)
\]

选股与权重是两个不同步骤：综合分数决定“买谁”，逆波动率决定“买多少”。权重经过迭代截帽，保证组合目标敞口和单股上限同时成立。若过滤后股票数量不足，程序会自动降低总敞口，而不会突破单股上限硬凑仓位。

### 风险开关

信号日沪深 300 收盘价不低于其 200 日均线时，目标股票仓位为 95%；否则为 30%。这是一个简单、可解释的系统性风险控制，不是对市场方向的保证。剩余部分保持现金。

## 二、回测中建模的 A 股约束

| 约束 | 实现方式 |
| --- | --- |
| 避免未来函数 | 月末收盘生成信号，下一交易日开盘才成交 |
| 成分股幸存者偏差 | 使用 `index_weight` 的历史月度成分快照 |
| ST 历史状态 | 使用历史曾用名区间，而不是当前名称回填过去 |
| 除权除息 | 原始价格负责成交，价格乘复权因子负责总回报估值 |
| T+1 | 当日买入的仓位当日不可卖出 |
| 买入整手 | 买单向下取整为 100 股整数倍 |
| 零股卖出 | 清仓时允许一次性卖出剩余零股 |
| 停牌 | 当日没有行情或成交量为 0 时拒单 |
| 涨跌停 | 开盘触及涨停不买，开盘触及跌停不卖；优先使用数据源的每日涨跌停价 |
| 滑点 | 默认买入加 5 bps、卖出减 5 bps，并限制在涨跌停价格内 |
| 成交容量 | 单日订单不超过已知 20 日平均成交额的 5% |
| 费用 | 默认佣金万 2.5、最低 5 元；卖出印花税万 5；过户费双向十万分之一 |
| 失败重试 | 固定月末信号和目标金额，最多尝试 3 个交易日，之后取消 |

手续费均可在配置中修改。券商实际佣金、最低收费和监管费率可能不同，运行真实资金前必须按自己的账户确认。

## 三、快速开始

需要 Python 3.10 或更高版本。

```bash
cd a_share_quant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. 先运行离线演示

离线演示使用程序生成的确定性假行情，只用于验证管线和报告，完全不代表策略收益：

```bash
python run.py demo --output results/demo
```

完成后打开：

```text
results/demo/report.html
```

### 2. 配置真实 A 股数据

复制配置：

```bash
cp config.example.yaml config.yaml
```

注册 Tushare Pro，确认自己的积分/接口权限后，把 Token 放入环境变量。不要把 Token 写进 YAML 或提交到 Git：

```bash
export TUSHARE_TOKEN="你的 token"
```

Windows PowerShell：

```powershell
$env:TUSHARE_TOKEN="你的 token"
```

下载、缓存并回测：

```bash
python run.py all --config config.yaml
```

首次下载需要逐月获取历史成分，并获取历史成分股的日线、复权因子、涨跌停价和曾用名，耗时取决于回测区间、接口频率和历史成分数量。缓存完成后，重复研究只需：

```bash
python run.py backtest --config config.yaml
```

也可以分开执行：

```bash
python run.py download --config config.yaml
python run.py validate-data --config config.yaml
python run.py backtest --config config.yaml
```

若修改回测日期或指数，下载器会检查缓存覆盖范围；需要强制重抓时，把 `data.refresh` 改为 `true`，完成后建议改回 `false`。

## 四、输出文件

默认写入 `results/latest/`：

| 文件 | 内容 |
| --- | --- |
| `report.html` | 自包含图表、指标、月度收益、配置和风险说明 |
| `performance.png` | 净值与回撤图 |
| `metrics.json` | 年化收益、波动、夏普、回撤、换手、费用、基准等 |
| `equity_curve.csv` | 每日净值、现金、持仓市值、仓位和基准净值 |
| `selections.csv` | 每次调仓的因子值、排名和目标权重 |
| `orders.csv` | 所有成交、拒单和撤单及原因 |
| `trades.csv` | 实际成交明细和费用拆分 |
| `final_positions.json` | 期末持仓市值 |
| `resolved_config.yaml` | 本次运行的完整配置快照 |
| `runtime.json` | Python、NumPy、pandas 和系统版本 |

## 五、核心参数怎么调

`config.example.yaml` 已包含全部参数。建议一次只改一类，并保留每次运行的 `resolved_config.yaml`。

### 股票池与数据

- `universe_index`：默认 `399300.SZ`。若换中证 500 等指数，先确认 Tushare 的指数代码和历史成分权限。
- `warmup_calendar_days`：默认 500 个自然日，为 252 个交易日动量和 200 日均线留缓冲。
- `calls_per_minute`：不能高于你的接口权限。遇到限流时调低。

### 选股

- `top_n`：默认 20。越小个股风险越集中，越大则信号稀释、交易笔数增加。
- `min_avg_amount_million`：20 日平均成交额下限，单位百万元。
- `stock_trend_filter`：关闭后允许长期趋势为负的股票进入候选。
- 五个 `*_weight`：会自动按权重总和归一化，不要求手工加到 1。

### 风险与交易

- `risk_on_exposure` / `risk_off_exposure`：风险开启和关闭时总股票仓位。
- `max_stock_weight`：单股绝对权重上限。必须满足 `top_n × max_stock_weight >= risk_on_exposure`。
- `slippage_bps`：单边滑点。建议同时压力测试 5、10、20 bps。
- `max_participation_of_20d_amount`：订单金额占过去 20 日平均成交额的上限。
- `rebalance_retry_days`：因涨跌停、停牌或资金不足导致未成交时的最大尝试天数。

## 六、如何做更可信的研究

不要只看一次全区间回测的年化收益。推荐按下面顺序评估：

1. 把较早年份作为参数研究期，较晚年份作为完全不参与调参的样本外期。
2. 做滚动或扩展窗口验证，例如用前 5 年确定参数、下一年验证，再向前滚动。
3. 对佣金、滑点和成交容量做至少 2～4 倍压力测试。
4. 分牛市、熊市、震荡市检查收益来源，不只看总体夏普。
5. 检查 `selections.csv` 的行业/风格集中度；当前版本没有行业中性化。
6. 检查 `orders.csv` 中涨跌停、停牌和容量拒单，避免把“理论目标仓位”当作“实际可成交仓位”。
7. 与等权沪深 300、沪深 300 指数和简单 200 日均线策略做基线对照。
8. 锁定参数后再增加最新数据；不要看到新结果后反复回头改参数。

## 七、已知局限

- 日线数据只能判断开盘是否触及涨跌停，不能模拟集合竞价排队、盘口深度、部分成交和盘中价格路径。
- Tushare 数据可能修订，接口权限和字段也可能变化；缓存和运行配置必须随研究结果一同归档。
- 指数历史成分降低了幸存者偏差，但不能消除指数编制本身的选择偏差。
- 分红送转通过复权因子转化为合成总回报单位，适合组合收益研究，但不是逐笔公司行动清算引擎。
- 长期停牌、退市、换股吸收合并等极端事件需要额外的公司行动数据逐笔处理。
- 默认因子只使用价格与流动性，没有基本面质量、估值、行业中性和风险模型。
- 基准使用价格指数，不是含股息全收益指数；长期相对收益会因此有口径差异。
- 示例不是券商实盘系统，没有账户同步、订单回报、撤改单、风控审批、交易时段保护和灾备。

若要走向仿真/实盘，至少还应加入：数据快照版本、券商适配层、目标仓位与账户持仓对账、价格笼子、单笔/单日损失限制、重复下单幂等键、人工审批、成交回报重放和紧急停机开关。

## 八、测试

无需 Tushare Token：

```bash
python -m unittest discover -s tests -v
```

测试覆盖：权重截帽、未来数据隔离、信号次日成交、100 股整手、佣金/印花税、目标仓位，以及涨停拒单后的次日重试。

## 九、项目结构

```text
a_share_quant/
├── config.example.yaml
├── requirements.txt
├── run.py
├── src/ashare_quant/
│   ├── backtest.py     # 组合、订单、费用、T+1、涨跌停与重试
│   ├── cli.py          # 命令行入口
│   ├── config.py       # 配置与参数校验
│   ├── data.py         # Tushare 下载、缓存、数据校验、离线演示
│   ├── factors.py      # 因子、动态成分、选股、风险开关与权重
│   └── report.py       # 指标、CSV、PNG 和 HTML 报告
└── tests/test_quant.py
```

## 十、规则与数据接口参考

- [Tushare A股日线行情 `daily`](https://tushare.pro/wctapi/documents/27.md)
- [Tushare 指数成分和权重 `index_weight`](https://tushare.pro/wctapi/documents/96.md)
- [Tushare 每日涨跌停价格 `stk_limit`](https://tushare.pro/wctapi/documents/183.md)
- [Tushare 股票曾用名 `namechange`](https://tushare.pro/wctapi/documents/100.md)
- [Tushare 交易日历 `trade_cal`](https://tushare.pro/wctapi/documents/26.md)
- [Tushare 指数日线 `index_daily`](https://tushare.pro/wctapi/documents/95.md)
- [上海证券交易所交易规则（2026年修订）](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)
- [深圳证券交易所交易规则（2026年修订）](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)
- [财政部、税务总局关于减半征收证券交易印花税的公告](https://xj.mof.gov.cn/zcfagui/202311/t20231108_3915476.htm)

交易所规则、税费和数据权限会调整。代码中的默认参数是截至项目编写时的研究假设，实际使用时应重新核对。
