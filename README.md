# A股动态股票池多因子量化回测

这是一个研究/回测级的 Python 项目。v1.5 在 v1.4 的可信度与下载稳定性基础上新增“质量动量”候选：用 FIP 连续信息动量区分平滑趋势与少数跳涨，并加入下行波动和阶段回撤质量。严格通道仍显式建模历史行业/市值、分红送股、信号与成交错位、涨跌停、停牌、退市、T+1、100 股整手、历史费用、滑点和成交容量。

> 重要：本项目不构成投资建议，不承诺收益，也没有连接券商下单。先用离线演示确认环境，再用自己的数据权限回测；任何真实资金使用前，都应做样本外检验、压力测试、人工复核和小资金仿真。

## 一、策略概览

默认策略是“历史沪深 300 成分股内的月度截面多因子策略”。它只做多，不加杠杆，不做日内交易。

每个自然月最后一个交易日收盘后：

1. 取得当时已生效的指数成分，而不是今天的成分股。
2. 剔除 ST、历史不足、成交额不足、价格过低以及长期趋势为负的股票。
3. 计算六个正权重价格因子，做截面缩尾和标准化；综合分数对当日有效申万行业和对数总市值做截面残差化。
4. 新股票按前 20 名进入；已实际持有的股票只要仍在前 35 名即可保留，减少排名边界附近的无效往返交易。
5. 用逆波动率分配权重，单股不超过 8%，单个行业目标权重不超过 25%。约束不可同时满足时保守降低股票总敞口。
6. 用沪深 300 **价格指数**的 200 日均线决定总股票仓位：风险开启 95%，风险关闭 30%；业绩比较单独使用沪深 300 **全收益指数**。
7. 信号日收盘冻结订单股数，下一交易日只按实际开盘价决定成交金额；遇到停牌、涨跌停或容量限制，最多重试 3 个交易日。

### 因子定义

所有价格因子都使用 `原始价格 × 复权因子` 的总回报序列。令该序列收盘价为 \(P_t\)：

| 因子 | 定义 | 默认权重 | 意图 |
| --- | --- | ---: | --- |
| 12-1 动量 | \(P_{t-21}/P_{t-252}-1\) | 25% | 捕捉中长期趋势，跳过最近约一个月 |
| 6-1 动量 | \(P_{t-21}/P_{t-126}-1\) | 15% | 补充中期趋势 |
| FIP 连续信息动量 | \(MOM12\_1\times(1-ID)\) | 25% | 在同等累计涨幅下偏向连续形成而非少数跳涨 |
| 长期趋势 | \(P_t/MA_{200}(P)-1\) | 10% | 偏向处于长期上升趋势的股票 |
| 低下行波动 | 60 日年化下行波动率的相反数 | 15% | 主要惩罚负收益波动，不惩罚上涨波动 |
| 回撤质量 | \(P_t/Max_{126}(P)-1\) | 10% | 避免仍处于较深阶段回撤的股票 |
| 低总波动 | 60 日年化波动率的相反数 | 0% | 保留为可选因子；总波动仍用于定仓 |
| 流动性 | \(\log(1+20日平均成交额)\) | 0% | 保留为可选因子；成交额仍是硬性准入门槛 |

FIP 的信息离散度为 \(ID=sign(MOM12\_1)\times(\%neg-\%pos)\)，正负日比例只取 12–1 月形成窗口内截至信号日可见的 231 个日收益。定义来自 [Da、Gurun 与 Warachka（2014）](https://doi.org/10.1093/rfs/hhu003)。对每个因子在当日股票池内进行 5%/95% 分位缩尾，再转为 z-score。默认综合分数为：

\[
\begin{aligned}
Score_i={}&0.25Z(MOM12\_1)+0.15Z(MOM6\_1)+0.25Z(FIPMom)\\
&+0.10Z(Trend)+0.15Z(-DownVol)+0.10Z(DrawdownQuality)
\end{aligned}
\]

公式、冻结权重和晋级标准详见 [`V1.5_ALPHA.md`](V1.5_ALPHA.md)。2013–2025 已被旧版本研究查看，v1.5 历史回放不属于未触碰样本外；项目也不会为了达到指定年化收益而自动调参。

原始综合分数形成后，程序以当日有效的行业哑变量和 `log(total_mv)` 做横截面最小二乘残差化；`size_neutralization_strength=1` 表示完整剔除线性市值暴露。这里的“中性”只针对选股分数，不意味着实际组合对所有风险因子严格零暴露，实际结果应查看 `style_exposure.csv`。

选股与权重是两个不同步骤：中性化后的综合分数决定“买谁”，逆波动率决定“买多少”。权重使用水位分配算法同时满足总敞口、单股上限和行业上限。若股票或行业数量不足，程序会自动降低总敞口，而不会突破硬约束凑仓位。

### 排名缓冲

缓冲使用信号时点的**实际持仓**，不是上一次理论目标。默认 `top_n=20`、`exit_rank=35`：未持有股票按得分从高到低补足 20 只；已持有股票若仍在前 35 名且继续满足 ST、历史、价格、流动性和趋势过滤，可优先保留。`selections.csv` 中 `selection_reason=HOLD_BUFFER` 表示由缓冲保留。缓冲降低换手，但也可能延迟响应因子反转，因此必须同时比较关闭缓冲的结果。

### 风险开关

信号日沪深 300 价格指数收盘价不低于其 200 日均线时，目标股票仓位为 95%；否则为 30%。这是一个简单、可解释的系统性风险控制，不是对市场方向的保证。剩余部分保持现金。价格指数只生成风险状态；长期收益、回撤和超额收益使用独立的全收益基准计算。

## 二、回测中建模的 A 股约束

| 约束 | 实现方式 |
| --- | --- |
| 避免未来函数 | 月末收盘生成信号，下一交易日开盘才成交 |
| 成分股幸存者偏差 | 使用 `index_weight` 的历史月度成分快照 |
| 行业回看偏差 | 使用 `index_member_all` 的纳入/剔除区间，按信号日查询，不用当前行业回填历史 |
| 市值风格数据 | 使用信号日 `daily_basic.total_mv/circ_mv`，不使用未来日期市值 |
| ST 历史状态 | 使用历史曾用名区间，而不是当前名称回填过去 |
| 除权除息 | 复权价只用于因子；组合按实际股数、现金分红应收和送股批次估值 |
| T+1 | 每笔买入形成独立批次，下一交易日才转为可卖 |
| 买入整手 | 买单向下取整为 100 股整数倍 |
| 零股卖出 | 清仓时允许一次性卖出剩余零股 |
| 停牌 | 当日没有行情或成交量为 0 时拒单 |
| 涨跌停 | 开盘触及涨停不买，开盘触及跌停不卖；优先使用数据源的每日涨跌停价 |
| 滑点 | 默认买入加 5 bps、卖出减 5 bps，并限制在涨跌停价格内 |
| 成交容量 | 单日订单不超过已知 20 日平均成交额的 5% |
| 费用 | 按成交日期查询费用表；覆盖 2022 年过户费和 2023 年印花税调整 |
| ST/退市 | 执行日重新检查 ST；退市后按明确策略结算或核销，禁止永久保留旧市值 |
| 陈旧价格 | 超过指定交易日没有行情时警告或终止回测，不再静默处理 |
| 失败重试 | 固定信号日目标股数，最多尝试 3 个交易日，之后取消 |

手续费均可在配置中修改。券商实际佣金、最低收费和监管费率可能不同，运行真实资金前必须按自己的账户确认。

## 三、快速开始

需要 Python 3.10 或更高版本。

推荐用已提交的 `uv.lock` 创建精确依赖环境：

```bash
uv sync --locked
```

也可以使用传统虚拟环境，但这只遵守依赖范围，不保证安装到完全相同的小版本：

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

### 2. 用公开数据复现实证回测（无需 Token）

公开通道使用第三方项目根据中证指数官方公告整理的历史成分区间，并通过公开 HTTPS 行情接口下载日线和后复权因子。使用 uv 时执行下面的锁定命令；显式指定 PyPI 可以避免本机 `UV_DEFAULT_INDEX` 镜像设置要求改写锁文件。使用 pip 时按后文安装额外依赖并取得历史成分文件：

```bash
uv sync --locked --extra public --default-index https://pypi.org/simple
```

```bash
pip install -r requirements-public.txt
git clone --depth 1 --branch v0.6.2 \
  https://github.com/unliftedq/index-constitution.git ../public_data/index-constitution
```

下载 2012–2025 数据。程序逐证券写入压缩缓存，支持中断后原命令续传：

```bash
python run.py public-download \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300 \
  --source sina --start 2012-01-01 --end 2025-12-31 --workers 6
```

v1.3 已下载的公开行情不必重下。首次升级先为现有文件建立不可变指纹，随后可随时复核：

```bash
python run.py public-verify \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300 --seal-legacy

python run.py public-verify \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300
```

`public-research` 和 `public-robustness` 也会在第一次运行时自动封存旧缓存，以后发现任何行情或成分文件变化都会终止并报错。

v1.4.2 起，`--workers` 只控制 HTTP 请求和 DataFrame 转换并发度。新浪 JavaScript 解码固定由一个专用线程串行执行，所有 worker 通过有界队列提交任务，不再各自初始化 V8。可在本机运行不访问网络的真实运行时压力测试：

```bash
uv run --no-sync python tests/stress_sina_decoder.py \
  --workers 6 --tasks 600 --repeats 3
```

运行六组 v1.3 旧策略和一个冻结权重的 v1.5 质量动量候选，并分别输出 2013–2017 开发期、2018–2021 验证期、2022–2025 已查看历史保留期和全区间指标。全部历史区间都已经被用于研究，后续不得再称为未触碰样本外；真正的前瞻检验应从 2026 年以后数据或模拟盘开始：

```bash
python run.py public-research \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300 \
  --output results/public_research
```

对冻结的 `quality_momentum_v1_5` 候选运行 5/10/20 bps 单边滑点压力和逐因子删除：

```bash
python run.py public-robustness \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300 \
  --output results/public_research/robustness
```

核心结果在 `results/public_research/period_metrics.csv`，每组策略还会生成净值、月度选择、调仓和数据质量文件。公开通道是为了在无商业数据权限时验证收益方向，不冒充交易所级仿真：它没有可靠的历史 ST、历史行业和历史市值快照，采用权重级成本模型，也不模拟整手、最低佣金、涨跌停排队和停牌延迟。因此，公开结果应作为筛选依据，最终候选仍要回到下面的严格 Tushare 通道复核。

### 3. 配置严格真实 A 股数据（Tushare）

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

首次下载需要逐月获取历史成分，并获取历史成分股的日线、复权因子、每日市值、涨跌停价、曾用名、历史行业区间、证券主表、退市日期和分红送股。相关 Tushare 接口通常需要至少 2000 积分。耗时取决于回测区间、接口频率和历史成分数量。缓存完成后，重复研究只需：

```bash
python run.py backtest --config config.yaml
```

运行完整研究套件：

```bash
python run.py research --config config.yaml
```

默认会生成 v1.4/v1.5 冻结 Alpha 同口径对照、“完整模型 + 当前正权重因子逐一剔除”、5/10/20 bps 滑点与 1/2 倍券商佣金组合、以及 5 年研究期后逐年向前滚动的固定参数样本外结果。法定印花税和过户费不会随佣金倍数放大。

也可以分开执行：

```bash
python run.py download --config config.yaml
python run.py validate-data --config config.yaml
python run.py backtest --config config.yaml
```

v1.4 严格缓存升级为 v4：新增独立的 `regime.csv.gz`，并在 `manifest.json` 保存所有实际输入文件的大小和 SHA256。旧 v3 缓存不能静默复用；首次升级必须把 `data.refresh` 改为 `true` 运行一次 `download`，完成后再改回 `false`。

## 四、输出文件

默认写入 `results/latest/`：

| 文件 | 内容 |
| --- | --- |
| `report.html` | 自包含图表、指标、月度收益、配置和风险说明 |
| `performance.png` | 净值与回撤图 |
| `metrics.json` | Alpha 配置身份、策略、全收益基准、风险匹配基准、换手、费用和公司行动指标 |
| `equity_curve.csv` | 每日策略净值、现金、分红应收、持仓、陈旧持仓数及两条基准净值 |
| `selections.csv` | 每次调仓的因子值、排名和目标权重 |
| `industry_exposure.csv` | 每次调仓的行业持股数、绝对目标权重和股票仓位内占比 |
| `style_exposure.csv` | 市值、动量、趋势、低波和流动性暴露及缓冲保留比例 |
| `orders.csv` | 所有成交、拒单和撤单及原因 |
| `trades.csv` | 实际成交明细和费用拆分 |
| `corporate_events.csv` | 现金分红、送股、到账和退市核销事件 |
| `final_positions.json` | 期末持仓市值 |
| `resolved_config.yaml` | 本次运行的完整配置快照 |
| `runtime.json` | Python、NumPy、pandas 和系统版本 |
| `reproducibility.json` | 配置、源码、Git、依赖、数据清单和本次运行的组合指纹 |
| `experiment_registry.jsonl` | 按运行指纹登记实验协议及每个产物的大小和 SHA256 |
| `artifact_manifest.json` | 本次输出文件集合、逐文件 SHA256 和集合指纹 |

`research` 命令默认在 `results/latest/research/` 生成：

| 文件 | 内容 |
| --- | --- |
| `alpha_comparison.csv` | 冻结的 `legacy_v1_4` 与 `quality_momentum_v1_5` 在相同非 Alpha 参数下的指标 |
| `factor_ablation.csv` | 完整模型和逐个剔除当前正权重因子的同口径指标 |
| `cost_stress.csv` | 不同滑点/券商佣金组合的收益、回撤、换手和费用 |
| `rolling_oos.csv` | 扩展研究窗、非重叠测试窗的固定参数样本外指标 |
| `research_manifest.json` | 本次研究模式、压力参数和窗口参数 |
| `reproducibility.json` | 研究配置、代码和数据的完整可复现身份 |
| `experiment_registry.jsonl` | 固定参数研究协议和实验身份登记 |
| `artifact_manifest.json` | 研究输出的逐文件 SHA256 和集合指纹 |

`public-research` 命令默认在 `results/public_research/` 生成：

| 文件 | 内容 |
| --- | --- |
| `period_metrics.csv` | 六组旧策略与一个 v1.5 候选在开发、验证、已查看历史保留期和全区间的收益、风险、基准与换手 |
| `research_manifest.json` | 数据路径、区间状态、策略参数和公开数据局限 |
| `reproducibility.json` | 历史成分、行情缓存、代码、依赖与策略参数指纹 |
| `experiment_registry.jsonl` | 明确记录 2022–2025 已查看状态的实验登记 |
| `artifact_manifest.json` | 报告、图表、指标和全部策略明细的逐文件 SHA256 |
| `<strategy>/equity_curve.csv` | 策略与沪深300ETF后复权基准净值 |
| `<strategy>/selections.csv` | 月末截面因子、排名、缓冲原因与目标权重 |
| `<strategy>/rebalances.csv` | 次日开盘调仓、双边换手与成本 |
| `<strategy>/data_quality.json` | 历史成员覆盖、缺失成交和已知限制 |
| `robustness/cost_stress.csv` | v1.5 候选在 5/10/20 bps 单边滑点下的全区间与已查看历史保留期结果 |
| `robustness/factor_ablation.csv` | 逐一删除 v1.5 六个正权重因子的同口径结果 |

每次生成结果后可校验所有已封存文件：

```bash
python run.py result-verify --output results/demo --strict
python run.py result-verify --output results/public_research
python run.py result-verify --output results/public_research/robustness --strict
```

第二条未使用 `--strict`，是因为公开研究目录允许随后增加独立封存的 `robustness/` 子目录。公开数据会被上游修订；研究结果必须与 `reproducibility.json` 中的数据指纹共同归档，不能仅凭重新下载的数据宣称严格复现。

## 五、核心参数怎么调

`config.example.yaml` 已包含全部参数。建议一次只改一类，并保留每次运行的 `resolved_config.yaml`。

### 股票池与数据

- `universe_index`：默认 `399300.SZ`。若换中证 500 等指数，先确认 Tushare 的指数代码和历史成分权限。
- `regime_index`：默认 `000300.SH`，即沪深 300 价格指数，只用于 200 日均线风险开关。
- `benchmark_index`：默认 `H00300.CSI`，即沪深 300 全收益指数。不要改回普通价格指数后直接比较长期超额。
- `warmup_calendar_days`：默认 500 个自然日，为 252 个交易日动量和 200 日均线留缓冲。
- `calls_per_minute`：不能高于你的接口权限。遇到限流时调低。
- `industry_standard` / `industry_level`：默认申万 2021 一级行业。层级越细，接口调用和约束不可行风险越高。

### 选股

- `top_n`：默认 20。越小个股风险越集中，越大则信号稀释、交易笔数增加。
- `selection_buffer_enabled` / `exit_rank`：是否启用实际持仓缓冲及退出排名；退出排名不得小于 `top_n`。
- `min_avg_amount_million`：20 日平均成交额下限，单位百万元。
- `stock_trend_filter`：关闭后允许长期趋势为负的股票进入候选。
- 八个 `*_weight`：会自动按权重总和归一化，不要求手工加到 1，但必须是非负有限数。默认六项为正权重，总波动和流动性选股项为 0。
- `industry_neutralization_enabled`：是否从选股分数中剔除行业截面效应。
- `size_neutralization_enabled` / `size_neutralization_strength`：是否及多大比例剔除对数总市值线性暴露。

### 风险与交易

- `risk_on_exposure` / `risk_off_exposure`：风险开启和关闭时总股票仓位。
- `max_stock_weight`：单股绝对权重上限。必须满足 `top_n × max_stock_weight >= risk_on_exposure`。
- `max_industry_weight`：单个行业占总资产的目标权重硬上限；行业不足时允许组合低配现金。
- `slippage_bps`：单边滑点。建议同时压力测试 5、10、20 bps。
- `max_participation_of_20d_amount`：订单金额占过去 20 日平均成交额的上限。
- `rebalance_retry_days`：因涨跌停、停牌或资金不足导致未成交时的最大尝试天数。
- `fee_schedule`：历史费率生效区间；区间必须连续且不得重叠。
- `maximum_stale_trading_days`：持仓价格允许陈旧的最大交易日数。
- `stale_price_policy`：`warn` 继续估值并报警，`error` 立即终止回测。
- `delist_value_policy`：默认 `zero` 保守核销；只有拥有可靠结算数据时才使用 `last_close`。
- `annual_cash_rate`：风险匹配基准和组合现金使用的年化收益率假设。

## 六、如何做更可信的研究

不要只看一次全区间回测的年化收益。推荐按下面顺序评估：

1. 把较早年份作为参数研究期，较晚年份作为完全不参与调参的样本外期。
2. 运行 `research`，检查逐因子消融是否仍有一致收益来源；单个因子剔除后崩溃通常意味着模型脆弱。
3. 做滚动或扩展窗口验证，例如用前 5 年确定并冻结参数、下一年验证，再向前滚动。内置工具不会自动寻优，训练窗只是研究/冻结边界。
4. 对佣金、滑点和成交容量做至少 2～4 倍压力测试。
5. 分牛市、熊市、震荡市检查收益来源，不只看总体夏普。
6. 检查 `industry_exposure.csv`、`style_exposure.csv` 和 `selections.csv`，确认约束和缓冲实际生效。
7. 检查 `orders.csv` 中涨跌停、停牌和容量拒单，避免把“理论目标仓位”当作“实际可成交仓位”。
8. 与沪深 300 全收益指数及“相同股票仓位＋现金”的风险匹配基准同时比较。
9. 锁定参数后再增加最新数据；不要看到新结果后反复回头改参数。

## 七、已知局限

- 日线数据只能判断开盘是否触及涨跌停，不能模拟集合竞价排队、盘口深度、部分成交和盘中价格路径。
- Tushare 数据可能修订，接口权限和字段也可能变化；v1.4 会冻结缓存指纹，但仍应把数据目录与研究结果一同归档。
- 指数历史成分降低了幸存者偏差，但不能消除指数编制本身的选择偏差。
- 分红送股已进入股份账本，但个人红利税与持有期相关，复杂税务仍需券商级清算数据。
- 普通现金分红、送股和退市已处理；配股、换股吸收合并、破产重整等特殊事件仍需扩展。
- 默认因子仍全部来自价格路径；成交额只做可交易性过滤，市值只用于风格控制，没有基本面质量、估值、盈利修正或完整协方差风险模型。
- 申万行业成员和每日市值依赖数据源历史覆盖与修订；下载后的 v4 缓存与 `reproducibility.json` 必须和结果一起归档。
- 行业/市值残差化是截面线性控制，不等于 Barra 类多因子风险模型，也不保证实际组合暴露严格为零。
- 默认基准已切换为全收益指数，但实际数据源是否完整仍应通过 `validate-data` 确认。
- 示例不是券商实盘系统，没有账户同步、订单回报、撤改单、风控审批、交易时段保护和灾备。

若要走向仿真/实盘，至少还应加入：数据快照版本、券商适配层、目标仓位与账户持仓对账、价格笼子、单笔/单日损失限制、重复下单幂等键、人工审批、成交回报重放和紧急停机开关。

## 八、测试

无需 Tushare Token：

```bash
python -m unittest discover -s tests -v
```

当前包含 58 项测试，覆盖：历史费率边界、T+1 股份批次、NaN/非法 OHLC、行情/业绩基准/择时指数/行业缺口、择时与业绩基准隔离、数据源数字日期、行业区间覆盖、旧缓存拒绝、成分快照断档、历史时点行业、未来数据隔离、FIP 路径方向、冻结 Alpha 对照、负权重拒绝、排名缓冲、行业/单股权重约束、信号日冻结股数、执行日 ST、涨停重试、现金与持仓恒等式、v1.5 确定性黄金结果、现金分红、送股、退市核销、陈旧估值、暴露报表、因子消融、成本压力、滚动样本外、公开成分区间边界、停牌锁仓、实验登记幂等性和严格/公开缓存篡改检测。可选 MiniRacer 未安装时，两个真实运行时压力测试会显示为跳过。

## 九、项目结构

```text
a_share_quant/
├── config.example.yaml
├── requirements.txt
├── uv.lock            # 精确依赖锁
├── run.py
├── V1.5_ALPHA.md       # Alpha 公式、冻结权重和历史检验边界
├── V1.4_TRUSTWORTHINESS.md
├── V1.2_VALIDATION.md  # 改造验收、合成对照和压力结果
├── src/ashare_quant/
│   ├── backtest.py     # 组合、订单、费用、T+1、涨跌停与重试
│   ├── alpha.py        # 严格/公开通道共享的价格 Alpha 与冻结权重
│   ├── cli.py          # 命令行入口
│   ├── config.py       # 配置与参数校验
│   ├── data.py         # Tushare 下载、缓存、数据校验、离线演示
│   ├── factors.py      # 因子、动态成分、选股、风险开关与权重
│   ├── provenance.py   # 数据、代码、配置与环境指纹
│   ├── report.py       # 指标、暴露、CSV、PNG 和 HTML 报告
│   └── research.py     # 冻结 Alpha 对照、因子消融、成本压力和滚动样本外
└── tests/test_quant.py
```

## 十、规则与数据接口参考

- [Tushare A股日线行情 `daily`](https://tushare.pro/wctapi/documents/27.md)
- [Tushare 指数成分和权重 `index_weight`](https://tushare.pro/wctapi/documents/96.md)
- [Tushare 每日涨跌停价格 `stk_limit`](https://tushare.pro/wctapi/documents/183.md)
- [Tushare 股票曾用名 `namechange`](https://tushare.pro/wctapi/documents/100.md)
- [Tushare 交易日历 `trade_cal`](https://tushare.pro/wctapi/documents/26.md)
- [Tushare 指数日线 `index_daily`](https://tushare.pro/wctapi/documents/95.md)
- [Tushare 分红送股 `dividend`](https://tushare.pro/wctapi/documents/103.md)
- [Tushare 股票基础信息与退市日期 `stock_basic`](https://tushare.pro/document/2?doc_id=25)
- [Tushare 每日指标与总/流通市值 `daily_basic`](https://tushare.pro/document/2?doc_id=32)
- [Tushare 申万行业分类 `index_classify`](https://tushare.pro/document/2?doc_id=181)
- [Tushare 申万行业历史成员 `index_member_all`](https://tushare.pro/document/2?doc_id=335)
- [中证指数沪深 300 资料：全收益指数 H00300](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf)
- [Da、Gurun、Warachka：Frog in the Pan（RFS 2014）](https://doi.org/10.1093/rfs/hhu003)
- [上海证券交易所交易规则（2026年修订）](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)
- [深圳证券交易所交易规则（2026年修订）](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)
- [财政部、税务总局关于减半征收证券交易印花税的公告](https://xj.mof.gov.cn/zcfagui/202311/t20231108_3915476.htm)

交易所规则、税费和数据权限会调整。代码中的默认参数是截至项目编写时的研究假设，实际使用时应重新核对。
