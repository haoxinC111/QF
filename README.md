# A股动态股票池多因子量化回测

这是一个研究/回测级的 Python 项目。v1.5.1 根据真实公开历史回放把生产 Alpha 默认恢复为 `legacy_v1_4`；v1.6 在保持该 Alpha 与入选证券不变的前提下，新增可审计的收缩协方差组合模型和平方根市场冲击模型。V2.0 Alpha 1 新增独立的 Point-in-Time 财报/估值侧车，Alpha 2 新增固定因子研究，Alpha 3 再把固定 PIT 候选放入同一严格账本做四臂影子归因。所有 V2 候选仍为研究专用，不能从配置静默进入生产；v1.6 生产路径与结果保持不变。严格通道继续建模历史行业/市值、分红送股、信号与成交错位、涨跌停、停牌、退市、T+1、100 股整手、历史费用、部分成交、滑点和成交容量。

> 重要：本项目不构成投资建议，不承诺收益，也没有连接券商下单。先用离线演示确认环境，再用自己的数据权限回测；任何真实资金使用前，都应做样本外检验、压力测试、人工复核和小资金仿真。

## 一、策略概览

默认策略是使用 `legacy_v1_4` Alpha 的“历史沪深 300 成分股内月度截面多因子策略”。它只做多，不加杠杆，不做日内交易。冻结的 `quality_momentum_v1_5` 可显式复核，但其状态为 `experimental`，默认晋级决定为 `rejected`。

每个自然月最后一个交易日收盘后：

1. 取得当时已生效的指数成分，而不是今天的成分股。
2. 剔除 ST、历史不足、成交额不足、价格过低以及长期趋势为负的股票。
3. 计算五个正权重价格因子，做截面缩尾和标准化；综合分数对当日有效申万行业和对数总市值做截面残差化。
4. 新股票按前 20 名进入；已实际持有的股票只要仍在前 35 名即可保留，减少排名边界附近的无效往返交易。
5. 生产基线用逆波动率分配权重；v1.6 实验模型可在同一入选集合上混合收缩协方差最小方差偏好和当前持仓。两者都保证单股不超过 8%、单行业不超过 25%，不可行时保守降低股票总敞口。
6. 用沪深 300 **价格指数**的 200 日均线决定总股票仓位：风险开启 95%，风险关闭 30%；业绩比较单独使用沪深 300 **全收益指数**。
7. 信号日收盘冻结订单股数，下一交易日只按实际开盘价决定成交金额；遇到停牌、涨跌停或容量限制，最多重试 3 个交易日。

### 因子定义

所有价格因子都使用 `原始价格 × 复权因子` 的总回报序列。令该序列收盘价为 \(P_t\)：

| 因子 | 定义 | `legacy_v1_4` 默认权重 | 意图 |
| --- | --- | ---: | --- |
| 12-1 动量 | \(P_{t-21}/P_{t-252}-1\) | 35% | 捕捉中长期趋势，跳过最近约一个月 |
| 6-1 动量 | \(P_{t-21}/P_{t-126}-1\) | 20% | 补充中期趋势 |
| FIP 连续信息动量 | \(MOM12\_1\times(1-ID)\) | 0% | v1.5 实验候选因子 |
| 长期趋势 | \(P_t/MA_{200}(P)-1\) | 15% | 偏向处于长期上升趋势的股票 |
| 低下行波动 | 60 日年化下行波动率的相反数 | 0% | v1.5 实验候选因子 |
| 回撤质量 | \(P_t/Max_{126}(P)-1\) | 0% | v1.5 实验候选因子 |
| 低总波动 | 60 日年化波动率的相反数 | 20% | 偏向较低历史波动的股票；总波动也用于定仓 |
| 流动性 | \(\log(1+20日平均成交额)\) | 10% | 在通过成交额门槛后偏向更高流动性 |

FIP 的信息离散度为 \(ID=sign(MOM12\_1)\times(\%neg-\%pos)\)，正负日比例只取 12–1 月形成窗口内截至信号日可见的 231 个日收益。定义来自 [Da、Gurun 与 Warachka（2014）](https://doi.org/10.1093/rfs/hhu003)。对每个因子在当日股票池内进行 5%/95% 分位缩尾，再转为 z-score。生产默认综合分数为：

\[
\begin{aligned}
Score_i={}&0.35Z(MOM12\_1)+0.20Z(MOM6\_1)+0.15Z(Trend)\\
&+0.20Z(-Volatility)+0.10Z(Liquidity)
\end{aligned}
\]

v1.5 实验候选的公式、冻结权重和原始协议详见 [`V1.5_ALPHA.md`](V1.5_ALPHA.md)，默认回退与数据完整性修复见 [`V1.5.1_GOVERNANCE.md`](V1.5.1_GOVERNANCE.md)。v1.6 的组合公式、冲击公式、四臂归因和晋级边界见 [`V1.6_PORTFOLIO_EXECUTION.md`](V1.6_PORTFOLIO_EXECUTION.md)。2013–2025 已被旧版本研究查看，不属于未触碰样本外；项目不会为了达到指定年化收益而自动调参。

原始综合分数形成后，程序以当日有效的行业哑变量和 `log(total_mv)` 做横截面最小二乘残差化；`size_neutralization_strength=1` 表示完整剔除线性市值暴露。这里的“中性”只针对选股分数，不意味着实际组合对所有风险因子严格零暴露，实际结果应查看 `style_exposure.csv`。

选股与权重是两个不同步骤：中性化后的综合分数决定“买谁”，逆波动率决定“买多少”。权重使用水位分配算法同时满足总敞口、单股上限和行业上限。若股票或行业数量不足，程序会自动降低总敞口，而不会突破硬约束凑仓位。

### 排名缓冲

缓冲使用信号时点的**实际持仓**，不是上一次理论目标。默认 `top_n=20`、`exit_rank=35`：未持有股票按得分从高到低补足 20 只；已持有股票若仍在前 35 名且继续满足 ST、历史、价格、流动性和趋势过滤，可优先保留。`selections.csv` 中 `selection_reason=HOLD_BUFFER` 表示由缓冲保留。缓冲降低换手，但也可能延迟响应因子反转，因此必须同时比较关闭缓冲的结果。

### 风险开关

信号日沪深 300 价格指数收盘价不低于其 200 日均线时，目标股票仓位为 95%；否则为 30%。这是一个简单、可解释的系统性风险控制，不是对市场方向的保证。剩余部分保持现金。价格指数只生成风险状态；长期收益、回撤和超额收益使用独立的全收益基准计算。

### V2.0 第一阶段：时点数据基础

`2.0.0a1` 把财报和日终估值放在独立 `data/pit_cache/` 侧车中，并通过基础行情 manifest 的 SHA256 与数据指纹绑定。财报默认在公告后的下一交易日可见；修订只从其自身可用日开始替换旧版本。读取时重新计算可用日并核对全部分区 SHA256，未来记录扰动不能改变过去快照。

该模块目前只支持下载、校验和导出研究快照，不参与默认选股或权重计算。因此它是后续基本面 Alpha 研究的可信输入层，不是收益提升声明。完整数据契约、命令和验收边界见 [`V2.0_ALPHA1_PIT_DATA.md`](V2.0_ALPHA1_PIT_DATA.md)。

### V2.0 第二阶段：基本面/估值因子研究

`2.0.0a4` 在 PIT 侧车上注册 11 个固定盈利、质量、成长和估值因子，以及一个不拟合权重的等权研究组合。`pit-research` 按月末历史成分生成研究面板，输出逐日覆盖率、21/63 日 IC、分组收益、行业/市值暴露、逐因子消融、成本压力和扩展窗滚动验证。

候选 `fundamental_value_composite_v2_alpha2` 的生产权重固定为 0；治理程序即使全部通过也只允许进入人工候选复核，不会修改 `MultiFactorStrategy`。公式、输出和真实数据验收边界见 [`V2.0_ALPHA2_FACTOR_RESEARCH.md`](V2.0_ALPHA2_FACTOR_RESEARCH.md)。

### V2.0 第三阶段：PIT 候选影子接入

`2.0.0a7` 新增 `pit-shadow`：它先严格校验 Alpha 2 研究包和基础行情/PIT 双指纹，再用同一个严格账本运行生产价格基线、PIT 覆盖匹配价格臂、PIT 单独臂和固定 25% PIT 混合臂。覆盖匹配臂用于把缺失数据筛选效应与真正的 PIT 排名增量分开。

影子执行分数结构上拒绝未来收益字段，候选无法写入生产 YAML。即使历史门槛全部通过，治理结果最多为 `eligible_for_forward_paper_tracking`。完整协议见 [`V2.0_ALPHA3_SHADOW_INTEGRATION.md`](V2.0_ALPHA3_SHADOW_INTEGRATION.md)。

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
| 滑点与冲击 | 默认固定 5 bps；v1.6 可显式启用波动率 × 参与率平方根冲击，并限制在涨跌停价格内 |
| 成交容量 | 单日订单不超过已知 20 日平均成交额的 5% |
| 费用 | 按成交日期查询费用表；覆盖 2022 年过户费和 2023 年印花税调整 |
| ST/退市 | 执行日重新检查 ST；退市后按明确策略结算或核销，禁止永久保留旧市值 |
| 陈旧价格 | 超过指定交易日没有行情时警告或终止回测，不再静默处理 |
| 部分成交与重试 | 固定信号日目标股数；容量不足按整手部分成交，剩余订单最多尝试 3 个交易日，之后取消 |

手续费均可在配置中修改。券商实际佣金、最低收费和监管费率可能不同，运行真实资金前必须按自己的账户确认。

## 三、快速开始

需要 Python 3.11 或更高版本。

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

只验证 v1.6 组合与成交链路（仍是假行情，不是收益证据）：

```bash
python run.py demo --output results/demo_v1_6 \
  --portfolio-model shrinkage_min_variance_v1_6 \
  --execution-model square_root_v1_6
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

用同一份公开缓存运行 v1.6 固定 Alpha 的四臂归因：

```bash
python run.py public-implementation \
  --membership ../public_data/index-constitution/history/csi300.csv \
  --cache ../public_data/sina_csi300 \
  --output results/public_implementation_v1_6 \
  --initial-capital 1000000

python run.py result-verify \
  --output results/public_implementation_v1_6 --strict
```

该命令会对比旧基线、仅新组合、仅新成交和组合+成交。它能直接复用 2012–2025 公开缓存，但仍是权重级近似：平方根冲击按 100 万元初始资金、信号日 ADV20 和波动率估算；不模拟整手、最低佣金、涨跌停排队、容量部分成交与失败重试。严格成交结论必须回到 Tushare 通道。

核心结果在 `results/public_research/period_metrics.csv`，每组策略还会生成净值、月度选择、调仓和数据质量文件。公开通道是为了在无商业数据权限时验证收益方向，不冒充交易所级仿真：它没有可靠的历史 ST、历史行业和历史市值快照，采用权重级成本模型，也不模拟整手、最低佣金、涨跌停排队和停牌延迟。因此，公开结果应作为筛选依据，最终候选仍要回到下面的严格 Tushare 通道复核。

v1.5.1 起，`data_quality.json` 不再只输出文件数和总体覆盖率，还会列出完全无行情的历史成员、每个信号月的成分数/报价数/缺失证券、低于 95% 告警阈值的月份，以及执行日缺失行情的去重证券。告警不会自动证明数据错误，也不会被静默忽略：应结合停牌和成分区间逐项核查。

### 3. 配置严格真实 A 股数据（Tushare）

复制配置：

```bash
cp config.example.yaml config.yaml
```

示例配置默认声明：

```yaml
strategy:
  alpha_profile: legacy_v1_4
```

若只想复核冻结的 v1.5 实验候选，可改为 `quality_momentum_v1_5`。报告会继续标记其 `experimental/rejected` 状态；这不会把它晋级为生产默认。自定义八项权重时必须使用 `alpha_profile: custom`，命名配置和冲突权重会直接报错。

v1.6 升级后组合与成交默认仍是旧生产基线。要运行新模型，必须显式配置：

```yaml
portfolio:
  construction_model: shrinkage_min_variance_v1_6
execution:
  market_impact_model: square_root_v1_6
```

报告会将两者标记为 `experimental/pending_validation`。这表示可以研究，不表示已获准替换生产默认。

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

默认会生成 v1.4/v1.5 冻结 Alpha 同口径对照、“完整模型 + 当前正权重因子逐一剔除”、5/10/20 bps 滑点与 1/2 倍券商佣金组合、5 年研究期后逐年向前滚动的固定参数评估，以及固定 Alpha 的 v1.6 组合/成交四臂归因。法定印花税和过户费不会随佣金倍数放大。

也可以分开执行：

```bash
python run.py download --config config.yaml
python run.py validate-data --config config.yaml
python run.py backtest --config config.yaml
```

若要准备 V2 的财报/估值时点数据，把配置中的 `point_in_time.enabled` 改为 `true`，在基础行情缓存完成后执行：

```bash
python run.py pit-download --config config.yaml
python run.py pit-verify --config config.yaml
python run.py pit-snapshot --config config.yaml --date 2020-04-01 \
  --output results/pit_snapshot_2020-04-01.csv
python run.py pit-research --config config.yaml \
  --output results/pit_factor_research_v2_alpha2
python run.py result-verify \
  --output results/pit_factor_research_v2_alpha2 --strict
python run.py pit-shadow --config config.yaml \
  --alpha2-research results/pit_factor_research_v2_alpha2 \
  --output results/pit_shadow_v2_alpha3
python run.py result-verify \
  --output results/pit_shadow_v2_alpha3 --strict
```

快照会同时生成包含文件 SHA256、PIT 数据指纹和基础行情指纹的 `.manifest.json`。Alpha 2/3 研究目录同时封存基础行情和 PIT 两份数据身份；这些命令都不会自动让财报数据进入生产策略。

v1.4 严格缓存升级为 v4：新增独立的 `regime.csv.gz`，并在 `manifest.json` 保存所有实际输入文件的大小和 SHA256。旧 v3 缓存不能静默复用；首次升级必须把 `data.refresh` 改为 `true` 运行一次 `download`，完成后再改回 `false`。

## 四、输出文件

默认写入 `results/latest/`：

| 文件 | 内容 |
| --- | --- |
| `report.html` | 自包含图表、指标、月度收益、配置和风险说明 |
| `performance.png` | 净值与回撤图 |
| `metrics.json` | Alpha/组合/成交模型身份与晋级状态、策略、基准、换手、参与率、滑点/冲击成本和公司行动指标 |
| `equity_curve.csv` | 每日策略净值、现金、分红应收、持仓、陈旧持仓数及两条基准净值 |
| `selections.csv` | 每次调仓的因子值、排名、风险原始目标、当前权重、最终目标、组合模型状态和协方差观测数 |
| `industry_exposure.csv` | 每次调仓的行业持股数、绝对目标权重和股票仓位内占比 |
| `style_exposure.csv` | 市值、动量、趋势、低波和流动性暴露及缓冲保留比例 |
| `orders.csv` | 所有成交、部分成交、拒单和撤单；含请求/剩余股数、容量、参与率、滑点与冲击审计 |
| `trades.csv` | 实际成交、参考价、模型/实现滑点、冲击成本和费用拆分 |
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
| `implementation_comparison.csv` | 固定 Alpha 的旧基线、仅新组合、仅新成交、组合+成交四臂归因 |
| `research_manifest.json` | 本次研究模式、压力参数和窗口参数 |
| `reproducibility.json` | 研究配置、代码和数据的完整可复现身份 |
| `experiment_registry.jsonl` | 固定参数研究协议和实验身份登记 |
| `artifact_manifest.json` | 研究输出的逐文件 SHA256 和集合指纹 |

`pit-research` 默认在 `results/pit_factor_research_v2_alpha2/` 生成 PIT 因子面板、覆盖率、IC、分组收益、行业/市值暴露、消融、滚动验证、成本压力、治理决定、双数据指纹和产物 SHA256。输出目录必须为空；完整文件表见 [`V2.0_ALPHA2_FACTOR_RESEARCH.md`](V2.0_ALPHA2_FACTOR_RESEARCH.md)。

`pit-shadow` 默认在 `results/pit_shadow_v2_alpha3/` 生成四臂严格账本、PIT 覆盖、年度一致性、选股重合、成本压力、治理决定和完整 SHA256。它要求 Alpha 2 目录先通过严格封存校验；完整文件表见 [`V2.0_ALPHA3_SHADOW_INTEGRATION.md`](V2.0_ALPHA3_SHADOW_INTEGRATION.md)。

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
| `<strategy>/data_quality.json` | 无行情成员、逐月成分行情覆盖、阈值告警、缺失成交和已知限制 |
| `robustness/cost_stress.csv` | v1.5 候选在 5/10/20 bps 单边滑点下的全区间与已查看历史保留期结果 |
| `robustness/factor_ablation.csv` | 逐一删除 v1.5 六个正权重因子的同口径结果 |

`public-implementation` 另在 `results/public_implementation_v1_6/` 生成 `implementation_comparison.csv`、四个方案的净值/选择/调仓明细、公开口径说明、复现指纹和产物 SHA256。

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

### V2 时点数据

- `point_in_time.enabled`：默认 `false`；只有显式启用后，PIT 命令和联合数据校验才会运行。
- `history_years`：在回测开始日前额外下载的财报历史年数，用于后续构造增长与 TTM 指标。
- `fundamental_lag_trading_days`：财报公告后的可见性滞后，默认 1；不要为了改善回测把它改成不符合信号时刻的数值。
- `valuation_lag_trading_days`：日终估值的可见性滞后；收盘后信号可用 0，开盘前信号应使用 1。
- `maximum_*_age_days`：导出快照时允许使用的最大数据年龄。
- `minimum_symbol_coverage`：整个请求区间内的证券最低覆盖率；它不是逐截面覆盖率或 Alpha 有效性证明。
- Alpha 2 研究参数通过 `pit-research` 命令显式传入，不会进入或改变生产 `strategy` 配置。

### 选股

- `top_n`：默认 20。越小个股风险越集中，越大则信号稀释、交易笔数增加。
- `selection_buffer_enabled` / `exit_rank`：是否启用实际持仓缓冲及退出排名；退出排名不得小于 `top_n`。
- `min_avg_amount_million`：20 日平均成交额下限，单位百万元。
- `stock_trend_filter`：关闭后允许长期趋势为负的股票进入候选。
- `alpha_profile`：`legacy_v1_4` 是生产默认；`quality_momentum_v1_5` 是冻结实验候选；`custom` 用于自定义权重。
- 八个 `*_weight`：仅在 `alpha_profile: custom` 时显式配置；会按权重总和归一化，不要求手工加到 1，但必须是非负有限数。`resolved_config.yaml` 始终保存最终展开后的实际权重和识别身份。
- `industry_neutralization_enabled`：是否从选股分数中剔除行业截面效应。
- `size_neutralization_enabled` / `size_neutralization_strength`：是否及多大比例剔除对数总市值线性暴露。

### 风险与交易

- `risk_on_exposure` / `risk_off_exposure`：风险开启和关闭时总股票仓位。
- `max_stock_weight`：单股绝对权重上限。必须满足 `top_n × max_stock_weight >= risk_on_exposure`。
- `max_industry_weight`：单个行业占总资产的目标权重硬上限；行业不足时允许组合低配现金。
- `portfolio.construction_model`：`inverse_vol_v1_4` 是生产默认；`shrinkage_min_variance_v1_6` 是实验模型，不改变入选股票，只改变目标权重。
- `covariance_lookback_days` / `minimum_covariance_observations`：v1.6 协方差回看和最低有效观测；不足时显式回退旧模型并写入状态。
- `covariance_shrinkage` / `minimum_variance_blend`：样本协方差向对角阵收缩比例，以及最小方差偏好与逆波动率偏好的混合比例。
- `turnover_smoothing`：把信号日实际持仓纳入目标偏好的强度；不会阻止不再入选证券退出，也不会突破单股/行业硬上限。
- `slippage_bps`：单边滑点。建议同时压力测试 5、10、20 bps。
- `market_impact_model`：`fixed_bps` 是生产默认；`square_root_v1_6` 按信号时点年化波动率和订单/滞后 ADV20 参与率增加冲击。
- `market_impact_coefficient` / `market_impact_volatility_floor` / `max_market_impact_bps`：平方根冲击系数、年化波动率下限和单边冲击上限。
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

- 日线数据只能判断开盘是否触及涨跌停；容量模型可以生成部分成交，但不能重放集合竞价排队、盘口深度、真实逐笔成交和盘中价格路径。
- Tushare 数据可能修订，接口权限和字段也可能变化；v1.4 会冻结缓存指纹，但仍应把数据目录与研究结果一同归档。
- 指数历史成分降低了幸存者偏差，但不能消除指数编制本身的选择偏差。
- 分红送股已进入股份账本，但个人红利税与持有期相关，复杂税务仍需券商级清算数据。
- 普通现金分红、送股和退市已处理；配股、换股吸收合并、破产重整等特殊事件仍需扩展。
- 生产默认因子仍全部来自价格路径；V2.0 Alpha 3 已提供 PIT 固定候选的严格账本影子归因，但尚未取得本仓库绑定的真实全量 PIT 证据，也未进入生产选股。v1.6 协方差是长-only 收缩最小方差近似，不是完整 Barra 风险模型或精确二次规划器。
- 申万行业成员和每日市值依赖数据源历史覆盖与修订；下载后的 v4 缓存与 `reproducibility.json` 必须和结果一起归档。
- 行业/市值残差化是截面线性控制，不等于 Barra 类多因子风险模型，也不保证实际组合暴露严格为零。
- 默认基准已切换为全收益指数，但实际数据源是否完整仍应通过 `validate-data` 确认。
- 示例不是券商实盘系统，没有账户同步、订单回报、撤改单、风控审批、交易时段保护和灾备。

若要走向仿真/实盘，至少还应加入：数据快照版本、券商适配层、目标仓位与账户持仓对账、价格笼子、单笔/单日损失限制、重复下单幂等键、人工审批、成交回报重放和紧急停机开关。

## 八、测试

当前包含 124 项测试，覆盖：Alpha 默认治理、命名配置冲突和旧 YAML 兼容、历史费率边界、T+1 股份批次、数据/缓存完整性、未来数据隔离、冻结 Alpha 对照、排名缓冲、行业/单股约束、信号日冻结股数、执行日 ST、涨停重试、现金与持仓恒等式、确定性黄金结果、公司行动、研究封存，以及 v1.6 协方差分散/历史不足回退/防前视、换手平滑、冲击单调性/上限、组合可行性、容量部分成交与三日重试、低于整手容量拒单、订单审计、CLI 参数路由、严格/公开四臂归因、公开冲击成本账本和公开产物 SHA256 封存。V2 另覆盖公告/修订可见性、未来值扰动隔离、指标/单位契约、PIT 分区/证券身份与基础行情绑定、下载续传封存、确定性数据指纹、配置迁移、严格标量类型、版本锁一致性、快照 SHA256、Alpha 2 因子研究，以及 Alpha 3 禁止未来标签、覆盖匹配、四臂严格账本、Alpha2 前置门禁、双指纹与全产物封存。可选 MiniRacer 未安装时，两个真实运行时压力测试会显示为跳过。

## 九、项目结构

```text
a_share_quant/
├── config.example.yaml
├── requirements.txt
├── uv.lock            # 精确依赖锁
├── run.py
├── V1.5_ALPHA.md       # Alpha 公式、冻结权重和历史检验边界
├── V1.5_VALIDATION.md  # 自动化验收、合成对照与公开数据续传状态
├── V1.5.1_GOVERNANCE.md # 默认回退、晋级状态、数据覆盖审计和 v1.6 边界
├── V1.6_PORTFOLIO_EXECUTION.md # 组合/成交公式、冻结参数、归因和晋级边界
├── V1.6_VALIDATION.md # 自动化验收、合成四臂结果和真实数据复核命令
├── V2.0_ALPHA1_PIT_DATA.md # PIT 财报/估值契约、可见性和使用方式
├── V2.0_ALPHA1_VALIDATION.md # 第一阶段自动化验收与 v1.6 回归
├── V2.0_ALPHA2_FACTOR_RESEARCH.md # PIT 因子公式、研究协议、治理和使用方式
├── V2.0_ALPHA2_VALIDATION.md # 第二阶段工程验收与真实数据复核命令
├── V2.0_ALPHA3_SHADOW_INTEGRATION.md # PIT 候选四臂严格账本与晋级边界
├── V2.0_ALPHA3_VALIDATION.md # 第三阶段工程验收与真实数据复核命令
├── V1.4_TRUSTWORTHINESS.md
├── V1.2_VALIDATION.md  # 改造验收、合成对照和压力结果
├── src/ashare_quant/
│   ├── backtest.py     # 组合、订单、费用、T+1、涨跌停与重试
│   ├── alpha.py        # 严格/公开通道共享的价格 Alpha 与冻结权重
│   ├── cli.py          # 命令行入口
│   ├── config.py       # 配置与参数校验
│   ├── data.py         # Tushare 下载、缓存、数据校验、离线演示
│   ├── pit_data.py     # V2 PIT 财报/估值下载、封存、校验和快照
│   ├── pit_research.py # V2 Alpha2 因子面板、诊断、治理与封存
│   ├── pit_shadow.py   # V2 Alpha3 影子分数、严格账本四臂归因与治理
│   ├── factors.py      # 因子、动态成分、选股、风险开关与权重
│   ├── portfolio.py    # 逆波动率基线、收缩协方差和换手平滑
│   ├── execution.py    # 固定滑点/平方根冲击公式与模型治理
│   ├── provenance.py   # 数据、代码、配置与环境指纹
│   ├── report.py       # 指标、暴露、CSV、PNG 和 HTML 报告
│   └── research.py     # Alpha、组合/成交归因、消融、成本压力和滚动评估
└── tests/
    ├── test_quant.py
    ├── test_public_research.py
    ├── test_pit_data.py
    ├── test_pit_research.py
    └── test_pit_shadow.py
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
- [Tushare 利润表 `income`](https://tushare.pro/document/2?doc_id=33)
- [Tushare 资产负债表 `balancesheet`](https://tushare.pro/document/2?doc_id=36)
- [Tushare 现金流量表 `cashflow`](https://tushare.pro/document/2?doc_id=44)
- [Tushare 财务指标 `fina_indicator`](https://tushare.pro/document/2?doc_id=79)
- [Tushare 申万行业分类 `index_classify`](https://tushare.pro/document/2?doc_id=181)
- [Tushare 申万行业历史成员 `index_member_all`](https://tushare.pro/document/2?doc_id=335)
- [中证指数沪深 300 资料：全收益指数 H00300](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf)
- [Da、Gurun、Warachka：Frog in the Pan（RFS 2014）](https://doi.org/10.1093/rfs/hhu003)
- [上海证券交易所交易规则（2026年修订）](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)
- [深圳证券交易所交易规则（2026年修订）](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)
- [财政部、税务总局关于减半征收证券交易印花税的公告](https://xj.mof.gov.cn/zcfagui/202311/t20231108_3915476.htm)

交易所规则、税费和数据权限会调整。代码中的默认参数是截至项目编写时的研究假设，实际使用时应重新核对。
