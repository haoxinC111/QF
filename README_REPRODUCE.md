# QF 数据归档 Phase A.1 + P0 复现说明

本包包含 A 股数据归档侧车的 Phase A.1 验收产物与 P0 全量归档基础设施。
**不含** 任何 Token / API Key、本地授权配置（`config.archive.yaml`）、以及
Raw/Bronze 全量数据（这些只存在于本地 `data_lake/`，不进 Git / ZIP）。

## 1. 环境准备

```bash
# Python 3.11+，依赖锁定在 uv.lock（PyPI 源生成）
env -u UV_DEFAULT_INDEX uv sync --locked --extra public --default-index https://pypi.org/simple
```

## 2. 配置

```bash
cp config.archive.example.yaml config.archive.yaml
```

`config.archive.example.yaml` 保持安全默认：

- `provider.authorization_confirmed: false`
- `provider.local_archival_allowed: false`

只有用户显式确认授权后，才能在本地 `config.archive.yaml`（已 gitignore）
把两项改为 `true`。网关地址与 Token 只从环境变量读取：

```bash
export QF_ARCHIVE_API_URL='https://<your-gateway>/pro'
export QF_ARCHIVE_API_TOKEN='<your-token>'
```

适配器**禁止**读取/转发官方 `TUSHARE_TOKEN` 到第三方网关
（`provider.forbid_token_env`），并有单元测试保证。

## 3. 验证与测试

```bash
env -u UV_DEFAULT_INDEX uv sync --locked --extra public --default-index https://pypi.org/simple
uv run --no-sync ruff check src tests
uv run --no-sync python -W error::FutureWarning -m unittest discover -s tests -v
```

## 4. Phase A.1 验收产物位置

| 产物 | 路径 |
|---|---|
| 权限探针 | `data_lake/reports/permission_report.json` |
| Endpoint 清单（含探针结果） | `data_lake/catalog/endpoint_inventory.yaml` |
| empty 任务调查 | `data_lake/reports/empty_task_review.json` |
| 市场跨源核验 | `data_lake/reports/market_cross_source_details.csv`、`cross_source_summary.json`、`cross_source_validation.md` |
| 财务跨源核验 | `data_lake/reports/financial_cross_source_details.csv`（结论合并进上述 summary/md） |
| soak 稳定性 | `data_lake/reports/soak_test_report.json/.md` |
| 测试日志摘要 | `data_lake/reports/test_run_summary.txt` |
| 最终门禁 | `data_lake/reports/phase_a1_decision.json` |
| 批次产物 | `data_lake/reports/batches/<batch_id>/{batch_manifest.jsonl,checksums.sha256,coverage_report.md,schema_report.json,failure_queue.jsonl,batch_decision.json}` |

注意：`data_lake/` 整体不进 Git；ZIP 中仅包含上述**报告类**文件
（json/md/csv/txt），不含 Raw/Bronze 数据文件。

## 5. 复算入口

```bash
# 权限探针（无需授权开关）
uv run --no-sync ashare-quant archive-probe --config config.archive.yaml --priorities P0 P1

# 市场跨源核验（种子 20260716，100 股 × 20 交易日）
uv run --no-sync python scripts/cross_source_market.py

# 财务跨源核验（21 家 × 4 季度，7 行业）
uv run --no-sync python scripts/cross_source_financial.py

# 1000 请求 soak 稳定性测试（75/min 单令牌桶，1 worker）
uv run --no-sync python scripts/soak_test.py

# 门禁汇总（只有 pass 才允许进入 P0 批次）
uv run --no-sync python scripts/phase_a1_decision.py

# P0 批次（B0..B4，顺序执行；B3 需要财务核验通过）
uv run --no-sync python scripts/run_p0_batch.py --batch B0_reference
uv run --no-sync python scripts/run_p0_batch.py --batch B1_market
uv run --no-sync python scripts/run_p0_batch.py --batch B2_universe
uv run --no-sync python scripts/run_p0_batch.py --batch B3_financial
uv run --no-sync python scripts/run_p0_batch.py --batch B4_events
```

## 6. 数据布局

```
data_lake/
  raw/<provider>/<snapshot>/<api>/<api>_<partition>_<snapshot>.json.zst   # 原始响应, 只追加
  bronze/<provider>/<api>/<api>_<partition>_<snapshot>.parquet            # 表格式
  catalog/                  # 任务状态库、schema 注册表、endpoint 清单
  reports/                  # 探针/核验/soak/批次报告
```

- Raw 不可变：同路径重写仅在 SHA256 完全一致时放行，否则拒绝。
- 原子发布：临时文件写成功后才 rename，失败不留 `.tmp` 半文件。
- schema 漂移：与已登记指纹不一致的响应会被隔离（quarantine），不落盘。
- 截断恢复：达到行数上限的响应先按日期二分，单日仍触顶则按 ts_code
  维度（全市场股票全集）拆分重取。

## 7. 下载完成后的 PIT 验收

B0、B1、B3 全部 `pass` 后，使用 `2.0.0a11` 的 Alpha5 门禁一次性完成离线桥接、源证据重放、逐月历史覆盖和回执封存：

```bash
uv run --no-sync python run.py pit-acceptance \
  --config config.yaml \
  --archive-root data_lake \
  --output results/pit_acceptance_v2_alpha5

uv run --no-sync python run.py result-verify \
  --output results/pit_acceptance_v2_alpha5 --strict
```

严格验收必须得到 `decision=pass`。`engineering_only` 只代表 fixture 工程样例通过；`blocked` 或任一 `quarantined` 任务都不能进入 PIT 因子研究。之后运行 `pit-research` / `pit-shadow` 时，应传入 `results/pit_acceptance_v2_alpha5/acceptance_report.json`。
