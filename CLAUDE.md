# A 股多因子量化回测项目 —— 本地管理规则

> 本文件只记录**当前项目**的特殊管理规则。全局偏好（中文回复、`uv` 优先、主目录单线开发等）继续遵循 `~/.claude/CLAUDE.md`。

## 1. 项目角色分工

- **云端 GPT 5.6 Sol-Ultra**：主要负责核心代码开发、策略迭代、生成新版本 ZIP。
- **本地 K2.7（Claude Code）**：负责接收 ZIP、解压、版本控制、本地复现、缺失项识别、文档整理。
- **用户**：在两端之间做决策；遇到需要云端模型回答的问题，由用户转达。

## 2. 版本来源与命名

- 云端每次交付物是一个 ZIP 文件，命名格式通常为：
  ```
  ashare_quant_multifactor_vX.Y.Z.zip
  ```
- 本地收到后，按语义版本号处理，不要凭文件修改时间判断先后。
- 已经历过的版本：
  - `v1.0.0`：基础严格回测引擎
  - `v1.2.0`：严格通道升级（行业/市值中性化、股份账本、研究套件）
  - `v1.3.0`：新增公开数据源研究通道（新浪财经/东方财富）
  - `v1.4.0`：修复公开停牌锁仓缩放，分离择时/业绩基准，加入数据与运行指纹
  - `v1.4.1`：清理旧结果产物，加入输出 SHA256、结果校验和运行模块来源
  - `v1.4.2`：新浪下载改为并发 HTTP + 单解码线程，加入跨平台压力测试
  - `v1.5.0`：新增共享质量动量 Alpha、冻结版本对照、逐因子消融和防前视测试
  - `v1.5.1`：恢复 legacy 默认 Alpha，标记 v1.5 候选治理状态，补充公开数据逐月覆盖审计

## 3. ZIP 处理标准流程

收到新 ZIP 后，按以下顺序执行：

1. **解压到临时目录**，不要直接覆盖当前项目。
   ```bash
   mkdir -p /tmp/quant_versions/vX.Y.Z
   unzip ashare_quant_multifactor_vX.Y.Z.zip -d /tmp/quant_versions/vX.Y.Z
   ```
2. **核对顶层结构**：有的版本把项目放在 `a_share_quant/` 子目录下，有的直接放在根目录。解压后先确认 `run.py` 和 `src/ashare_quant/` 的位置。
3. **与当前代码做差异比较**：
   - 核心模块：`src/ashare_quant/{config,data,factors,backtest,report,cli,research,public_research,provenance}.py`
   - 配置：`config.example.yaml`、`pyproject.toml`、`requirements*.txt`
   - 文档：`README.md`、`V1.*_VALIDATION.md`、`PUBLIC_SOURCE_AUDIT.md`
   - 测试：`tests/`
   - 结果：`results/`（生成物不进入源码包；需要发布时作为绑定数据指纹的独立 Release 附件）
4. **确认用户意图**后再合并到当前项目，不要自动覆盖工作目录。
5. **写入 Git 历史**：每次版本升级应为一个独立 commit，commit message 写明版本号和核心变化。

## 4. 本地复现检查清单

复现前必须先确认以下事项，缺失任何一项都要向用户报告：

### 4.1 源码完整性
- 当前目录是否已经是目标版本？（对比 `pyproject.toml` 里的 `version`）
- 新版本是否新增了模块（如 `research.py`、`public_research.py`）？
- `run.py` 是否兼容当前目录结构？

### 4.2 依赖
- 检查 `requirements.txt` 和 `requirements-public.txt` 是否有新增包。
- 公开数据源需要 `requests` 和 `akshare>=1.18,<2`（`akshare` 会带入 `py_mini_racer`）。
- 严格 Tushare 通道需要 `tushare>=1.4.21,<2`。

### 4.3 数据与外部文件
- **严格通道**：需要 `TUSHARE_TOKEN` 环境变量，且账号积分至少 2000。
- **公开通道**：必须准备 `unliftedq/index-constitution` 的 `history/csi300.csv`，固定提交 `16d9d69fc0bf7f0f5e9aace868e16e26f2ecb5c2`。
- 首次运行公开回测时需要联网下载行情缓存到 `data/public_eastmoney/`。
- v1.4 的严格缓存为 v4，新增 `regime.csv.gz` 和逐文件 SHA256；旧 v3 缓存需要 `data.refresh: true` 重新生成。
- v1.3 公开缓存无需重下，先运行 `public-verify --seal-legacy` 生成指纹；此后校验失败不得自动覆盖或重新封存。

### 4.4 网络与环境
- 确认能访问 `finance.sina.com.cn` 和/或东方财富公开接口。
- 注意 rate limit：东方财富批量约 35 只后容易超时，sina 源需要 `akshare` 解码。
- 不要假设 BaoStock 可用（其 TCP 端口在某些网络下会被阻断）。

### 4.5 配置
- 用 `config.example.yaml` 生成 `config.yaml`，不要直接修改示例文件。
- v1.4 配置新增 `data.regime_index`；升级时必须确认它是价格指数，并与全收益 `benchmark_index` 分离。

## 5. 何时必须提问或上报

遇到以下情况，不要自行绕过，先向用户说明：

1. **版本号跳跃或不一致**（例如收到 `v1.4.0` 但源码里写的是 `1.3.0`）。
2. **压缩包里缺少关键文件**（如 `src/ashare_quant/` 部分模块缺失、没有 `config.example.yaml`）。
3. **新增外部依赖无法安装**（尤其是 `akshare`/`py_mini_racer` 在 macOS/Apple Silicon 上的编译问题）。
4. **网络下载失败**（sina/eastmoney/Tushare 限流、IP 被封、端口不通）。
5. **运行结果与压缩包内 `results/` 不一致**，需要判断是数据差异、随机种子还是代码差异。
6. **用户要求把云端生成的结果推送到 GitHub 时**，需要确认是否包含 `results/`、`data/cache/` 等大文件。

## 6. Git 与 GitHub 操作原则

- 默认在 `a_share_quant/` 目录内直接开发，不要频繁切 worktree（遵循全局 `~/.claude/CLAUDE.md`）。
- 每次版本升级产生一个 commit，message 格式：
  ```
  vX.Y.Z: <一句话核心变化>
  ```
- 推送前确认：
  - 目标分支是否正确；
  - 是否意外提交了 `config.yaml`、缓存、虚拟环境或 `.DS_Store`；
  - `results/` 是否按用户要求包含或排除。

## 7. 与云端 GPT 的协作边界

- 本地 K2.7 **不替云端模型写策略代码**，只负责：
  - 解压、合并、版本控制；
  - 运行复现、捕获报错；
  - 整理缺失项清单转交给用户。
- 如果用户说“让云端模型回答”，本地应把问题、错误日志、缺失项清单整理清楚后交给用户，不擅自猜测云端代码意图。

## 8. 常用检查命令速查

```bash
# 查看当前版本号
python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"

# 运行离线 demo（无需 Token）
python run.py demo --output results/demo

# 严格通道数据校验
python run.py validate-data --config config.yaml

# 公开通道下载
python run.py public-download --membership csi300.csv --cache data/public_eastmoney --source sina

# 公开通道研究
python run.py public-research --membership csi300.csv --cache data/public_eastmoney --output results/public_research

# 运行测试
python -m unittest discover tests
```

## 9. 已知问题（v1.4 本地复现后发现）

- `uv.lock` 使用 PyPI 源生成。若本地默认 mirror 不是 PyPI（如 `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple`），使用 `uv sync --locked --extra public --default-index https://pypi.org/simple`。不要用已弃用的 `UV_INDEX_URL` 覆盖，也不要为了同步而去掉 `--locked`。
- v1.4.0 压缩包里的 `results/public_research/` 是旧构建产物；v1.4.1 已停止追踪整个 `results/`。本地新结果只能作为绑定完整数据指纹的参考结果，源码包和 GitHub tag 不再携带生成结果。
- issue #1（mini-racer 并发初始化 SIGTRAP）在 v1.4.2 改为单解码线程：worker 不再初始化或调用 V8；现代运行时使用 context manager，兼容实现存在 `close()` 时显式关闭。Linux/macOS CI 运行 6-worker 子进程压力测试；若真实下载仍出现原生崩溃，再升级为独立解码子进程。

## 10. 本文件维护

- 每次项目出现新的“本地管理问题”或“复现陷阱”时，更新本文件。
- 不要把策略逻辑、因子公式写进本文件，那些属于代码和 README。
