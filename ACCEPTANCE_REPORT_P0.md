# P0 全量归档交付验收报告(QF-integration 隔离环境)

- 验收日期: 2026-07-23/24
- 验收方: 本地 Claude Code(K2.7)
- 交付方: 下载 Agent
- 交付物版本: `QF_data_acquisition_phase_a1_p0_2.0.0a9.zip`(+ .sha256)、`QF-fixtures/catalog_snapshot.sqlite`、`QF-fixtures/samples.zip`(+ .sha256)
- 集成基线: QF-integration HEAD `7aaa571`(v2.0.0a11,冻结快照 `5b4d054`)
- 状态库快照指纹: `32ce478375271345b90087a9d02354fb2d0471a257b03e7c4d2e727c1b60e077`

---

## 总结论

**research_eligible = false。当前不可进入真实 Alpha 研究。**

阻断原因不是交付物质量(完整性校验全部通过),而是验收过程中**新发现两个交付方未披露的网关静默截断缺陷**,导致 v4 基础行情缓存的 G7/G8 前置门禁 fail-closed:

1. `index_member_all` 全表单拉恰 3000 行且无 `is_new=N` 历史行 → 窗口内 342 只成分股中 106 只无行业成员记录(G7 FAIL);
2. `namechange` 恰 10000 行且 `ann_date` 不早于 2020-12-22 → 当前 ST 成员 600837.SH、601989.SH 无更名记录(G8 FAIL)。

按用户契约(「若缺少必需字段,停止并报告,禁止静默使用 fixture、当前成分股或重新下载」),构建器已停止并产出机器可读缺口报告 `results/base_cache_build_gap_report.json`(退出码 2)。**PIT 指纹未产出**——不是构建器缺陷,而是数据源缺口。

**放行条件**: 下载 Agent 修复 B0 两个端点(见 §7.2)后,重跑
`scripts/build_base_cache_from_archive.py` → `pit-lake-build` → `pit-lake-verify` → strict acceptance。

---

## 1. 交付物完整性校验(任务 1)——全部通过

| 校验项 | 结果 |
|---|---|
| 主 ZIP 外部 SHA256 | ✅ 与交付方公布值一致 |
| ZIP 安全性(路径穿越/绝对路径扫描) | ✅ 无恶意条目 |
| 包内 `PACKAGE_SHA256SUMS` 逐文件复核 | ✅ 全部一致 |
| `catalog_snapshot.sqlite` `PRAGMA integrity_check` | ✅ ok |
| `samples.zip` 20/20 样本指纹 | ✅(SHA256SUMS 路径相对解压根目录,需带 `samples/` 前缀校验) |

## 2. 源码差异审查与集成(任务 2)

三方合并模型: base = 冻结快照 `5b4d054`,ours = HEAD `7aaa571`(a11),theirs = 交付 a9 + 热修。HEAD 的 archive 模块除 `batch.py`/CLI/`pit_*` 外均为交付版本的严格子集。

集成的交付文件(覆盖 HEAD,HEAD 与冻结快照对这些文件一致):
`src/ashare_quant/archive/{state,registry,provider,pipeline}.py`、`scripts/run_p0_batch.py`、`tests/test_archive.py`,以及 9 个交付方独有脚本(`freeze_b4_specs`、`migrate_aborted_b3`、`migrate_b2_legacy_cleanup`、`migrate_orphaned_context_drift`、`preflight_b4`、`probe_sharefloat_overflow`、`run_b2_repair`、`run_b3_repair`、`run_b4_repair`)。

手工合并:
- `src/ashare_quant/archive/batch.py`: 抽取 `batch_decision_gates(by_status, total)`,合并交付版的扩展终态逻辑(bisected、superseded_* 计入终态;success_rate 用 success+bisected)与 HEAD 的 `no_quarantined` 门。
- `src/ashare_quant/pit_lake.py`: 见 §3.2。

只读保护: 下载目录、活动状态库、全量 Raw/Bronze 全程未写入一字节;QF-integration 内使用的 `data_lake/catalog/archive.duckdb` 是交付快照的只读副本(chmod 444),连接一律 `mode=ro`。

## 3. PATH_REMAP 映射与 PIT 构建链(任务 3)

### 3.1 Raw 逐文件重校验(PATH_REMAP §2.4)

`scripts/reverify_raw_sha.py` 对全部 194,836 个 success 任务逐一做「映射后文件存在 + 内容 SHA256 == DB raw_sha256」:

- 194,792 直接吻合;**44 个 Phase A 历史任务**采用旧口径(`raw_sha256` 记录解压后 JSON 载荷的 SHA),解压后逐一验证 44/44 吻合,内容与行数完好;
- 缺失 0、真实不一致 0 → **verdict: pass**(`results/raw_sha_reverify_report.json`)。

### 3.2 pit_lake 修复(集成中发现并解决的三类封存滞后)

封存 manifest 是追加式历史记录;封存后的受控恢复/迁移不改写历史行,因此**终态判定以状态库当前真值为准**。为此 `pit_lake.py` 做了以下增强:

1. `_load_batch_evidence` 增加 catalog 真值对账(以 DB 重算 by_status),解决 B1 封存后 suspend_d 任务恢复为 success 导致的「非终态 quarantined:1」误报;
2. 新增 `_verify_seal_acceptance`: 对主 decision 非 pass 的批次(B3),验证 `SEAL_<prefix>.json` 为修复后权威验收(快照一致、manifest SHA 绑定、repair_decision=pass 且唯一、final_verification 全部终态);
3. `load()` 增加校验和回退: 52 个 B3 封存后瞬态恢复重写的 Bronze 不在封存 checksums 内,回退到「raw 文件 SHA256 == 任务 raw_sha256」绑定,计入 lineage 的 `checksum_fallback_tasks`;
4. `TERMINAL_STATUSES` 扩展至全部 8 种终态(研究排除仍由 `status=='success'` 选择承担,quarantined 保持 fail-closed);`raw_path` 列为可选(fixture 库无此列)。

### 3.3 strict pit-lake-build 的前置: archive → v4 基础缓存兼容层(新交付)

strict PIT 绑定 `data/cache/manifest.json`(v4 schema)的契约未做任何修改。新增:

- `src/ashare_quant/base_cache_bridge.py`(映射版本 `archive-to-v4-base-cache/1.0.0`): B1 Bronze 提供 daily/adj_factor/stk_limit/daily_basic,B2 Bronze 提供历史 index_weight,B0 提供 trade_cal/stock_basic/namechange/index_member_all/index_classify;成分宇宙**按生效日期的月末快照**构建,禁止当前成分回填;变换口径与下载器完全一致(vol×100、amount×1000、pre_close→prev_close、namechange 区间回填 is_st、涨跌停回退速率、div_proc=='实施');gzip mtime=0 确定性写入,同输入重跑同指纹;
- `scripts/build_base_cache_from_archive.py`: G1–G8 前置门禁,任一失败即停止并输出缺口报告(退出码 2);manifest 扩展 `archive_provenance` 段(catalog SHA、B1/B2 验收证据 SHA、任务集合/输入文件 SHA、映射版本、生成时间)。

**运行结果(当前数据): G1–G6 PASS,G7/G8 FAIL,按契约停止。**

| 门禁 | 结果 | 说明 |
|---|---|---|
| G1 catalog 钉定 | ✅ | 与交付快照 SHA 一致 |
| G2 必需端点 | ✅ | 12 个端点齐备 |
| G3 日历覆盖 | ✅ | 窗口 581 个交易日 |
| G4 行情日覆盖 | ✅ | daily/adj_factor/daily_basic/stk_limit 581/581,缺口 0 |
| G5 PIT 成分宇宙 | ✅ | 月末快照 16/16 月,成分异常 0 |
| G6 证券主表 | ✅ | 5,866 只,342 只窗口成员缺失 0 |
| G7 行业成员 | ❌ | 236/342 只;`index_member_all` 恰 3000 行疑似截断 |
| G8 更名完整性 | ❌ | 恰 10000 行;ST 成员 600837.SH/601989.SH 缺记录 |

## 4. 研究数据过滤审计(任务 4)——通过

`results/research_filter_audit.json`:

- 选择契约: 两条消费路径(`base_cache_bridge.load_success_tasks`、`pit_lake._select_strict_tasks`)均只取 `status='success'`;quarantined(164)、orphaned_context_drift(768)、aborted_prestart(2)、superseded_truncated_cap(529)、superseded_legacy_collision(189)、superseded_invalid_partition(32)——**合计 1,684 个任务天然排除**;
- 代际去重: 12 个必需端点 99,633 个 success 任务 → 94,738 个唯一逻辑分区,4,895 个被丢弃重复代(4,878 index_weight + 17 其他)全部记录进 provenance;
- **B4 share_float 15 股已知缺口规则保留**: B4 不属于 STRICT_BATCHES,基础缓存不消费 share_float;15 股缺口(12 确认 + 3 疑似)沿用 `DELIVERY_REPORT_P0.md` 口径,待 B4 修复后重验;
- 本构建未使用任何 fixture;fixture-mode 产物固定 `research_eligible=false` 的规则不受影响。

**状态卫生债务(移交下载 Agent)**: 4,895 个被丢弃重复代仍是 `status='success'`(封存前截断的遗留代),建议后续迁移为 `superseded_truncated_cap`,以免其他消费方误用;本构建靠确定性代际选择去重兜底。

## 5. 数据统计(任务 5)

| 指标 | 数值 |
|---|---|
| 证券总数(去重 ts_code) | 5,866(L 5,529 + D 337) |
| 行情窗口覆盖 | 2022-08-19 ~ 2025-01-10,581 个交易日,daily/adj_factor/daily_basic/stk_limit **581/581** |
| PIT 成分宇宙 | 16/16 月末快照完整,快照成分 300/300 |
| 基准指数 index_daily | 399300.SZ / 000300.SH / H00300.CSI 窗口均 581/581(全历史 2002-01-04 起) |
| B3 财务端点 | income/balancesheet/cashflow/fina_indicator/fina_audit/fina_mainbz/disclosure_date/forecast/express 全部 success(5,690–6,054 任务/端点);balancesheet_vip 另有 767 个 orphaned_context_drift 已排除 |
| B4 share_float | success 6,192 / bisected 1,589 / confirmed_empty 1,403 / superseded_truncated_cap 528 / quarantined 164(全部排除) |
| **PIT 指纹** | **未产出**(G7/G8 阻断,见总结论) |

已知缺口清单:
1. **B4 share_float 15 股**(交付方已披露,conditional_pass_known_gap);
2. **index_member_all 截断**(新发现,106/342 成员缺失)——阻断;
3. **namechange 截断**(新发现,2 只当前 ST 成员缺记录;历史区间回填 is_st 的完整性受影响)——阻断;
4. index_weight 1,224 个保留代恰 7,000 行(截断线):仅 1 个属于 399300.SZ(2024 年任务,已验证 24 个日期覆盖全年、月末快照 300/300 完整、月中临时快照被过滤,不影响 membership);其余 1,223 个为大成分指数,v4 基础缓存不消费,但**未来用其他指数建宇宙前必须先修复**;
5. block_trade 约 1,000 行疑似截断(未确证,基础缓存不消费)。

## 6. 测试与静态检查(任务 6)

- `ruff check .`: **All checks passed**;
- `python -m unittest discover tests`: **277 tests OK**(含新增 `tests/test_base_cache_bridge.py` 11 个:月末快照过滤/异常检测、代际去重、确定性重写、×100/×1000/回退速率/is_st 变换、缺 adj_factor fail-closed、div_proc 过滤、ts_code/symbol 撞列)。

## 7. 移交事项

### 7.1 移交云端 GPT(源码侧)

**`data.py::_historical_names` 潜在 bug(须在云端代码库修复)**: `pd.Timestamp.min.normalize()` 在 datetime64[ns] 下溢出回绕——`Timestamp.min` 是 1677-09-21 00:12:43,normalize 到当日 00:00 低于下界,回绕到 2262 年,导致 `between()` 恒为 False,**`start_date` 为空的「自始有效」更名记录被静默丢弃**(影响 is_st 历史回填)。兼容层已在 `base_cache_bridge.historical_names` 用 `pd.Timestamp.min/max`(不 normalize)规避并留注释,但生产下载器路径仍需云端修复。

### 7.2 移交下载 Agent(数据侧,修复后本验收可继续)

1. **index_member_all**: 登记 3,000 行截断 cap,按指数/分批重抓,补齐 106 只成员的 `is_new=N` 历史行;
2. **namechange**: 登记 10,000 行截断 cap,按 ann_date 子区间重抓 2020-12-22 之前历史,补齐 600837.SH/601989.SH 记录;
3. 状态卫生: 将 4,895 个截断遗留 success 代迁移为 `superseded_truncated_cap`;
4. 附带: 评估 1,223 个大成分指数 @7,000 行 index_weight 分区与 block_trade @~1,000 行的修复优先级(不阻断当前验收)。

## 8. 修改文件清单(本次验收新增/修改)

修改: `src/ashare_quant/pit_lake.py`、`src/ashare_quant/archive/batch.py`、`src/ashare_quant/archive/{state,registry,provider,pipeline}.py`、`scripts/run_p0_batch.py`、`tests/test_archive.py`

新增: `src/ashare_quant/base_cache_bridge.py`、`scripts/build_base_cache_from_archive.py`、`scripts/reverify_raw_sha.py`、`tests/test_base_cache_bridge.py`、9 个交付方迁移/修复脚本、`results/{raw_sha_reverify_report,research_filter_audit,acceptance_stats,base_cache_build_gap_report}.json`(生成物,不入 Git)

按指示: **不 push、不 tag**;本地提交一个验收 commit。
