# B0 截断修复交付验收报告(QF-integration 隔离环境,第二轮)

- 验收日期: 2026-07-24
- 验收方: 本地 Claude Code(K2.7)
- 交付方: 下载 Agent
- 交付物: `QF_data_acquisition_phase_a1_p0_2.0.0a9.zip`(重打包,166 成员,72.2MB)+ `QF-fixtures/catalog_snapshot.sqlite`(新快照)+ `B0_REPAIR_REPORT.md`
- 前置: 第一轮验收(`ACCEPTANCE_REPORT_P0.md`,commit `4853d8c`)因 index_member_all/namechange 静默截断 fail-closed,本轮为修复后复验

---

## 总结论

**research_eligible = true。可以进入真实 Alpha 研究。**

- archive→v4 基础缓存: **G1–G8 全过**,两次重复构建指纹完全一致(确定性);
- strict PIT 全链: `pit-lake-build` ✅ → `pit-lake-verify` ✅ → `pit-acceptance` **decision=pass, mode=strict, research_eligible=true**;
- 研究过滤契约不变(仅消费 `status='success'`),8 条 quarantined 冲突与 000817.SZ legacy 与研究期间 PIT 股票池**交集为 0**,无需继续 fail-closed。

关键指纹:

| 物 | SHA256 |
|---|---|
| 主 ZIP | `1ad9262512437f4848db9d6e8ac55a4b71f134ae2c50244f599ec2a7f22408f6` |
| 状态库快照 | `1d6acee0f1182253f966047b0adfcf9c6c09ddfa2036b7b172963fc33edd1d11` |
| v4 基础缓存数据指纹(两次构建一致) | `5dc7de50a83b50c9364cf26bfdc94940333ca782c5f6d2df3f3203bfcec054a1` |
| PIT 指纹(build=verify) | `44b18ed3a71caad49a63aee778886eeb18da578fc1596902c4fd1f70d91e1be7` |
| Alpha5 验收回执指纹 | `57c20defe1a0cf476c0cd7dff4309e1d81eafc4c4e2f6aadf85a97a2b43f6f01` |

---

## 1. 交付物完整性校验——全部通过

- 主 ZIP、catalog 快照外部 SHA256 与交付方公布值一致;`.sha256` 文件内容一致;
- ZIP 安全: 166 条目,无路径穿越/绝对路径;包内 `PACKAGE_SHA256SUMS` 逐文件复核 0 失败;
- `PRAGMA integrity_check` = ok;`samples.zip` 指纹仍与其 `.sha256` 一致(fixtures 未变);
- 新状态库 284,411 任务(+5,942): success 200,759(+5,923)、confirmed_empty 72,900(+6)、新状态 `superseded_incomplete_window` 10;
- **seals**: B0_reference 追加 2026-07-24 行(5,921 任务: success 5,886 + confirmed_empty 23 + superseded_* 12),B2_universe 追加两行(canonical 行 65,732 + 全量行 129,797),两批 `batch_decision.json` 均 **pass**;`b0_repair_acceptance.json` 9/9 门 pass;
- **supersedes 对账**: 旧 index_member_all @3000 → `superseded_truncated_cap`,64 继任叶子(61 success + 3 confirmed_empty);旧 namechange @10000 → `superseded_truncated_cap`;10 个窗口任务 → `superseded_incomplete_window`,继任为 5,866 个 ts_code 任务(manifest SHA `b859229f…`);两份迁移 jsonl(3 行 + 10 行)与状态库当前真值逐行一致;
- 旧文件全部保留: 738 个 superseded/quarantined 任务的 1,082 个 Raw/Bronze 文件逐一核查,缺失 0。

## 2. namechange 修复证据复核——通过

- **5,866/5,866 全解决**: 宇宙文件与 stock_basic(L 5,529 + D 337)集合完全相等;success 5,864(distinct ts_code 5,864)+ confirmed_empty 2,无未解决;
- **两个 confirmed_empty 的双重证据**:
  - `689009.SH`(九号公司,L): (a) 修复时逐 ts_code 直查返回空(2026-07-24T12:27,attempts=1,无错误,响应 SHA 已记录);(b) 旧截断文件中仅有的 2 行「九号公司-WD」(ann 20210416,reason 其他,完全重复)被分类 `legacy_observed` 保留兜底——上游已删改的变体,历史未丢;
  - `T600018.SH`(上港集箱（退）,D,delist 2006-10-20): (a) 逐 ts_code 直查返回空;(b) 旧截断文件同样 0 行——两种取数路径一致为空,且退市日远早于研究窗口;
- **600837.SH ≥7**: 并集恰 7 条(农垦商社/PT农商社/ST农商社/G都市/都市股份×2/海通证券)✅;旧截断文件 0 条(修复报告属实);
- **601989.SH ≥1**: 并集恰 1 条(中国重工 20091216)✅;吸收合并退市无更名史,阈值按实证 ≥1 合理;
- 旧截断文件对两只 ST 成员均为 0 条,新权威分区补齐,与交付报告一致。

## 3. B2 任务数口径变化: 115,434 → 65,732 无损证明

**口径**: 115,434 是 2026-07-20 旧 manifest 的**全批次任务域**(success 61,560 + confirmed_empty 53,872 + quarantined 2);65,732 是 2026-07-24 seal 的**canonical 域**(success 65,730 + superseded_truncated_cap 锚点 2)。同日另有全量 manifest 行 129,797 涵盖全部状态。

**逐任务对账(旧 115,434 → 新状态库)**:

| 迁移 | 数量 |
|---|---|
| success → success(原样保留) | 61,559 |
| success → superseded_truncated_cap(旧 index_member_all @3000) | 1 |
| quarantined → success / bisected(隔离解除) | 1 / 1 |
| confirmed_empty → confirmed_empty | 53,872 |
| **旧任务在新库缺失** | **0** |

- 旧 success 任务 `raw_sha256`/`row_count` **变化 0**(canonical 数据未动一字节);
- 65,730 = 61,559 旧保留 + 1 隔离解除 + 4,170 修复新增(61 个 index_member_all 叶子 + 4,109 个 B2 修复/补抓任务);
- 旧 confirmed_empty 53,872/53,872 全部进入新全量 manifest;
- Raw 重校验: **200,759/200,759 success 任务文件 SHA 全过**(44 个 Phase A 解压口径除外并已逐验),verdict=pass。

## 4. quarantined 冲突与 000817.SZ 的 PIT 池影响——零影响,解除 fail-closed

研究期间历史 PIT 股票池 = 399300.SZ 窗口(2023-10-31 ~ 2025-01-27)月末快照去重成员 **342 只**;加强口径为窗口内**全部**快照出现过的 362 只。

- **8 条 quarantined 冲突键**(000961.SZ/000971.SZ/002089.SZ/300038.SZ/300104.SZ/300325.SZ/603157.SH/688086.SH): 全部为已退市 ST 股,**均不在 342 池内,且未在窗口任何快照出现**;冲突键已从去重并集隔离(Raw 两版保留);
- **000817.SZ**(辽河油田): delist 2006-01-04 ≪ membership_start 2023-10-31,不在池内、不在任何窗口快照;唯一缺失键三种参数探针均取不回,旧行保留为 `legacy_observed`;
- 结论: 对研究期间历史 PIT 股票池**无任何影响**,无需继续 fail-closed。

## 5. 全链重跑(只读挂载全量 archive)

集成侧配合修复(不改数据,只改读法,与交付报告注意事项一致):

1. `build_base_cache_from_archive.py` G7/G8 改为对 index_member_all(61 叶子)/namechange(5,864 ts_code 任务)**全部 success 分区 concat**(旧版只读 tasks[0]);
2. G8 增加**退市标记豁免**(与交付验收门 9 同口径): 当前名带「退」但无 ST/退 更名记录的成员,须同时满足 delist_date 存在 + 全量更名记录在并集方可豁免并记录——本轮恰豁免 1 只(601989.SH,2 条并集记录);
3. `build_industry_membership` 增加 is_new=Y/N 双叶子「out_date 非空优先」经济键去重(与交付并集口径一致);
4. `pit_lake.TERMINAL_STATUSES` 纳入新终态 `superseded_incomplete_window`;
5. catalog 钉定更新为新快照 SHA(旧值留档注释);修复 bars 拼装遍历全历史日期的越界 bug(首轮未跑到该路径)。

结果:

| 步骤 | 结果 |
|---|---|
| G1–G8 | **全 PASS**(G7 行业成员 342/342、疑似截断 0;G8 更名 16,397 行覆盖 5,864 只、缺记录 0、豁免 1、疑似截断 0) |
| v4 缓存构建 ×2 | 342 只历史成分,bars 197,835 行;两次指纹**完全一致** `5dc7de50…` |
| `MarketDataBundle.from_cache(strict=True)` 回读 | 通过 |
| `pit-lake-build` | strict,财报 113,106 行 + 估值 167,533 行,研究资格 True |
| `pit-lake-verify` | 28,656 任务源文件复核全过,指纹与 build 一致 |
| `pit-acceptance` | **decision=pass,mode=strict,research_eligible=true** |

## 6. 质量门

- `ruff check .`: All checks passed;
- `python -m unittest discover tests`: **277 tests OK**。

## 7. 遗留事项(不阻断)

- `legacy_observed` 记录(2,902 键 namechange + 1 键 index_member_all)仅存于旧 Raw(保留未删),如需全量历史视角可联合旧文件,但不得把旧任务重标 success;
- index_weight 1,223 个大成分指数 @7,000 行保留代(不进入 399300.SZ 宇宙构建)与 block_trade @~1,000 疑似截断: 未来扩展其他指数宇宙/事件数据前需修复,优先级移交下载 Agent;
- 4,895 个截断遗留 success 代的状态卫生迁移建议(首轮报告 §4)本轮部分推进(B0 两端点已迁移),其余维持代际选择去重兜底。

按指示: **不 push、不 tag**;本地提交验收 commit。
