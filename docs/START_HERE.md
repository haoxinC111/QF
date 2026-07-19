# QF 数据获取交接包

版本：`handoff-v1`  
生成日期：`2026-07-16`

## 交接目标

实现一个可替换数据源、可断点续传、可证明完整性、可长期离线复现的 A 股数据归档
模块，并由归档数据生成 QF V2 所需的行情与 PIT 财务缓存。

## 阅读顺序

1. `DATA_ACQUISITION_V2_DESIGN.md`
2. `TUSHARE_DATA_ARCHIVE_PLAN.md`
3. `TUSHARE_DATA_REQUIREMENTS.md`
4. `config.archive.example.yaml`

## 本地 Agent 第一阶段任务

第一阶段只实现和执行批次 A，不要直接开始全量下载：

1. 实现 `ArchiveProvider` 接口与直接 HTTP 兼容适配器。
2. 实现 endpoint registry、任务状态数据库和全局令牌桶。
3. 保存每个请求的原始 `json.zst`，并无损转换为 Parquet。
4. 实现 schema 指纹、SHA256、断点续传和疑似截断自动拆分。
5. 对 P0/P1 接口做权限探针，生成 `permission_report.json`。
6. 下载5个交易日、50只股票、4个财务季度的小样本。
7. 输出容量、吞吐、错误率、字段和跨来源抽样核验报告。
8. 批次 A 通过并由用户确认后，才能启用批次 B。

## 不可违反的约束

- 不得把官方 `TUSHARE_TOKEN` 发送到第三方网关。
- 第三方 Token 只能从 `QF_ARCHIVE_API_TOKEN` 读取。
- 第三方地址只能从 `QF_ARCHIVE_API_URL` 读取，不硬编码进仓库。
- `authorization_confirmed=false` 或 `local_archival_allowed=false` 时禁止全量下载。
- 不使用修改 Tushare SDK 私有变量的方式做正式归档。
- 不采用文档中的未公开实时爬虫方案。
- Token 不得进入配置、命令行、日志、异常、manifest 或 Git。
- Raw 数据只追加不覆盖；数据文件不得提交到 Git。
- HTTP 200 不代表完整；达到行数上限必须拆分重取。
- 所有财务修订版本都要保留，不能在原始层只留最新记录。

## 第一阶段交付物

```text
permission_report.json
endpoint_inventory.yaml
download_manifest.jsonl
schema_registry/
checksums.sha256
coverage_report.md
capacity_estimate.md
cross_source_validation.md
```

同时提交：

- 实现代码与单元测试；
- 失败恢复测试；
- 行数截断测试；
- Token 脱敏测试；
- 配置严格校验测试；
- 固定版本号和 Git commit。

## 成功标准

批次 A 成功不等于数据源已经可信。只有授权确认、接口探针、跨来源抽样、截断检测和
容量评估同时通过，才能进入全量归档。
