"""Endpoint registry and inventory management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .pipeline import EndpointSpec


@dataclass
class InventoryEndpoint:
    api_name: str
    priority: str
    dataset: str
    primary_key: list[str]
    primary_split: str | None
    fallback_split: str | None
    all_fields: bool = True
    fields: str = ""
    params: dict[str, Any] | None = None
    enabled: bool = True
    # --- archival metadata ---
    batch: str = ""                      # B0_reference / B1_market / ... / P1 batches
    split_unit: str = "snapshot"         # snapshot|trade_date|month|quarter|year|symbol|symbol_year|symbol_quarter|index_year|index_month
    required_params: list[str] = field(default_factory=list)
    supported_splits: list[str] = field(default_factory=list)
    earliest_date: str | None = None     # configured earliest (YYYYMMDD / YYYYMM / YYYY)
    pit_rule: str = ""                   # PIT visibility rule description
    probe_params: dict[str, Any] | None = None   # explicit probe request params
    probe_note: str = ""

    def to_spec(self) -> EndpointSpec:
        return EndpointSpec(
            api_name=self.api_name,
            dataset=self.dataset,
            priority=self.priority,
            primary_key=self.primary_key,
            primary_split=self.primary_split,
            fallback_split=self.fallback_split,
            all_fields=self.all_fields,
            fields=self.fields or "",
            params_template=dict(self.params or {}),
        )


class EndpointInventory:
    """Load and query the endpoint inventory."""

    def __init__(self, endpoints: list[InventoryEndpoint]) -> None:
        self.endpoints = {ep.api_name: ep for ep in endpoints}

    @classmethod
    def from_yaml(cls, path: Path) -> "EndpointInventory":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        endpoints = []
        for item in data.get("endpoints", []):
            endpoints.append(
                InventoryEndpoint(
                    api_name=item["api_name"],
                    priority=item.get("priority", "P2"),
                    dataset=item.get("dataset", item["api_name"]),
                    primary_key=item.get("primary_key", []),
                    primary_split=item.get("primary_split"),
                    fallback_split=item.get("fallback_split"),
                    all_fields=bool(item.get("all_fields", True)),
                    fields=item.get("fields", ""),
                    params=item.get("params"),
                    enabled=bool(item.get("enabled", True)),
                    batch=item.get("batch", ""),
                    split_unit=item.get("split_unit", "snapshot"),
                    required_params=item.get("required_params", []),
                    supported_splits=item.get("supported_splits", []),
                    earliest_date=item.get("earliest_date"),
                    pit_rule=item.get("pit_rule", ""),
                    probe_params=item.get("probe_params"),
                    probe_note=item.get("probe_note", ""),
                )
            )
        return cls(endpoints)

    @classmethod
    def from_list(cls, endpoints: list[dict[str, Any]]) -> "EndpointInventory":
        return cls([InventoryEndpoint(**ep) for ep in endpoints])

    def to_yaml(self, path: Path, probe_results: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe_results = probe_results or {}
        entries = []
        for ep in self.endpoints.values():
            item = {k: v for k, v in asdict(ep).items() if v not in (None, "", [], {})}
            result = probe_results.get(ep.api_name)
            if result:
                item["probe"] = result
            entries.append(item)
        payload = {"schema_version": 1, "endpoints": entries}
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def list_by_priority(self, priorities: list[str] | None = None) -> list[InventoryEndpoint]:
        eps = list(self.endpoints.values())
        if priorities:
            eps = [ep for ep in eps if ep.priority in priorities]
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        eps.sort(key=lambda ep: (order.get(ep.priority, 99), ep.api_name))
        return eps

    def list_by_batch(self, batch_id: str) -> list[InventoryEndpoint]:
        return [ep for ep in self.list_by_priority() if ep.batch == batch_id and ep.enabled]


# ---------------------------------------------------------------------------
# Full P0/P1 endpoint inventory per DATA_ACQUISITION_V2_DESIGN.md and
# TUSHARE_DATA_ARCHIVE_PLAN.md.  Static archival metadata only; probe results
# are merged in at report time.
# ---------------------------------------------------------------------------

_MARKET_PIT = "收盘后生成，下一交易日可用"
_FIN_PIT = "ann_date/f_ann_date 公告后下一交易日可见；保留全部修订"
_EVENT_PIT = "以首次公告日为可见时间"

DEFAULT_ENDPOINTS: list[dict[str, Any]] = [
    # ======================= P0 · 基础身份 (B0_reference) =======================
    {
        "api_name": "trade_cal", "priority": "P0", "dataset": "calendar",
        "primary_key": ["exchange", "cal_date"], "primary_split": None, "fallback_split": None,
        "batch": "B0_reference", "split_unit": "snapshot",
        "params": {"exchange": "SSE", "is_open": ""},
        "required_params": [], "supported_splits": ["exchange", "is_open"],
        "earliest_date": "19901219",
        "pit_rule": "日历快照，无 PIT 限制",
        "probe_params": {"exchange": "SSE", "start_date": "20250101", "end_date": "20250131"},
    },
    {
        "api_name": "stock_basic", "priority": "P0", "dataset": "security_master",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "B0_reference", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["list_status", "exchange"],
        "pit_rule": "下载日快照；身份区间由 list_date/delist_date 重建",
        "probe_params": {"list_status": "L"},
    },
    {
        "api_name": "stock_company", "priority": "P0", "dataset": "security_master",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B0_reference", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["exchange", "ts_code"],
        "pit_rule": "下载日快照",
        "probe_params": {"exchange": "SSE"},
    },
    {
        "api_name": "namechange", "priority": "P0", "dataset": "security_master",
        "primary_key": ["ts_code", "start_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B0_reference", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["ts_code", "ann_date"],
        "pit_rule": "历史改名时段表，用于 ST/退市名称识别，禁止当前名称回填",
        "probe_params": {"ts_code": "000001.SZ"},
    },
    {
        "api_name": "new_share", "priority": "P0", "dataset": "security_master",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "B0_reference", "split_unit": "year",
        "required_params": [], "supported_splits": ["start_date/end_date"],
        "earliest_date": "19901219",
        "pit_rule": "IPO 事件表，以发行公告为可见时间",
        "probe_params": {"start_date": "20240101", "end_date": "20241231"},
    },
    # ======================= P0 · 行情与可交易状态 (B1_market) =======================
    {
        "api_name": "daily", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code", "start_date/end_date"],
        "earliest_date": "19901219", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "adj_factor", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "19901219", "pit_rule": "复权因子变动与分红送转事件核对",
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "daily_basic", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20050104", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "stk_limit", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20200217", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "suspend_d", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date", "suspend_type"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "suspend_type"],
        "earliest_date": "20180101", "pit_rule": "停复牌事件，公告后可见",
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "dividend", "priority": "P0", "dataset": "corporate_events",
        "primary_key": ["ts_code", "end_date", "div_proc", "imp_ann_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "ann_date", "record_date", "ex_date"],
        "earliest_date": "19901219", "pit_rule": "预案/实施/取消全部保留；可见时间按研究问题选择",
        "probe_params": {"ts_code": "000001.SZ"},
    },
    {
        "api_name": "block_trade", "priority": "P0", "dataset": "market_daily",
        "primary_key": ["ts_code", "trade_date", "buyer", "seller"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "B1_market", "split_unit": "month",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "start_date/end_date"],
        "earliest_date": "20050101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    # ======================= P0 · 指数与行业 (B2_universe) =======================
    {
        "api_name": "index_basic", "priority": "P0", "dataset": "index_metadata",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "B2_universe", "split_unit": "index_basic_segments",
        "required_params": [], "supported_splits": ["market", "publisher", "category"],
        "pit_rule": "下载日快照",
        "probe_params": {"market": "SSE"},
    },
    {
        "api_name": "index_daily", "priority": "P0", "dataset": "index_daily",
        "primary_key": ["ts_code", "trade_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B2_universe", "split_unit": "index_year_main",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "trade_date", "start_date/end_date"],
        "earliest_date": "20020104", "pit_rule": _MARKET_PIT,
        "probe_params": {"ts_code": "000300.SH", "start_date": "20250101", "end_date": "20250131"},
    },
    {
        "api_name": "index_weight", "priority": "P0", "dataset": "index_membership",
        "primary_key": ["index_code", "trade_date", "con_code"], "primary_split": "trade_date", "fallback_split": None,
        "batch": "B2_universe", "split_unit": "index_year_main",
        "required_params": [], "supported_splits": ["index_code", "trade_date", "start_date/end_date"],
        "earliest_date": "20111101", "pit_rule": "成分权重按 trade_date 有效，禁止当前成分回填历史",
        "probe_params": {"index_code": "399300.SZ", "start_date": "20250101", "end_date": "20250131"},
    },
    {
        "api_name": "index_classify", "priority": "P0", "dataset": "industry_classification",
        "primary_key": ["index_code"], "primary_split": None, "fallback_split": None,
        "batch": "B2_universe", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["level", "src"],
        "pit_rule": "行业分类快照；成员区间由 index_member_all 提供",
        "probe_params": {"level": "L1", "src": "SW2021"},
    },
    {
        "api_name": "index_member_all", "priority": "P0", "dataset": "industry_classification",
        "primary_key": ["l1_code", "ts_code", "in_date"], "primary_split": None, "fallback_split": None,
        "batch": "B2_universe", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["l1_code", "is_new"],
        "pit_rule": "必须保留 in_date/out_date 有效区间",
        "probe_params": {"l1_code": "801780.SI", "is_new": "Y"},
    },
    # ======================= P0 · 财务与 PIT (B3_financial) =======================
    {
        "api_name": "income_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "report_type", "update_flag"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code", "report_type"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "balancesheet_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "report_type", "update_flag"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code", "report_type"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "cashflow_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "report_type", "update_flag"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code", "report_type"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "fina_indicator_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "update_flag"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "forecast_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "ann_date", "type"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code"],
        "earliest_date": "20081231", "pit_rule": "以首次公开时间为可见时间，修订另建版本",
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "express_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "ann_date"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["period"], "supported_splits": ["period", "ts_code"],
        "earliest_date": "20120331", "pit_rule": "以首次公开时间为可见时间，修订另建版本",
        "probe_params": {"period": "20240930"},
    },
    {
        "api_name": "fina_audit", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date"], "primary_split": "period", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "year",
        "required_params": [], "supported_splits": ["ts_code", "period"],
        "earliest_date": "19951231", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ", "period": "20231231"},
    },
    {
        "api_name": "fina_mainbz_vip", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "bz_item", "type"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "period", "type"],
        "earliest_date": "20071231", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "period+type 单季度恰好 10000 行（截断上限）；按 ts_code 取全历史（千行级）规避",
    },
    {
        "api_name": "disclosure_date", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "period"],
        "earliest_date": "20101231", "pit_rule": "预约披露日，公告后可见",
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "period 单年恰好 6000 行（截断上限）；按 ts_code 取全部预约记录（百余行）规避",
    },
    # 非 VIP 财务接口：用于 VIP 抽样交叉核验
    {
        "api_name": "income", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "ann_date", "update_flag"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period", "start_date/end_date"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ", "end_date": "20241231"},
        "enabled": False,   # VIP 覆盖后仅按需启用
    },
    {
        "api_name": "balancesheet", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "ann_date", "update_flag"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ", "end_date": "20241231"},
        "enabled": False,
    },
    {
        "api_name": "cashflow", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "ann_date", "update_flag"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ", "end_date": "20241231"},
        "enabled": False,
    },
    {
        "api_name": "fina_indicator", "priority": "P0", "dataset": "financial_pit",
        "primary_key": ["ts_code", "end_date", "update_flag"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B3_financial", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period"],
        "earliest_date": "19950331", "pit_rule": _FIN_PIT,
        "probe_params": {"ts_code": "000001.SZ", "end_date": "20241231"},
        "enabled": False,
    },
    # ======================= P1 · 股东/治理事件 (B4_events) =======================
    {
        "api_name": "repurchase", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "ann_date", "end_date"], "primary_split": "ann_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "month",
        "required_params": [], "supported_splits": ["ann_date", "start_date/end_date", "ts_code"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ann_date": "20240115"},
        "probe_note": "年度/半年区间恰好 2000 行（截断上限）；按月窗口切分",
    },
    {
        "api_name": "share_float", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "float_date"], "primary_split": "float_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol_chunk",
        "required_params": [], "supported_splits": ["ts_code", "ann_date", "float_date"],
        "earliest_date": "20050101", "pit_rule": "解禁事件，公告后可见",
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "支持逗号代码列表分块；日期区间查询 HTTP 500",
    },
    {
        "api_name": "pledge_stat", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "end_date"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "end_date"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "季度横截面覆盖不全（部分统计日为空）；按 ts_code 取全历史",
    },
    {
        "api_name": "pledge_detail", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "start_date", "pledgor"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000002.SZ"},
        "probe_note": "000001.SZ 无质押明细属真实空；逗号代码列表不支持",
    },
    {
        "api_name": "stk_holdernumber", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "end_date", "ann_date"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "end_date", "ann_date"],
        "earliest_date": "20160101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "季度横截面恰好 5500 行（截断上限）；按 ts_code 取全历史",
    },
    {
        "api_name": "stk_holdertrade", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "ann_date", "holder_name", "in_de"], "primary_split": "ann_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol",
        "required_params": [], "supported_splits": ["ts_code", "ann_date"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "按 ts_code 取全历史",
    },
    {
        "api_name": "top10_holders", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "end_date", "holder_name"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol_chunk",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000001.SZ", "period": "20240930"},
        "probe_note": "period 横截面 HTTP 500；支持逗号代码列表分块",
    },
    {
        "api_name": "top10_floatholders", "priority": "P1", "dataset": "corporate_events",
        "primary_key": ["ts_code", "end_date", "holder_name"], "primary_split": "end_date", "fallback_split": "ts_code",
        "batch": "B4_events", "split_unit": "symbol",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period"],
        "earliest_date": "20150101", "pit_rule": _EVENT_PIT,
        "probe_params": {"ts_code": "000001.SZ", "period": "20231231"},
        "probe_note": "proxy 对部分季度（如 2024Q2/Q3）覆盖不全，归档后需在覆盖率报告单独列出",
    },
    # ======================= P1 · 资金/杠杆/异常交易 =======================
    {
        "api_name": "moneyflow", "priority": "P1", "dataset": "moneyflow",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_moneyflow", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20100101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "margin", "priority": "P1", "dataset": "margin",
        "primary_key": ["trade_date", "exchange_id"], "primary_split": "trade_date", "fallback_split": None,
        "batch": "P1_margin", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "exchange_id"],
        "earliest_date": "20100331", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "margin_detail", "priority": "P1", "dataset": "margin",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_margin", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20100331", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "margin_secs", "priority": "P1", "dataset": "margin",
        "primary_key": ["trade_date", "ts_code"], "primary_split": "trade_date", "fallback_split": None,
        "batch": "P1_margin", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date"],
        "pit_rule": "每日标的快照",
        "probe_params": {"trade_date": "20250715"},
        "probe_note": "无参数快照恰好 6000 行（截断上限）；按 trade_date 逐日归档",
    },
    {
        "api_name": "top_list", "priority": "P1", "dataset": "abnormal_trading",
        "primary_key": ["ts_code", "trade_date", "reason"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_abnormal", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20150101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "top_inst", "priority": "P1", "dataset": "abnormal_trading",
        "primary_key": ["ts_code", "trade_date", "exalter"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_abnormal", "split_unit": "trade_date",
        "required_params": ["trade_date"], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20150101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "limit_list_d", "priority": "P1", "dataset": "abnormal_trading",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_abnormal", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "limit_type"],
        "earliest_date": "20200101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "limit_list_ths", "priority": "P1", "dataset": "abnormal_trading",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_abnormal", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "limit_type"],
        "earliest_date": "20200101", "pit_rule": "第三方整理，非 PIT；下载日快照口径需标注",
        "probe_params": {"trade_date": "20250102"},
    },
    # ======================= P1 · 分析师/机构预期 =======================
    {
        "api_name": "report_rc", "priority": "P1", "dataset": "analyst_expectation",
        "primary_key": ["ts_code", "report_date", "org_name", "author_name"], "primary_split": "report_date", "fallback_split": "ts_code",
        "batch": "P1_analyst", "split_unit": "symbol_year",
        "required_params": ["ts_code"], "supported_splits": ["ts_code"],
        "earliest_date": "20180101", "pit_rule": "保留报告发布日期与机构/分析师身份，不只留最新预测",
        "probe_params": {"ts_code": "000001.SZ"},
        "probe_note": "日期区间查询返回 HTTP 500；仅支持 ts_code 查询（可叠加 start/end 需实测）",
    },
    {
        "api_name": "stk_surv", "priority": "P1", "dataset": "analyst_expectation",
        "primary_key": ["ts_code", "surv_date", "fund_visitors"], "primary_split": "surv_date", "fallback_split": "ts_code",
        "batch": "P1_analyst", "split_unit": "month",
        "required_params": [], "supported_splits": ["ts_code", "surv_date", "start_date/end_date"],
        "earliest_date": "20180101", "pit_rule": "以调研公开时间为可见时间",
        "probe_params": {"start_date": "20250101", "end_date": "20250131"},
    },
    {
        "api_name": "broker_recommend", "priority": "P1", "dataset": "analyst_expectation",
        "primary_key": ["ts_code", "month", "broker"], "primary_split": "month", "fallback_split": "ts_code",
        "batch": "P1_analyst", "split_unit": "month",
        "required_params": ["month"], "supported_splits": ["month", "ts_code"],
        "earliest_date": "202001", "pit_rule": "以券商推荐发布时间为可见时间",
        "probe_params": {"month": "202501"},
    },
    # ======================= P1 · 筹码 =======================
    {
        "api_name": "cyq_perf", "priority": "P1", "dataset": "chips",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_chips", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "start_date/end_date"],
        "earliest_date": "20180101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "cyq_chips", "priority": "P1", "dataset": "chips",
        "primary_key": ["ts_code", "trade_date", "price"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_chips", "split_unit": "trade_date",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "trade_date", "start_date/end_date"],
        "earliest_date": "20180101", "pit_rule": _MARKET_PIT,
        "probe_params": {"ts_code": "000001.SZ", "trade_date": "20250102"},
        "probe_note": "必须带 ts_code；单日全市场返回空。每股每日百余行，体量最大，先容量评估再决定全量",
    },
    # ======================= P1 · 技术因子 =======================
    {
        "api_name": "stk_factor_pro", "priority": "P1", "dataset": "technical_factors",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_factor", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "start_date/end_date"],
        "earliest_date": "20180101", "pit_rule": "可由 OHLCV 重算；仅作对照",
        "probe_params": {"trade_date": "20250102"},
    },
    # ======================= P1 · ETF/基金 =======================
    {
        "api_name": "fund_basic", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "P1_funds", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["market", "fund_type", "status"],
        "pit_rule": "下载日快照",
        "probe_params": {"market": "E", "status": "L"},
    },
    {
        "api_name": "fund_daily", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_funds", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "start_date/end_date"],
        "earliest_date": "20100101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "fund_adj", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_funds", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code"],
        "earliest_date": "20100101", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102"},
    },
    {
        "api_name": "fund_share", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code", "trade_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "P1_funds", "split_unit": "symbol_year",
        "required_params": [], "supported_splits": ["ts_code", "trade_date", "start_date/end_date"],
        "earliest_date": "20150101", "pit_rule": _MARKET_PIT,
        "probe_params": {"ts_code": "510300.SH"},
        "probe_note": "单基金全历史恰好返回 2000 行（疑似截断上限），需按 ts_code+年度窗口切分",
    },
    {
        "api_name": "fund_nav", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code", "nav_date"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "P1_funds", "split_unit": "year",
        "required_params": [], "supported_splits": ["ts_code", "market"],
        "earliest_date": "20100101", "pit_rule": _MARKET_PIT,
        "probe_params": {"ts_code": "510300.SH"},
    },
    {
        "api_name": "fund_portfolio", "priority": "P1", "dataset": "funds",
        "primary_key": ["ts_code", "end_date", "symbol"], "primary_split": None, "fallback_split": "ts_code",
        "batch": "P1_funds", "split_unit": "year",
        "required_params": ["ts_code"], "supported_splits": ["ts_code", "period", "start_date/end_date"],
        "earliest_date": "20150101", "pit_rule": "持仓披露按报告期可见",
        "probe_params": {"ts_code": "510300.SH"},
    },
    # ======================= P1 · 宏观 =======================
    {
        "api_name": "shibor", "priority": "P1", "dataset": "macro",
        "primary_key": ["date"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "year",
        "required_params": [], "supported_splits": ["start_date/end_date"],
        "earliest_date": "20061008", "pit_rule": "公布日可见",
        "probe_params": {"start_date": "20250101", "end_date": "20250110"},
    },
    {
        "api_name": "shibor_lpr", "priority": "P1", "dataset": "macro",
        "primary_key": ["date"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_date/end_date"],
        "earliest_date": "20190820", "pit_rule": "公布日可见",
        "probe_params": {"start_date": "20250101", "end_date": "20250131"},
    },
    {
        "api_name": "cn_gdp", "priority": "P1", "dataset": "macro",
        "primary_key": ["quarter"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_q/end_q"],
        "pit_rule": "公布日可见",
        "probe_params": {},
    },
    {
        "api_name": "cn_cpi", "priority": "P1", "dataset": "macro",
        "primary_key": ["month"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_m/end_m"],
        "pit_rule": "公布日可见",
        "probe_params": {"start_m": "202501", "end_m": "202505"},
    },
    {
        "api_name": "cn_ppi", "priority": "P1", "dataset": "macro",
        "primary_key": ["month"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_m/end_m"],
        "pit_rule": "公布日可见",
        "probe_params": {"start_m": "202501", "end_m": "202505"},
    },
    {
        "api_name": "cn_pmi", "priority": "P1", "dataset": "macro",
        "primary_key": ["month"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_m/end_m"],
        "pit_rule": "公布日可见",
        "probe_params": {"start_m": "202501", "end_m": "202505"},
    },
    {
        "api_name": "cn_m", "priority": "P1", "dataset": "macro",
        "primary_key": ["month"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_m/end_m"],
        "pit_rule": "公布日可见",
        "probe_params": {"start_m": "202501", "end_m": "202505"},
    },
    {
        "api_name": "sf_month", "priority": "P1", "dataset": "macro",
        "primary_key": ["month"], "primary_split": None, "fallback_split": None,
        "batch": "P1_macro", "split_unit": "snapshot",
        "required_params": [], "supported_splits": ["start_m/end_m"],
        "pit_rule": "公布日可见",
        "probe_params": {"start_m": "202501", "end_m": "202505"},
    },
    # ======================= P1 · 股指衍生品 =======================
    {
        "api_name": "fut_basic", "priority": "P1", "dataset": "index_derivatives",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "P1_derivatives", "split_unit": "snapshot",
        "required_params": ["exchange"], "supported_splits": ["exchange", "fut_type"],
        "pit_rule": "合约信息快照",
        "probe_params": {"exchange": "CFFEX", "fut_type": "1"},
    },
    {
        "api_name": "fut_daily", "priority": "P1", "dataset": "index_derivatives",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_derivatives", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "exchange", "start_date/end_date"],
        "earliest_date": "20100416", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102", "exchange": "CFFEX"},
    },
    {
        "api_name": "opt_basic", "priority": "P1", "dataset": "index_derivatives",
        "primary_key": ["ts_code"], "primary_split": None, "fallback_split": None,
        "batch": "P1_derivatives", "split_unit": "snapshot",
        "required_params": ["exchange"], "supported_splits": ["exchange", "call_put"],
        "pit_rule": "合约信息快照",
        "probe_params": {"exchange": "SSE", "call_put": "C"},
    },
    {
        "api_name": "opt_daily", "priority": "P1", "dataset": "index_derivatives",
        "primary_key": ["ts_code", "trade_date"], "primary_split": "trade_date", "fallback_split": "ts_code",
        "batch": "P1_derivatives", "split_unit": "trade_date",
        "required_params": [], "supported_splits": ["trade_date", "ts_code", "exchange", "start_date/end_date"],
        "earliest_date": "20150209", "pit_rule": _MARKET_PIT,
        "probe_params": {"trade_date": "20250102", "exchange": "SSE"},
    },
]

# Legacy aliases kept for sample.py compatibility (Phase A uses these names).
PHASE_A_SAMPLE_ENDPOINTS = DEFAULT_ENDPOINTS


def default_inventory() -> EndpointInventory:
    return EndpointInventory.from_list(DEFAULT_ENDPOINTS)


# Batch execution order for P0 full archival.
P0_BATCH_ORDER = ["B0_reference", "B1_market", "B2_universe", "B3_financial", "B4_events"]
