"""archive → v4 基础行情缓存兼容层的单元测试(合成数据,不触网不碰归档)。"""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ashare_quant.base_cache_bridge import (
    BronzeTask,
    build_bars_for_symbol,
    build_corporate_actions,
    build_securities,
    fallback_limit_rate,
    membership_from_index_weight,
    month_end_dates,
    select_generations,
    write_deterministic_csv_gz,
)


def _task(api: str, params: dict, snapshot: str, row_count: int = 10, tid: str = "") -> BronzeTask:
    return BronzeTask(
        task_id=tid or f"{api}-{snapshot}-{len(params)}",
        api_name=api,
        params=params,
        row_count=row_count,
        bronze_path=f"/x/data_lake/bronze/p/{api}/{api}_x_{snapshot}.parquet",
        raw_sha256="0" * 64,
        snapshot=snapshot,
    )


class MonthEndTests(unittest.TestCase):
    def test_month_end_dates_picks_last_trading_day(self) -> None:
        calendar = pd.DatetimeIndex(
            ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-27", "2024-02-29"]
        )
        ends = month_end_dates(calendar)
        self.assertEqual(
            ends, {pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")}
        )


class MembershipTests(unittest.TestCase):
    def _weights(self) -> pd.DataFrame:
        rows = []
        # 月末快照(完整 300)与月中临时快照(完整)、一个残缺月末快照
        for date, count in (
            ("20240131", 300),
            ("20240102", 300),  # 月中快照:应被月末过滤剔除
            ("20240229", 300),
            ("20240329", 100),  # 残缺月末:应触发异常
        ):
            for i in range(count):
                rows.append(
                    {
                        "index_code": "399300.SZ",
                        "trade_date": date,
                        "con_code": f"{i:06d}.SZ",
                        "weight": 0.3,
                    }
                )
        return pd.DataFrame(rows)

    def test_month_end_filter_and_partial_snapshot_anomaly(self) -> None:
        calendar = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-29"])
        membership, anomalies = membership_from_index_weight(
            [self._weights()],
            "399300.SZ",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-03-31"),
            calendar,
        )
        dates = set(membership["date"])
        self.assertNotIn(pd.Timestamp("2024-01-02"), dates)  # 月中快照被剔除
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["constituents"], 100)

    def test_empty_frames_yield_empty(self) -> None:
        membership, anomalies = membership_from_index_weight(
            [], "399300.SZ", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31"),
            pd.DatetimeIndex(["2024-01-31"]),
        )
        self.assertTrue(membership.empty)
        self.assertEqual(anomalies, [])


class GenerationSelectionTests(unittest.TestCase):
    def test_repair_generation_wins_and_drop_is_recorded(self) -> None:
        params = {"index_code": "399300.SZ", "start_date": "20240101", "end_date": "20241231"}
        old = _task("index_weight", params, "p0_B2_universe_20260717_022655", row_count=7000)
        new = _task("index_weight", params, "p0_B2_repair_20260721_013031", row_count=6900)
        result = select_generations([old, new])
        chosen = result.tasks[("index_weight", '{"end_date": "20241231", "index_code": "399300.SZ", "start_date": "20240101"}')]
        self.assertEqual(chosen.snapshot, "p0_B2_repair_20260721_013031")
        self.assertEqual(len(result.dropped_duplicates), 1)
        self.assertEqual(result.dropped_duplicates[0]["row_count"], 7000)

    def test_phase_a_loses_to_batch_snapshot(self) -> None:
        params = {"trade_date": "20240102"}
        phase = _task("daily", params, "phase_a_20260716_135101")
        main = _task("daily", params, "p0_B1_market_20260716_182403")
        result = select_generations([phase, main])
        chosen = next(iter(result.tasks.values()))
        self.assertEqual(chosen.snapshot, "p0_B1_market_20260716_182403")


class DeterminismTests(unittest.TestCase):
    def test_rewrite_produces_identical_bytes(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv.gz"
            write_deterministic_csv_gz(frame, path)
            first = path.read_bytes()
            write_deterministic_csv_gz(frame, path)
            second = path.read_bytes()
            with gzip.open(path, "rt") as fh:
                header = fh.readline().strip()
        self.assertEqual(first, second)
        self.assertEqual(header, "a,b")


class BarBuildTests(unittest.TestCase):
    def test_unit_conversions_and_limit_fallback(self) -> None:
        daily = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240102"],
                "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
                "pre_close": [10.0], "vol": [100.0], "amount": [200.0],
            }
        )
        adj = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"], "adj_factor": [1.5]})
        basic = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"], "total_mv": [1000.0], "circ_mv": [800.0]})
        limit = pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        names = pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["ST测试"], "start_date": [None], "end_date": [None]})
        frame = build_bars_for_symbol("000001.SZ", daily, adj, limit, basic, names)
        row = frame.iloc[0]
        self.assertEqual(row["volume"], 10000.0)  # 手 → 股
        self.assertEqual(row["amount"], 200000.0)  # 千元 → 元
        self.assertTrue(bool(row["is_st"]))
        # ST 主板回退 ±5%
        self.assertAlmostEqual(row["up_limit"], 10.5)
        self.assertAlmostEqual(row["down_limit"], 9.5)

    def test_missing_adj_factor_fails_closed(self) -> None:
        daily = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 2,
                "trade_date": ["20240102", "20240103"],
                "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
                "close": [1.0, 1.0], "pre_close": [1.0, 1.0],
                "vol": [1.0, 1.0], "amount": [1.0, 1.0],
            }
        )
        adj = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"], "adj_factor": [1.0]})
        basic = pd.DataFrame({"ts_code": ["000001.SZ"] * 2, "trade_date": ["20240102", "20240103"], "total_mv": [1.0, 1.0], "circ_mv": [1.0, 1.0]})
        limit = pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        with self.assertRaises(ValueError):
            build_bars_for_symbol("000001.SZ", daily, adj, limit, basic, pd.DataFrame())


class FallbackRateTests(unittest.TestCase):
    def test_rates_match_downloader(self) -> None:
        self.assertEqual(fallback_limit_rate("430001.BJ", False), 0.30)
        self.assertEqual(fallback_limit_rate("830001.BJ", False), 0.30)
        self.assertEqual(fallback_limit_rate("300001.SZ", False), 0.20)
        self.assertEqual(fallback_limit_rate("688001.SH", False), 0.20)
        self.assertEqual(fallback_limit_rate("600001.SH", True), 0.05)
        self.assertEqual(fallback_limit_rate("600001.SH", False), 0.10)


class ActionsTests(unittest.TestCase):
    def test_dividend_filter_and_rename(self) -> None:
        dividend = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "div_proc": ["实施", "预案", "实施"],
                "record_date": ["20240601", "20240601", "20240701"],
                "ex_date": ["20240605", "20240605", "20240705"],
                "pay_date": ["20240610", "20240610", "20240710"],
                "div_listdate": ["20240605", "20240605", "20240705"],
                "cash_div": [0.5, 0.5, 0.3],
                "stk_div": [0.0, 0.0, 0.1],
            }
        )
        result = build_corporate_actions(
            [dividend], {"000001.SZ", "000002.SZ"},
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
        )
        self.assertEqual(len(result), 2)  # 预案被剔除
        self.assertEqual(set(result["symbol"]), {"000001.SZ", "000002.SZ"})
        self.assertIn("cash_dividend", result.columns)


class SecuritiesTests(unittest.TestCase):
    def test_ts_code_symbol_collision_handled(self) -> None:
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],  # Tushare 原生 symbol 是 6 位纯数字
                "name": ["平安银行"],
                "industry": ["银行"],
                "market": ["主板"],
                "list_status": ["L"],
                "list_date": ["19910403"],
                "delist_date": [None],
            }
        )
        result = build_securities([frame])
        self.assertEqual(len(result), 1)
        # v4 契约的 symbol 必须取带后缀的 ts_code,不能用 6 位原生 symbol
        self.assertEqual(result.iloc[0]["symbol"], "000001.SZ")


if __name__ == "__main__":
    unittest.main()
