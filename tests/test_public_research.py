from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ashare_quant.public_research as public_module
from ashare_quant.alpha import (
    ALPHA_MODEL_VERSION,
    FORMATION_RETURN_DAYS,
    QUALITY_MOMENTUM_V1_5_WEIGHTS,
)
from ashare_quant.public_research import (
    MiniRacer,
    PublicDownloadConfig,
    PublicStrategyConfig,
    _SinaDecoder,
    _SinaPayload,
    _build_features,
    _build_sina_bars,
    _calculate_metrics,
    _eastmoney_secid,
    _rebalance_public_positions,
    download_public_history,
    load_membership,
    members_at,
    public_strategy_variants,
    seal_public_cache,
    verify_public_cache,
)


class PublicResearchTests(unittest.TestCase):
    def test_sina_decoder_uses_one_runtime_on_its_owner_thread(self) -> None:
        events: dict[str, object] = {"created": 0, "calls": []}

        class FakeRuntime:
            def __enter__(self) -> FakeRuntime:
                events["entered_thread"] = threading.get_ident()
                return self

            def __exit__(self, *args: object) -> None:
                events["exited_thread"] = threading.get_ident()

            def eval(self, script: str) -> None:
                self.assert_owner()
                events["script"] = script

            def call(self, function: str, encoded: str) -> object:
                self.assert_owner()
                self_calls = events["calls"]
                assert isinstance(self_calls, list)
                self_calls.append(threading.get_ident())
                if encoded == "bad":
                    raise ValueError("bad payload")
                return json.loads(encoded)

            def assert_owner(self) -> None:
                self.asserted_owner = events["owner_thread"]
                if threading.get_ident() != self.asserted_owner:
                    raise AssertionError("runtime used outside decoder thread")

        def factory() -> FakeRuntime:
            events["created"] = int(events["created"]) + 1
            events["owner_thread"] = threading.get_ident()
            return FakeRuntime()

        with _SinaDecoder(
            queue_size=3,
            runtime_factory=factory,
            decode_script="function d(value) { return JSON.parse(value); }",
        ) as decoder:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                decoded = list(
                    executor.map(
                        decoder.decode,
                        [json.dumps({"value": value}) for value in range(60)],
                    )
                )
            with self.assertRaisesRegex(ValueError, "bad payload"):
                decoder.decode("bad")
            self.assertEqual(decoder.decode('{"value": 61}'), {"value": 61})

        self.assertEqual(events["created"], 1)
        self.assertEqual([item["value"] for item in decoded], list(range(60)))
        self.assertEqual(events["entered_thread"], events["owner_thread"])
        self.assertEqual(events["exited_thread"], events["owner_thread"])
        self.assertEqual(set(events["calls"]), {events["owner_thread"]})

    def test_sina_decoder_calls_close_for_legacy_compatible_runtime(self) -> None:
        events: dict[str, object] = {}

        class ClosableRuntime:
            def eval(self, script: str) -> None:
                events["owner_thread"] = threading.get_ident()

            def call(self, function: str, encoded: str) -> object:
                self.assert_owner()
                return encoded

            def close(self) -> None:
                self.assert_owner()
                events["closed"] = True

            def assert_owner(self) -> None:
                if threading.get_ident() != events["owner_thread"]:
                    raise AssertionError("lifecycle left decoder thread")

        with _SinaDecoder(
            queue_size=1,
            runtime_factory=ClosableRuntime,
            decode_script="function d(value) { return value; }",
        ) as decoder:
            self.assertEqual(decoder.decode("ok"), "ok")
        self.assertTrue(events["closed"])

    def test_sina_payload_transform_preserves_hfq_semantics(self) -> None:
        payload = _SinaPayload(
            symbol="SH600000",
            start_date="2020-01-01",
            end_date="2020-01-31",
            encoded_history="unused",
            factor_payload={"data": [{"d": "2020-01-01", "f": "2.0"}]},
        )
        decoded = [
            {
                "date": "2020-01-02",
                "open": "10",
                "close": "11",
                "high": "12",
                "low": "9",
                "volume": "1000",
                "amount": "10000",
            },
            {
                "date": "2020-01-03",
                "open": "11",
                "close": "12",
                "high": "13",
                "low": "10",
                "volume": "1100",
                "amount": "12000",
            },
        ]
        frame = _build_sina_bars(payload, decoded)
        self.assertEqual(frame["symbol"].unique().tolist(), ["SH600000"])
        self.assertEqual(frame["close"].tolist(), [22.0, 24.0])
        self.assertEqual(frame["open"].tolist(), [20.0, 22.0])
        self.assertTrue(pd.isna(frame["pct_change"].iloc[0]))
        self.assertAlmostEqual(float(frame["pct_change"].iloc[1]), 100 / 11)

    @unittest.skipIf(MiniRacer is None, "需要可选 MiniRacer 运行时")
    def test_sina_download_six_workers_share_one_actual_runtime(self) -> None:
        original_runtime = MiniRacer
        assert original_runtime is not None
        runtime_creations: list[int] = []
        request_threads: set[int] = set()
        request_lock = threading.Lock()

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

            def raise_for_status(self) -> None:
                return None

        def fake_get(url: str, **kwargs: object) -> FakeResponse:
            del kwargs
            with request_lock:
                request_threads.add(threading.get_ident())
            time.sleep(0.01)
            if url.endswith("/hfq.js"):
                return FakeResponse(
                    "var hfq={'data':[{'d':'2020-01-01','f':'1.0'}]}\n"
                )
            return FakeResponse('var history="token";')

        def runtime_factory() -> object:
            runtime_creations.append(threading.get_ident())
            return original_runtime()

        decode_script = """
        function d(value) {
          return [
            {date: '2020-01-02', open: 10, close: 11, high: 12, low: 9, volume: 1000, amount: 10000},
            {date: '2020-01-03', open: 11, close: 12, high: 13, low: 10, volume: 1100, amount: 12000},
            {date: '2020-01-06', open: 12, close: 13, high: 14, low: 11, volume: 1200, amount: 14000}
          ];
        }
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership = root / "membership.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": f"SH60000{number}",
                        "name": f"测试{number}",
                        "opt-in": "2020-01-01",
                        "opt-out": "",
                    }
                    for number in range(6)
                ]
            ).to_csv(membership, index=False)
            with (
                mock.patch.object(public_module.requests, "get", side_effect=fake_get),
                mock.patch.object(public_module, "MiniRacer", side_effect=runtime_factory),
                mock.patch.object(public_module, "hk_js_decode", decode_script),
            ):
                manifest = download_public_history(
                    membership,
                    root / "cache",
                    PublicDownloadConfig(
                        start_date="2020-01-01",
                        end_date="2020-01-31",
                        workers=6,
                        retries=1,
                        request_pause_seconds=0.0,
                    ),
                )

        self.assertEqual(len(runtime_creations), 1)
        self.assertGreaterEqual(len(request_threads), 2)
        self.assertNotIn(runtime_creations[0], request_threads)
        self.assertEqual(manifest["available_count"], 8)
        self.assertEqual(
            manifest["decoder_architecture"],
            "single_dedicated_thread_bounded_queue",
        )

    @unittest.skipIf(MiniRacer is None, "需要可选 MiniRacer 运行时")
    def test_mini_racer_subprocess_stress(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests" / "stress_sina_decoder.py"),
                "--workers",
                "6",
                "--tasks",
                "120",
                "--repeats",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["workers"], 6)
        self.assertEqual(summary["decoded"], 240)

    def test_membership_uses_exclusive_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "SH600000",
                        "name": "甲",
                        "opt-in": "2020-01-01",
                        "opt-out": "2020-06-15",
                    },
                    {
                        "symbol": "SZ000001",
                        "name": "乙",
                        "opt-in": "2020-06-15",
                        "opt-out": "",
                    },
                ]
            ).to_csv(path, index=False)
            membership = load_membership(path)
            on_boundary = set(members_at(membership, "2020-06-15")["symbol"])
            self.assertEqual(on_boundary, {"SZ000001"})

    def test_market_code_mapping(self) -> None:
        self.assertEqual(_eastmoney_secid("SH600000"), "1.600000")
        self.assertEqual(_eastmoney_secid("SZ000001"), "0.000001")
        with self.assertRaises(ValueError):
            _eastmoney_secid("US.AAPL")

    def test_features_do_not_use_future_prices(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=300)
        close = pd.Series(np.linspace(10.0, 20.0, len(dates)))
        bars = pd.DataFrame(
            {
                "date": dates,
                "symbol": "SH600000",
                "name": "测试",
                "open": close,
                "close": close,
                "amount": 100_000_000.0,
            }
        )
        signal_date = dates[270]
        original = _build_features(bars, 60).set_index("date").loc[signal_date]
        changed = bars.copy()
        changed.loc[changed["date"] > signal_date, ["open", "close"]] *= 50.0
        recomputed = _build_features(changed, 60).set_index("date").loc[signal_date]
        for column in [
            "mom_12_1",
            "mom_6_1",
            "information_discreteness",
            "fip_momentum",
            "trend",
            "volatility",
            "downside_volatility",
            "drawdown_quality",
        ]:
            self.assertAlmostEqual(float(original[column]), float(recomputed[column]), places=12)

    def test_fip_factor_rewards_gradual_winner_over_jump_winner(self) -> None:
        target_gross_return = 1.30
        prefix = [0.0] * 27
        recent_month = [0.0] * 21
        gradual_formation = [
            target_gross_return ** (1.0 / FORMATION_RETURN_DAYS) - 1.0
        ] * FORMATION_RETURN_DAYS
        negative_return = -0.001
        jump_return = target_gross_return / (
            (1.0 + negative_return) ** (FORMATION_RETURN_DAYS - 1)
        ) - 1.0
        discrete_formation = [negative_return] * (FORMATION_RETURN_DAYS - 1) + [
            jump_return
        ]

        def price_path(formation: list[float]) -> np.ndarray:
            returns = np.asarray(prefix + formation + recent_month, dtype=float)
            return 100.0 * np.r_[1.0, np.cumprod(1.0 + returns)]

        dates = pd.bdate_range("2020-01-01", periods=280)
        bars = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": symbol,
                        "name": symbol,
                        "open": values,
                        "close": values,
                        "amount": 100_000_000.0,
                    }
                )
                for symbol, values in {
                    "GRADUAL": price_path(gradual_formation),
                    "DISCRETE": price_path(discrete_formation),
                }.items()
            ],
            ignore_index=True,
        )
        latest = (
            _build_features(bars, 60)
            .sort_values("date")
            .groupby("symbol", sort=False)
            .tail(1)
            .set_index("symbol")
        )

        self.assertAlmostEqual(
            float(latest.at["GRADUAL", "mom_12_1"]),
            float(latest.at["DISCRETE", "mom_12_1"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(latest.at["GRADUAL", "information_discreteness"]),
            -1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(latest.at["DISCRETE", "information_discreteness"]),
            (FORMATION_RETURN_DAYS - 2) / FORMATION_RETURN_DAYS,
            places=12,
        )
        self.assertLess(
            float(latest.at["GRADUAL", "information_discreteness"]),
            float(latest.at["DISCRETE", "information_discreteness"]),
        )
        self.assertGreater(
            float(latest.at["GRADUAL", "fip_momentum"]),
            float(latest.at["DISCRETE", "fip_momentum"]),
        )

    def test_v1_5_public_candidate_uses_frozen_weights(self) -> None:
        candidate = next(
            variant
            for variant in public_strategy_variants()
            if variant.name == ALPHA_MODEL_VERSION
        )
        self.assertEqual(candidate.top_n, 15)
        self.assertAlmostEqual(
            candidate.weight_fip_momentum,
            QUALITY_MOMENTUM_V1_5_WEIGHTS["fip_momentum"],
        )
        self.assertAlmostEqual(
            candidate.weight_low_downside_volatility,
            QUALITY_MOMENTUM_V1_5_WEIGHTS["low_downside_vol"],
        )
        self.assertAlmostEqual(
            candidate.weight_drawdown_quality,
            QUALITY_MOMENTUM_V1_5_WEIGHTS["drawdown_quality"],
        )

    def test_metrics_cagr_and_drawdown(self) -> None:
        curve = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
                "nav": [1.0, 0.8, 1.21],
            }
        )
        metrics = _calculate_metrics(curve)
        self.assertAlmostEqual(float(metrics["cagr"]), 0.10, places=2)
        self.assertAlmostEqual(float(metrics["max_drawdown"]), -0.20, places=12)

    def test_public_config_rejects_infeasible_stock_cap(self) -> None:
        with self.assertRaises(ValueError):
            PublicStrategyConfig(top_n=10, max_stock_weight=0.05).validate()

    def test_public_config_rejects_negative_factor_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "非负有限数"):
            PublicStrategyConfig(weight_fip_momentum=-0.01).validate()

    def test_suspended_position_is_never_scaled_to_fund_buys(self) -> None:
        config = PublicStrategyConfig()
        outcome = _rebalance_public_positions(
            current_positions={"LOCKED": 0.60, "OLD": 0.35},
            requested_target={"NEW": 0.95},
            tradable_symbols={"OLD", "NEW"},
            nav_open=1.0,
            when=pd.Timestamp("2024-01-02"),
            config=config,
        )
        self.assertAlmostEqual(outcome.positions["LOCKED"], 0.60, places=12)
        self.assertGreaterEqual(outcome.cash, 0.0)
        self.assertLess(outcome.executable_scale, 1.0)
        self.assertAlmostEqual(outcome.locked_value, 0.60, places=12)
        self.assertAlmostEqual(
            outcome.cash + sum(outcome.positions.values()) + outcome.cost,
            1.0,
            places=12,
        )

    def test_public_cache_fingerprint_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership_path = root / "csi300.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "SH600000",
                        "name": "测试",
                        "opt-in": "2020-01-01",
                        "opt-out": "",
                    },
                    {
                        "symbol": "SH600001",
                        "name": "缺失行情",
                        "opt-in": "2020-01-01",
                        "opt-out": "",
                    },
                ]
            ).to_csv(membership_path, index=False)
            cache = root / "cache"
            bars = cache / "bars"
            bars.mkdir(parents=True)
            for symbol in ["SH600000", "SH000300", "SH510300"]:
                (bars / f"{symbol}.csv.gz").write_bytes(f"sealed-{symbol}".encode())

            manifest = seal_public_cache(membership_path, cache)
            self.assertEqual(manifest["available_count"], 3)
            verified = verify_public_cache(membership_path, cache)
            self.assertTrue(verified["verification"]["verified"])

            (bars / "SH600000.csv.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "完整性校验失败"):
                verify_public_cache(membership_path, cache)
            with self.assertRaisesRegex(ValueError, "已经封存"):
                seal_public_cache(membership_path, cache)

            (bars / "SH600000.csv.gz").write_bytes(b"sealed-SH600000")
            (bars / "SH600001.csv.gz").write_bytes(b"new-unsealed-file")
            with self.assertRaisesRegex(ValueError, "文件集合不一致"):
                verify_public_cache(membership_path, cache)
            resumable = verify_public_cache(
                membership_path,
                cache,
                _allow_unsealed_files=True,
            )
            self.assertEqual(
                resumable["verification"]["unsealed_paths"],
                ["bars/SH600001.csv.gz"],
            )

    def test_public_config_requires_separate_regime_and_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须分离"):
            PublicStrategyConfig(
                regime_symbol="SH000300",
                performance_benchmark_symbol="SH000300",
            ).validate()


if __name__ == "__main__":
    unittest.main()
