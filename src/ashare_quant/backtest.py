from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import MarketDataBundle
from .factors import MultiFactorStrategy, SignalPlan


@dataclass
class Position:
    units: float
    last_buy_date: pd.Timestamp | None = None


@dataclass
class PendingRebalance:
    plan: SignalPlan
    first_execution_date: pd.Timestamp
    target_values: dict[str, float]
    remaining: set[str]
    liquidity: dict[str, float]
    reserve_cash: float
    attempts: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    selections: pd.DataFrame
    final_positions: dict[str, float]
    warnings: list[str] = field(default_factory=list)


class Backtester:
    def __init__(self, bundle: MarketDataBundle, config: AppConfig) -> None:
        self.bundle = bundle.prepare()
        self.config = config
        self.strategy = MultiFactorStrategy(self.bundle, config.strategy)
        self.cash = float(config.backtest.initial_cash)
        self.positions: dict[str, Position] = {}
        self.pending: PendingRebalance | None = None
        self.trade_records: list[dict[str, Any]] = []
        self.order_records: list[dict[str, Any]] = []
        self.selection_records: list[pd.DataFrame] = []
        self.equity_records: list[dict[str, Any]] = []
        self.total_fees = 0.0
        self.total_notional = 0.0
        self.warnings: list[str] = []

        bars = self.bundle.bars.copy()
        bars["adj_open"] = bars["open"] * bars["adj_factor"]
        bars["adj_close"] = bars["close"] * bars["adj_factor"]
        self.bars = bars.set_index(["date", "symbol"]).sort_index()
        all_calendar = self.bundle.calendar
        self.close_marks = (
            bars.pivot(index="date", columns="symbol", values="adj_close")
            .reindex(all_calendar)
            .ffill()
        )
        self.benchmark_close = (
            self.bundle.benchmark.set_index("date")["close"].reindex(all_calendar).ffill()
        )

    def run(self) -> BacktestResult:
        start = pd.Timestamp(self.config.backtest.start_date)
        end = pd.Timestamp(self.config.backtest.end_date)
        calendar = self.bundle.calendar[
            (self.bundle.calendar >= start) & (self.bundle.calendar <= end)
        ]
        if len(calendar) < 2:
            raise ValueError("回测区间内交易日不足")

        schedule = self._build_schedule(calendar)
        benchmark_base = float(self.benchmark_close.loc[calendar].dropna().iloc[0])

        for date in calendar:
            if date in schedule:
                if self.pending is not None:
                    self._cancel_pending(date, "new_rebalance_superseded")
                self._activate(schedule[date], date)
            if self.pending is not None:
                self._attempt_pending(date)
            self._record_close(date, benchmark_base)

        if self.pending is not None:
            self._cancel_pending(calendar[-1], "backtest_ended")

        equity = pd.DataFrame(self.equity_records)
        trades = pd.DataFrame(self.trade_records)
        orders = pd.DataFrame(self.order_records)
        selections = (
            pd.concat(self.selection_records, ignore_index=True)
            if self.selection_records
            else pd.DataFrame()
        )
        final_positions = {
            symbol: self._position_value(symbol, calendar[-1], use_open=False)
            for symbol in sorted(self.positions)
        }
        if orders.empty:
            self.warnings.append("回测区间没有产生订单；请检查历史长度、股票池和过滤参数。")
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            orders=orders,
            selections=selections,
            final_positions=final_positions,
            warnings=self.warnings,
        )

    def _build_schedule(self, calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, SignalPlan]:
        full_calendar = self.bundle.calendar
        month_ends = (
            pd.Series(calendar, index=calendar)
            .groupby(calendar.to_period("M"))
            .max()
            .tolist()
        )
        schedule: dict[pd.Timestamp, SignalPlan] = {}
        for signal_date in month_ends:
            location = full_calendar.searchsorted(signal_date, side="right")
            if location >= len(full_calendar):
                continue
            execution_date = full_calendar[location]
            if execution_date > calendar[-1]:
                continue
            plan = self.strategy.generate(signal_date)
            schedule[execution_date] = plan
            if not plan.selection.empty:
                selection = plan.selection.copy()
                selection["planned_execution_date"] = execution_date
                self.selection_records.append(selection)
        return schedule

    def _activate(self, plan: SignalPlan, execution_date: pd.Timestamp) -> None:
        opening_equity = self._portfolio_equity(execution_date, use_open=True)
        target_values = {
            symbol: float(weight) * opening_equity for symbol, weight in plan.weights.items()
        }
        remaining = set(self.positions).union(target_values)
        liquidity: dict[str, float] = {}
        for symbol in remaining:
            value = plan.liquidity.get(symbol, np.nan)
            if not np.isfinite(value) or value <= 0:
                value = self.strategy.trailing_amount(symbol, plan.signal_date)
            liquidity[symbol] = float(value) if np.isfinite(value) else np.inf
        self.pending = PendingRebalance(
            plan=plan,
            first_execution_date=execution_date,
            target_values=target_values,
            remaining=remaining,
            liquidity=liquidity,
            reserve_cash=opening_equity * self.config.execution.cash_buffer,
        )

    def _attempt_pending(self, date: pd.Timestamp) -> None:
        pending = self.pending
        if pending is None:
            return
        pending.attempts += 1
        symbols = sorted(pending.remaining)
        sell_symbols = [
            symbol
            for symbol in symbols
            if self._position_value(symbol, date, use_open=True)
            > pending.target_values.get(symbol, 0.0) + 1e-8
        ]
        for symbol in sell_symbols:
            self._execute_sell(symbol, date, pending)

        buy_symbols = [
            symbol
            for symbol in sorted(pending.remaining)
            if pending.target_values.get(symbol, 0.0)
            > self._position_value(symbol, date, use_open=True) + 1e-8
        ]
        for symbol in buy_symbols:
            self._execute_buy(symbol, date, pending)

        for symbol in list(pending.remaining):
            target = pending.target_values.get(symbol, 0.0)
            current = self._position_value(symbol, date, use_open=True)
            if self._within_one_lot(symbol, date, abs(target - current)):
                pending.remaining.discard(symbol)

        if not pending.remaining:
            self.pending = None
        elif pending.attempts >= self.config.execution.rebalance_retry_days:
            self._cancel_pending(date, "retry_window_expired")

    def _execute_sell(
        self, symbol: str, date: pd.Timestamp, pending: PendingRebalance
    ) -> None:
        position = self.positions.get(symbol)
        if position is None or position.units <= 1e-12:
            pending.remaining.discard(symbol)
            return
        if position.last_buy_date is not None and position.last_buy_date >= date:
            self._record_order(pending, date, symbol, "SELL", "REJECTED", "T+1", 0, 0, 0, 0)
            return
        bar = self._bar(symbol, date)
        if bar is None or float(bar["volume"]) <= 0:
            self._record_order(
                pending, date, symbol, "SELL", "REJECTED", "suspended_or_no_bar", 0, 0, 0, 0
            )
            return
        if float(bar["open"]) <= float(bar["down_limit"]) + 0.005:
            self._record_order(
                pending, date, symbol, "SELL", "REJECTED", "open_at_down_limit", 0, 0, 0, 0
            )
            return

        target_value = pending.target_values.get(symbol, 0.0)
        current_value = position.units * float(bar["adj_open"])
        desired_value = max(0.0, current_value - target_value)
        factor = float(bar["adj_factor"])
        slippage = self.config.execution.slippage_bps / 10_000.0
        price = max(float(bar["down_limit"]), float(bar["open"]) * (1.0 - slippage))
        desired_shares = desired_value / max(price, 1e-12)
        participation_cap = self._participation_cap(pending, symbol)
        cap_shares = participation_cap / max(price, 1e-12)
        full_exit = target_value <= 1e-8 and desired_value <= participation_cap + 1e-8
        if full_exit:
            shares = position.units * factor
        else:
            shares = self._floor_lot(min(desired_shares, cap_shares))
        if shares <= 0:
            reason = "below_lot" if desired_shares < self.config.execution.lot_size else "participation_cap"
            self._record_order(
                pending, date, symbol, "SELL", "REJECTED", reason, 0, price, 0, 0
            )
            if reason == "below_lot":
                pending.remaining.discard(symbol)
            return

        units_sold = min(position.units, shares / factor)
        shares = units_sold * factor
        notional = shares * price
        fees = self._fees(notional, "SELL")
        position.units -= units_sold
        if position.units <= 1e-10:
            del self.positions[symbol]
        self.cash += notional - fees
        self.total_fees += fees
        self.total_notional += notional
        remaining_value = self._position_value(symbol, date, use_open=True)
        partial = remaining_value > target_value + self._lot_value(symbol, date)
        status = "PARTIAL" if partial else "FILLED"
        self._record_order(
            pending, date, symbol, "SELL", status, "", shares, price, notional, fees
        )
        self.trade_records.append(self.order_records[-1].copy())
        if not partial:
            pending.remaining.discard(symbol)

    def _execute_buy(
        self, symbol: str, date: pd.Timestamp, pending: PendingRebalance
    ) -> None:
        bar = self._bar(symbol, date)
        if bar is None or float(bar["volume"]) <= 0:
            self._record_order(
                pending, date, symbol, "BUY", "REJECTED", "suspended_or_no_bar", 0, 0, 0, 0
            )
            return
        if float(bar["open"]) >= float(bar["up_limit"]) - 0.005:
            self._record_order(
                pending, date, symbol, "BUY", "REJECTED", "open_at_up_limit", 0, 0, 0, 0
            )
            return

        target_value = pending.target_values.get(symbol, 0.0)
        current_value = self._position_value(symbol, date, use_open=True)
        desired_value = max(0.0, target_value - current_value)
        slippage = self.config.execution.slippage_bps / 10_000.0
        price = min(float(bar["up_limit"]), float(bar["open"]) * (1.0 + slippage))
        participation_cap = self._participation_cap(pending, symbol)
        shares = self._floor_lot(min(desired_value, participation_cap) / max(price, 1e-12))
        if shares <= 0:
            self._record_order(
                pending, date, symbol, "BUY", "REJECTED", "below_lot", 0, price, 0, 0
            )
            pending.remaining.discard(symbol)
            return

        available_cash = max(0.0, self.cash - pending.reserve_cash)
        shares = min(shares, self._affordable_shares(available_cash, price, shares))
        if shares <= 0:
            self._record_order(
                pending, date, symbol, "BUY", "REJECTED", "insufficient_cash", 0, price, 0, 0
            )
            return

        notional = shares * price
        fees = self._fees(notional, "BUY")
        total_cost = notional + fees
        if total_cost > available_cash + 1e-8:
            self._record_order(
                pending, date, symbol, "BUY", "REJECTED", "insufficient_cash", 0, price, 0, 0
            )
            return
        factor = float(bar["adj_factor"])
        units = shares / factor
        position = self.positions.setdefault(symbol, Position(units=0.0))
        position.units += units
        position.last_buy_date = date
        self.cash -= total_cost
        self.total_fees += fees
        self.total_notional += notional
        new_value = self._position_value(symbol, date, use_open=True)
        partial = new_value + self._lot_value(symbol, date) < target_value
        status = "PARTIAL" if partial else "FILLED"
        self._record_order(
            pending, date, symbol, "BUY", status, "", shares, price, notional, fees
        )
        self.trade_records.append(self.order_records[-1].copy())
        if not partial:
            pending.remaining.discard(symbol)

    def _record_order(
        self,
        pending: PendingRebalance,
        date: pd.Timestamp,
        symbol: str,
        side: str,
        status: str,
        reason: str,
        shares: float,
        price: float,
        notional: float,
        fees: float,
    ) -> None:
        commission, stamp, transfer = self._fee_components(notional, side) if notional > 0 else (0, 0, 0)
        self.order_records.append(
            {
                "signal_date": pending.plan.signal_date,
                "date": date,
                "attempt": pending.attempts,
                "symbol": symbol,
                "side": side,
                "status": status,
                "reason": reason,
                "shares": float(shares),
                "price": float(price),
                "notional": float(notional),
                "commission": float(commission),
                "stamp_duty": float(stamp),
                "transfer_fee": float(transfer),
                "fees": float(fees),
            }
        )

    def _cancel_pending(self, date: pd.Timestamp, reason: str) -> None:
        pending = self.pending
        if pending is None:
            return
        for symbol in sorted(pending.remaining):
            target = pending.target_values.get(symbol, 0.0)
            current = self._position_value(symbol, date, use_open=False)
            side = "BUY" if target > current else "SELL"
            self._record_order(
                pending, date, symbol, side, "CANCELLED", reason, 0, 0, 0, 0
            )
        self.pending = None

    def _record_close(self, date: pd.Timestamp, benchmark_base: float) -> None:
        holdings = sum(
            self._position_value(symbol, date, use_open=False) for symbol in self.positions
        )
        equity = self.cash + holdings
        benchmark_value = float(self.benchmark_close.loc[date])
        self.equity_records.append(
            {
                "date": date,
                "nav": equity,
                "cash": self.cash,
                "holdings_value": holdings,
                "gross_exposure": holdings / equity if equity > 0 else np.nan,
                "positions": len(self.positions),
                "benchmark_nav": self.config.backtest.initial_cash
                * benchmark_value
                / benchmark_base,
                "cumulative_fees": self.total_fees,
                "cumulative_notional": self.total_notional,
            }
        )

    def _bar(self, symbol: str, date: pd.Timestamp) -> pd.Series | None:
        try:
            row = self.bars.loc[(date, symbol)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row
        except KeyError:
            return None

    def _position_value(self, symbol: str, date: pd.Timestamp, use_open: bool) -> float:
        position = self.positions.get(symbol)
        if position is None:
            return 0.0
        if use_open:
            bar = self._bar(symbol, date)
            if bar is not None:
                return position.units * float(bar["adj_open"])
        try:
            mark = self.close_marks.at[date, symbol]
        except KeyError:
            return 0.0
        return position.units * float(mark) if pd.notna(mark) else 0.0

    def _portfolio_equity(self, date: pd.Timestamp, use_open: bool) -> float:
        return self.cash + sum(
            self._position_value(symbol, date, use_open=use_open) for symbol in self.positions
        )

    def _participation_cap(self, pending: PendingRebalance, symbol: str) -> float:
        rate = self.config.execution.max_participation_of_20d_amount
        if rate <= 0:
            return np.inf
        amount = pending.liquidity.get(symbol, np.inf)
        if not np.isfinite(amount) or amount <= 0:
            return np.inf
        return float(amount) * rate

    def _floor_lot(self, shares: float) -> float:
        lot = self.config.execution.lot_size
        return float(np.floor(max(0.0, shares) / lot + 1e-12) * lot)

    def _lot_value(self, symbol: str, date: pd.Timestamp) -> float:
        bar = self._bar(symbol, date)
        if bar is None:
            return np.inf
        return self.config.execution.lot_size * float(bar["open"])

    def _within_one_lot(self, symbol: str, date: pd.Timestamp, gap_value: float) -> bool:
        lot_value = self._lot_value(symbol, date)
        return np.isfinite(lot_value) and gap_value < lot_value

    def _affordable_shares(self, cash: float, price: float, maximum: float) -> float:
        shares = self._floor_lot(min(maximum, cash / max(price, 1e-12)))
        lot = self.config.execution.lot_size
        while shares > 0:
            notional = shares * price
            if notional + self._fees(notional, "BUY") <= cash + 1e-8:
                return shares
            shares -= lot
        return 0.0

    def _fee_components(self, notional: float, side: str) -> tuple[float, float, float]:
        if notional <= 0:
            return 0.0, 0.0, 0.0
        commission = max(
            self.config.execution.minimum_commission,
            notional * self.config.execution.commission_rate,
        )
        stamp = notional * self.config.execution.stamp_duty_sell if side == "SELL" else 0.0
        transfer = notional * self.config.execution.transfer_fee_rate
        return commission, stamp, transfer

    def _fees(self, notional: float, side: str) -> float:
        return float(sum(self._fee_components(notional, side)))
