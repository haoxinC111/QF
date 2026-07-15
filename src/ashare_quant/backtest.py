from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import MarketDataBundle
from .execution import DEFAULT_EXECUTION_MODEL, market_impact_bps
from .factors import MultiFactorStrategy, SignalPlan


@dataclass
class Lot:
    shares: float
    acquired_date: pd.Timestamp
    sellable_date: pd.Timestamp
    source: str = "BUY"


@dataclass
class Position:
    lots: list[Lot] = field(default_factory=list)

    @property
    def total_shares(self) -> float:
        return float(sum(lot.shares for lot in self.lots))

    def sellable_shares(self, date: pd.Timestamp) -> float:
        return float(sum(lot.shares for lot in self.lots if lot.sellable_date <= date))

    def add(self, lot: Lot) -> None:
        if lot.shares > 1e-12:
            self.lots.append(lot)

    def remove_sellable(self, shares: float, date: pd.Timestamp) -> float:
        remaining = float(shares)
        removed = 0.0
        new_lots: list[Lot] = []
        for lot in sorted(self.lots, key=lambda item: (item.sellable_date, item.acquired_date)):
            if remaining > 1e-12 and lot.sellable_date <= date:
                take = min(lot.shares, remaining)
                lot.shares -= take
                remaining -= take
                removed += take
            if lot.shares > 1e-12:
                new_lots.append(lot)
        self.lots = new_lots
        return removed


@dataclass
class DividendReceivable:
    symbol: str
    amount: float
    ex_date: pd.Timestamp
    pay_date: pd.Timestamp


@dataclass
class PendingRebalance:
    plan: SignalPlan
    first_execution_date: pd.Timestamp
    signal_equity: float
    target_values: dict[str, float]
    target_shares: dict[str, float]
    remaining: set[str]
    liquidity: dict[str, float]
    volatility: dict[str, float]
    reserve_cash: float
    attempts: int = 0


@dataclass(frozen=True)
class ExecutionQuote:
    reference_price: float
    price: float
    lagged_adv20: float
    participation_cap_notional: float
    participation_rate: float
    signal_volatility: float
    fixed_slippage_bps: float
    market_impact_bps: float
    modeled_slippage_bps: float
    realized_slippage_bps: float


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    selections: pd.DataFrame
    corporate_events: pd.DataFrame
    final_positions: dict[str, float]
    warnings: list[str] = field(default_factory=list)


class Backtester:
    def __init__(self, bundle: MarketDataBundle, config: AppConfig) -> None:
        config.validate()
        self.bundle = bundle.prepare(strict=config.data.strict_validation)
        self.config = config
        self.strategy = MultiFactorStrategy(
            self.bundle,
            config.strategy,
            config.portfolio,
        )
        self.cash = float(config.backtest.initial_cash)
        self.positions: dict[str, Position] = {}
        self.receivables: list[DividendReceivable] = []
        self.pending: PendingRebalance | None = None
        self.trade_records: list[dict[str, Any]] = []
        self.order_records: list[dict[str, Any]] = []
        self.selection_records: list[pd.DataFrame] = []
        self.corporate_records: list[dict[str, Any]] = []
        self.equity_records: list[dict[str, Any]] = []
        self.total_fees = 0.0
        self.total_notional = 0.0
        self.total_dividends = 0.0
        self.total_cash_interest = 0.0
        self.total_delist_writeoff = 0.0
        self.warnings: list[str] = []
        self._stale_warned: set[str] = set()
        self._processed_delistings: set[str] = set()

        bars = self.bundle.bars.copy()
        self.bars = bars.set_index(["date", "symbol"]).sort_index()
        all_calendar = self.bundle.calendar
        self.close_marks = (
            bars.pivot(index="date", columns="symbol", values="close")
            .reindex(all_calendar)
            .ffill()
        )
        self.benchmark_close = (
            self.bundle.benchmark.set_index("date")["close"].reindex(all_calendar).ffill()
        )
        self.bar_dates = {
            symbol: pd.DatetimeIndex(group["date"].sort_values())
            for symbol, group in bars.groupby("symbol", sort=False)
        }
        self.calendar_location = {date: index for index, date in enumerate(all_calendar)}
        self.actions_by_date = {
            date: group.copy()
            for date, group in self.bundle.corporate_actions.groupby("ex_date", sort=False)
        }
        security_frame = self.bundle.securities.set_index("symbol")
        self.security_master = security_frame.to_dict(orient="index")

    def run(self) -> BacktestResult:
        start = pd.Timestamp(self.config.backtest.start_date)
        end = pd.Timestamp(self.config.backtest.end_date)
        calendar = self.bundle.calendar[
            (self.bundle.calendar >= start) & (self.bundle.calendar <= end)
        ]
        if len(calendar) < 2:
            raise ValueError("回测区间内交易日不足")

        schedule = self._build_schedule(calendar)
        benchmark_history = self.benchmark_close.loc[calendar].dropna()
        if benchmark_history.empty:
            raise ValueError("回测区间没有有效基准行情")
        benchmark_base = float(benchmark_history.iloc[0])

        for index, date in enumerate(calendar):
            if index > 0:
                self._accrue_cash_interest()
            self._apply_delistings(date)
            current_weights = (
                self._current_weights_at_recorded_close(schedule[date])
                if date in schedule
                else {}
            )
            self._apply_corporate_actions(date)
            self._settle_dividends(date)
            if date in schedule:
                if self.pending is not None:
                    self._cancel_pending(date, "new_rebalance_superseded")
                current_holdings = {
                    symbol
                    for symbol, position in self.positions.items()
                    if position.total_shares > 1e-12
                }
                plan = self.strategy.generate(
                    schedule[date],
                    current_holdings=current_holdings,
                    current_weights=current_weights,
                )
                if not plan.selection.empty:
                    selection = plan.selection.copy()
                    selection["planned_execution_date"] = date
                    self.selection_records.append(selection)
                self._activate(plan, date)
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
        corporate_events = pd.DataFrame(self.corporate_records)
        final_positions = {
            symbol: self._position_value(symbol, calendar[-1], use_open=False)
            for symbol in sorted(self.positions)
        }
        if orders.empty:
            self.warnings.append("回测区间没有产生订单；请检查历史长度、股票池和过滤参数。")
        if not self.config.data.benchmark_is_total_return:
            self.warnings.append("当前基准不是全收益指数，长期超额收益不可直接比较。")
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            orders=orders,
            selections=selections,
            corporate_events=corporate_events,
            final_positions=final_positions,
            warnings=self.warnings,
        )

    def _build_schedule(self, calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
        full_calendar = self.bundle.calendar
        month_ends = (
            pd.Series(calendar, index=calendar)
            .groupby(calendar.to_period("M"))
            .max()
            .tolist()
        )
        schedule: dict[pd.Timestamp, pd.Timestamp] = {}
        for signal_date in month_ends:
            location = full_calendar.searchsorted(signal_date, side="right")
            if location >= len(full_calendar):
                continue
            execution_date = full_calendar[location]
            if execution_date > calendar[-1]:
                continue
            schedule[execution_date] = pd.Timestamp(signal_date)
        return schedule

    def _activate(self, plan: SignalPlan, execution_date: pd.Timestamp) -> None:
        signal_equity = self._equity_at_recorded_close(plan.signal_date)
        target_values = {
            symbol: float(weight) * signal_equity for symbol, weight in plan.weights.items()
        }
        target_shares: dict[str, float] = {}
        for symbol, value in target_values.items():
            reference_price = float(plan.reference_prices.get(symbol, np.nan))
            if not np.isfinite(reference_price) or reference_price <= 0:
                raise ValueError(f"{symbol} 缺少信号日收盘价，无法冻结订单股数")
            target_shares[symbol] = self._floor_lot(value / reference_price)
        remaining = set(self.positions).union(target_shares)
        liquidity: dict[str, float] = {}
        volatility: dict[str, float] = {}
        for symbol in remaining:
            value = plan.liquidity.get(symbol, np.nan)
            if not np.isfinite(value) or value <= 0:
                value = self.strategy.trailing_amount(symbol, plan.signal_date)
            liquidity[symbol] = float(value) if np.isfinite(value) else np.inf
            risk = plan.volatility.get(symbol, np.nan)
            if not np.isfinite(risk) or risk <= 0:
                risk = self.strategy.trailing_volatility(symbol, plan.signal_date)
            volatility[symbol] = float(risk) if np.isfinite(risk) else np.nan
        self.pending = PendingRebalance(
            plan=plan,
            first_execution_date=execution_date,
            signal_equity=signal_equity,
            target_values=target_values,
            target_shares=target_shares,
            remaining=remaining,
            liquidity=liquidity,
            volatility=volatility,
            reserve_cash=signal_equity * self.config.execution.cash_buffer,
        )

    def _attempt_pending(self, date: pd.Timestamp) -> None:
        pending = self.pending
        if pending is None:
            return
        pending.attempts += 1
        sell_symbols = [
            symbol
            for symbol in sorted(pending.remaining)
            if self._shares(symbol) > pending.target_shares.get(symbol, 0.0) + 1e-8
        ]
        for symbol in sell_symbols:
            self._execute_sell(symbol, date, pending)

        buy_symbols = [
            symbol
            for symbol in sorted(pending.remaining)
            if pending.target_shares.get(symbol, 0.0) > self._shares(symbol) + 1e-8
        ]
        for symbol in buy_symbols:
            self._execute_buy(symbol, date, pending)

        for symbol in list(pending.remaining):
            target = pending.target_shares.get(symbol, 0.0)
            gap = abs(target - self._shares(symbol))
            if gap <= 1e-8 or (
                target > 0 and gap < self.config.execution.lot_size - 1e-8
            ):
                pending.remaining.discard(symbol)

        if not pending.remaining:
            self.pending = None
        elif pending.attempts >= self.config.execution.rebalance_retry_days:
            self._cancel_pending(date, "retry_window_expired")

    def _execute_sell(
        self, symbol: str, date: pd.Timestamp, pending: PendingRebalance
    ) -> None:
        position = self.positions.get(symbol)
        if position is None or position.total_shares <= 1e-12:
            pending.remaining.discard(symbol)
            return
        target = pending.target_shares.get(symbol, 0.0)
        desired = max(0.0, position.total_shares - target)
        sellable = position.sellable_shares(date)
        if sellable <= 1e-12:
            self._record_order(
                pending,
                date,
                symbol,
                "SELL",
                "REJECTED",
                "T+1",
                0,
                0,
                0,
                0,
                requested_shares=desired,
            )
            return
        bar, rejection = self._execution_bar(symbol, date, "SELL")
        if rejection:
            self._record_order(
                pending,
                date,
                symbol,
                "SELL",
                "REJECTED",
                rejection,
                0,
                0,
                0,
                0,
                requested_shares=desired,
            )
            return
        assert bar is not None
        if self._requires_missing_liquidity_rejection(pending, symbol):
            self._record_order(
                pending,
                date,
                symbol,
                "SELL",
                "REJECTED",
                "missing_signal_liquidity",
                0,
                0,
                0,
                0,
                requested_shares=desired,
            )
            return
        reference_price = float(bar["open"])
        participation_cap = self._participation_cap(pending, symbol)
        capacity_price = (
            self._execution_quote(pending, symbol, bar, "SELL", 0.0).price
            if self.config.execution.market_impact_model == DEFAULT_EXECUTION_MODEL
            else reference_price
        )
        cap_shares = participation_cap / max(capacity_price, 1e-12)
        desired = min(desired, sellable)
        full_exit = (
            target <= 1e-8
            and desired >= position.total_shares - 1e-8
            and desired * capacity_price <= participation_cap + 1e-8
        )
        odd_component = position.total_shares % self.config.execution.lot_size
        odd_cleanup = (
            desired > 1e-8
            and desired <= odd_component + 1e-8
            and desired * capacity_price <= participation_cap + 1e-8
        )
        shares = (
            desired
            if full_exit or odd_cleanup
            else self._floor_lot(min(desired, cap_shares))
        )
        if shares <= 0:
            reason = "below_lot" if desired < self.config.execution.lot_size else "participation_cap"
            quote = self._execution_quote(pending, symbol, bar, "SELL", 0.0)
            self._record_order(
                pending,
                date,
                symbol,
                "SELL",
                "REJECTED",
                reason,
                0,
                quote.price,
                0,
                0,
                requested_shares=desired,
                quote=quote,
            )
            if reason == "below_lot" and target > 0:
                pending.remaining.discard(symbol)
            return

        quote = self._execution_quote(pending, symbol, bar, "SELL", shares)
        price = quote.price
        shares = position.remove_sellable(shares, date)
        notional = shares * price
        fees = self._fees(notional, "SELL", date)
        if position.total_shares <= 1e-10:
            del self.positions[symbol]
        self.cash += notional - fees
        self.total_fees += fees
        self.total_notional += notional
        remaining_gap = max(0.0, self._shares(symbol) - target)
        partial = remaining_gap >= self.config.execution.lot_size - 1e-8
        status = "PARTIAL" if partial else "FILLED"
        self._record_order(
            pending,
            date,
            symbol,
            "SELL",
            status,
            "",
            shares,
            price,
            notional,
            fees,
            requested_shares=desired,
            quote=quote,
        )
        self.trade_records.append(self.order_records[-1].copy())
        if not partial:
            pending.remaining.discard(symbol)

    def _execute_buy(
        self, symbol: str, date: pd.Timestamp, pending: PendingRebalance
    ) -> None:
        desired = max(
            0.0,
            pending.target_shares.get(symbol, 0.0) - self._shares(symbol),
        )
        bar, rejection = self._execution_bar(symbol, date, "BUY")
        if rejection:
            self._record_order(
                pending,
                date,
                symbol,
                "BUY",
                "REJECTED",
                rejection,
                0,
                0,
                0,
                0,
                requested_shares=desired,
            )
            return
        assert bar is not None
        if self._requires_missing_liquidity_rejection(pending, symbol):
            self._record_order(
                pending,
                date,
                symbol,
                "BUY",
                "REJECTED",
                "missing_signal_liquidity",
                0,
                0,
                0,
                0,
                requested_shares=desired,
            )
            return
        reference_price = float(bar["open"])
        participation_cap = self._participation_cap(pending, symbol)
        capacity_price = (
            self._execution_quote(pending, symbol, bar, "BUY", 0.0).price
            if self.config.execution.market_impact_model == DEFAULT_EXECUTION_MODEL
            else reference_price
        )
        shares = self._floor_lot(
            min(desired, participation_cap / max(capacity_price, 1e-12))
        )
        if shares <= 0:
            reason = (
                "below_lot"
                if desired < self.config.execution.lot_size
                else "participation_cap"
            )
            quote = self._execution_quote(pending, symbol, bar, "BUY", 0.0)
            self._record_order(
                pending,
                date,
                symbol,
                "BUY",
                "REJECTED",
                reason,
                0,
                quote.price,
                0,
                0,
                requested_shares=desired,
                quote=quote,
            )
            if reason == "below_lot":
                pending.remaining.discard(symbol)
            return

        available_cash = max(0.0, self.cash - pending.reserve_cash)
        quote = self._execution_quote(pending, symbol, bar, "BUY", shares)
        affordable = self._affordable_shares(
            available_cash,
            quote.price,
            shares,
            date,
        )
        if affordable < shares:
            shares = affordable
            quote = self._execution_quote(pending, symbol, bar, "BUY", shares)
        if shares <= 0:
            self._record_order(
                pending,
                date,
                symbol,
                "BUY",
                "REJECTED",
                "insufficient_cash",
                0,
                quote.price,
                0,
                0,
                requested_shares=desired,
                quote=quote,
            )
            return

        price = quote.price
        notional = shares * price
        fees = self._fees(notional, "BUY", date)
        total_cost = notional + fees
        if total_cost > available_cash + 1e-8:
            self._record_order(
                pending,
                date,
                symbol,
                "BUY",
                "REJECTED",
                "insufficient_cash",
                0,
                price,
                0,
                0,
                requested_shares=desired,
                quote=quote,
            )
            return
        position = self.positions.setdefault(symbol, Position())
        position.add(
            Lot(
                shares=shares,
                acquired_date=date,
                sellable_date=self._next_trading_date(date),
                source="BUY",
            )
        )
        self.cash -= total_cost
        self.total_fees += fees
        self.total_notional += notional
        remaining_gap = max(0.0, pending.target_shares.get(symbol, 0.0) - position.total_shares)
        partial = remaining_gap >= self.config.execution.lot_size - 1e-8
        status = "PARTIAL" if partial else "FILLED"
        self._record_order(
            pending,
            date,
            symbol,
            "BUY",
            status,
            "",
            shares,
            price,
            notional,
            fees,
            requested_shares=desired,
            quote=quote,
        )
        self.trade_records.append(self.order_records[-1].copy())
        if not partial:
            pending.remaining.discard(symbol)

    def _execution_bar(
        self, symbol: str, date: pd.Timestamp, side: str
    ) -> tuple[pd.Series | None, str]:
        master = self.security_master.get(symbol, {})
        list_date = master.get("list_date")
        delist_date = master.get("delist_date")
        if pd.notna(list_date) and date < pd.Timestamp(list_date):
            return None, "not_listed"
        if pd.notna(delist_date) and date > pd.Timestamp(delist_date):
            return None, "delisted"
        bar = self._bar(symbol, date)
        if bar is None or float(bar["volume"]) <= 0:
            return None, "suspended_or_no_bar"
        if side == "BUY" and self.config.execution.reject_st_on_execution and bool(bar["is_st"]):
            return None, "st_on_execution"
        if side == "BUY" and float(bar["open"]) >= float(bar["up_limit"]) - 0.005:
            return None, "open_at_up_limit"
        if side == "SELL" and float(bar["open"]) <= float(bar["down_limit"]) + 0.005:
            return None, "open_at_down_limit"
        return bar, ""

    def _apply_corporate_actions(self, date: pd.Timestamp) -> None:
        frame = self.actions_by_date.get(date)
        if frame is None:
            return
        for action in frame.itertuples(index=False):
            symbol = str(action.symbol)
            position = self.positions.get(symbol)
            if position is None or position.total_shares <= 0:
                continue
            entitled_shares = position.total_shares
            cash_per_share = float(action.cash_dividend)
            stock_per_share = float(action.stock_dividend)
            if cash_per_share > 0:
                amount = entitled_shares * cash_per_share
                pay_date = pd.Timestamp(action.pay_date) if pd.notna(action.pay_date) else date
                self.receivables.append(DividendReceivable(symbol, amount, date, pay_date))
                self.corporate_records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "event": "CASH_DIVIDEND_RECEIVABLE",
                        "shares": entitled_shares,
                        "amount": amount,
                    }
                )
            if stock_per_share > 0:
                new_shares = entitled_shares * stock_per_share
                sellable_date = (
                    pd.Timestamp(action.stock_list_date)
                    if pd.notna(action.stock_list_date)
                    else date
                )
                position.add(Lot(new_shares, date, sellable_date, "STOCK_DIVIDEND"))
                self.corporate_records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "event": "STOCK_DIVIDEND",
                        "shares": new_shares,
                        "amount": 0.0,
                    }
                )

    def _settle_dividends(self, date: pd.Timestamp) -> None:
        remaining: list[DividendReceivable] = []
        for receivable in self.receivables:
            if receivable.pay_date <= date:
                self.cash += receivable.amount
                self.total_dividends += receivable.amount
                self.corporate_records.append(
                    {
                        "date": date,
                        "symbol": receivable.symbol,
                        "event": "CASH_DIVIDEND_PAID",
                        "shares": 0.0,
                        "amount": receivable.amount,
                    }
                )
            else:
                remaining.append(receivable)
        self.receivables = remaining

    def _apply_delistings(self, date: pd.Timestamp) -> None:
        for symbol in list(self.positions):
            if symbol in self._processed_delistings:
                continue
            delist_date = self.security_master.get(symbol, {}).get("delist_date")
            if pd.isna(delist_date) or date <= pd.Timestamp(delist_date):
                continue
            value = self._position_value(symbol, date, use_open=False)
            if self.config.backtest.delist_value_policy == "last_close":
                self.cash += value
                settlement = value
                writeoff = 0.0
            else:
                settlement = 0.0
                writeoff = value
                self.total_delist_writeoff += writeoff
            del self.positions[symbol]
            self._processed_delistings.add(symbol)
            self.corporate_records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "event": "DELISTED",
                    "shares": 0.0,
                    "amount": settlement,
                    "writeoff": writeoff,
                }
            )
            self.warnings.append(
                f"{symbol} 于 {pd.Timestamp(delist_date).date()} 退市，按 "
                f"{self.config.backtest.delist_value_policy} 规则处理。"
            )

    def _accrue_cash_interest(self) -> None:
        annual_rate = self.config.backtest.annual_cash_rate
        if annual_rate == 0 or self.cash <= 0:
            return
        daily_rate = (1.0 + annual_rate) ** (1.0 / 252.0) - 1.0
        interest = self.cash * daily_rate
        self.cash += interest
        self.total_cash_interest += interest

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
        *,
        requested_shares: float | None = None,
        quote: ExecutionQuote | None = None,
    ) -> None:
        commission, stamp, transfer = (
            self._fee_components(notional, side, date) if notional > 0 else (0.0, 0.0, 0.0)
        )
        target_shares = float(pending.target_shares.get(symbol, 0.0))
        requested = (
            float(requested_shares)
            if requested_shares is not None
            else abs(target_shares - self._shares(symbol))
        )
        remaining = abs(target_shares - self._shares(symbol))
        reference_price = quote.reference_price if quote is not None else 0.0
        lagged_adv20 = (
            quote.lagged_adv20
            if quote is not None
            else float(pending.liquidity.get(symbol, np.nan))
        )
        participation_cap = (
            quote.participation_cap_notional
            if quote is not None
            else self._participation_cap(pending, symbol)
        )
        signal_volatility = (
            quote.signal_volatility
            if quote is not None
            else float(pending.volatility.get(symbol, np.nan))
        )
        fixed_slippage_bps = (
            quote.fixed_slippage_bps
            if quote is not None
            else float(self.config.execution.slippage_bps)
        )
        impact_bps = quote.market_impact_bps if quote is not None else 0.0
        modeled_slippage_bps = (
            quote.modeled_slippage_bps
            if quote is not None
            else fixed_slippage_bps
        )
        realized_slippage_bps = (
            quote.realized_slippage_bps if quote is not None else 0.0
        )
        reference_notional = float(reference_price) * float(shares)
        self.order_records.append(
            {
                "signal_date": pending.plan.signal_date,
                "date": date,
                "attempt": pending.attempts,
                "symbol": symbol,
                "side": side,
                "status": status,
                "reason": reason,
                "portfolio_model": pending.plan.portfolio_model,
                "portfolio_status": pending.plan.portfolio_status,
                "execution_model": self.config.execution.market_impact_model,
                "target_shares": target_shares,
                "requested_shares": requested,
                "shares": float(shares),
                "remaining_shares": float(remaining),
                "fill_ratio": float(shares / requested) if requested > 0 else 0.0,
                "reference_price": float(reference_price),
                "price": float(price),
                "lagged_adv20": float(lagged_adv20),
                "participation_cap_notional": float(participation_cap),
                "participation_rate": (
                    float(quote.participation_rate) if quote is not None else 0.0
                ),
                "signal_volatility": float(signal_volatility),
                "fixed_slippage_bps": float(fixed_slippage_bps),
                "market_impact_bps": float(impact_bps),
                "modeled_slippage_bps": float(modeled_slippage_bps),
                "realized_slippage_bps": float(realized_slippage_bps),
                "estimated_fixed_slippage_cost": (
                    reference_notional * fixed_slippage_bps / 10_000.0
                ),
                "estimated_market_impact_cost": (
                    reference_notional * impact_bps / 10_000.0
                ),
                "realized_slippage_cost": (
                    reference_notional * realized_slippage_bps / 10_000.0
                ),
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
            target = pending.target_shares.get(symbol, 0.0)
            side = "BUY" if target > self._shares(symbol) else "SELL"
            self._record_order(pending, date, symbol, side, "CANCELLED", reason, 0, 0, 0, 0)
        self.pending = None

    def _record_close(self, date: pd.Timestamp, benchmark_base: float) -> None:
        holdings = sum(self._position_value(symbol, date, use_open=False) for symbol in self.positions)
        receivables = sum(item.amount for item in self.receivables)
        equity = self.cash + holdings + receivables
        benchmark_value = float(self.benchmark_close.loc[date])
        stale_positions = sum(self._is_stale(symbol, date) for symbol in self.positions)
        self.equity_records.append(
            {
                "date": date,
                "nav": equity,
                "cash": self.cash,
                "dividend_receivables": receivables,
                "holdings_value": holdings,
                "gross_exposure": holdings / equity if equity > 0 else np.nan,
                "positions": len(self.positions),
                "stale_positions": stale_positions,
                "benchmark_nav": self.config.backtest.initial_cash * benchmark_value / benchmark_base,
                "cumulative_fees": self.total_fees,
                "cumulative_notional": self.total_notional,
                "cumulative_dividends": self.total_dividends,
                "cumulative_cash_interest": self.total_cash_interest,
                "cumulative_delist_writeoff": self.total_delist_writeoff,
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
                return position.total_shares * float(bar["open"])
        try:
            mark = self.close_marks.at[date, symbol]
        except KeyError:
            return 0.0
        if pd.isna(mark):
            return 0.0
        self._handle_stale(symbol, date)
        return position.total_shares * float(mark)

    def _portfolio_equity(self, date: pd.Timestamp, use_open: bool) -> float:
        holdings = sum(
            self._position_value(symbol, date, use_open=use_open) for symbol in self.positions
        )
        return self.cash + holdings + sum(item.amount for item in self.receivables)

    def _equity_at_recorded_close(self, date: pd.Timestamp) -> float:
        for record in reversed(self.equity_records):
            if record["date"] == date:
                return float(record["nav"])
        raise ValueError(f"找不到信号日 {date.date()} 的组合净值")

    def _current_weights_at_recorded_close(
        self, date: pd.Timestamp
    ) -> dict[str, float]:
        equity = self._equity_at_recorded_close(date)
        if equity <= 0:
            return {}
        weights: dict[str, float] = {}
        for symbol, position in self.positions.items():
            try:
                close = float(self.close_marks.at[date, symbol])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(close) and close > 0 and position.total_shares > 0:
                weights[symbol] = position.total_shares * close / equity
        return weights

    def _shares(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        return position.total_shares if position is not None else 0.0

    def _participation_cap(self, pending: PendingRebalance, symbol: str) -> float:
        rate = self.config.execution.max_participation_of_20d_amount
        if rate <= 0:
            return np.inf
        amount = pending.liquidity.get(symbol, np.inf)
        if not np.isfinite(amount) or amount <= 0:
            return np.inf
        return float(amount) * rate

    def _requires_missing_liquidity_rejection(
        self, pending: PendingRebalance, symbol: str
    ) -> bool:
        if self.config.execution.market_impact_model == DEFAULT_EXECUTION_MODEL:
            return False
        amount = float(pending.liquidity.get(symbol, np.nan))
        return not np.isfinite(amount) or amount <= 0

    def _execution_quote(
        self,
        pending: PendingRebalance,
        symbol: str,
        bar: pd.Series,
        side: str,
        shares: float,
    ) -> ExecutionQuote:
        reference_price = float(bar["open"])
        amount = float(pending.liquidity.get(symbol, np.nan))
        participation = (
            max(0.0, float(shares) * reference_price / amount)
            if np.isfinite(amount) and amount > 0
            else 0.0
        )
        signal_volatility = float(pending.volatility.get(symbol, np.nan))
        impact = market_impact_bps(
            model=self.config.execution.market_impact_model,
            annualized_volatility=signal_volatility,
            participation_rate=participation,
            coefficient=self.config.execution.market_impact_coefficient,
            annualized_volatility_floor=(
                self.config.execution.market_impact_volatility_floor
            ),
            maximum_bps=self.config.execution.max_market_impact_bps,
        )
        fixed = float(self.config.execution.slippage_bps)
        modeled = fixed + impact
        direction = 1.0 if side == "BUY" else -1.0
        unconstrained = reference_price * (1.0 + direction * modeled / 10_000.0)
        if side == "BUY":
            price = min(float(bar["up_limit"]), unconstrained)
            realized = max(0.0, (price / reference_price - 1.0) * 10_000.0)
        else:
            price = max(float(bar["down_limit"]), unconstrained)
            realized = max(0.0, (1.0 - price / reference_price) * 10_000.0)
        return ExecutionQuote(
            reference_price=reference_price,
            price=float(price),
            lagged_adv20=amount,
            participation_cap_notional=self._participation_cap(pending, symbol),
            participation_rate=float(participation),
            signal_volatility=signal_volatility,
            fixed_slippage_bps=fixed,
            market_impact_bps=float(impact),
            modeled_slippage_bps=float(modeled),
            realized_slippage_bps=float(realized),
        )

    def _floor_lot(self, shares: float) -> float:
        lot = self.config.execution.lot_size
        return float(np.floor(max(0.0, shares) / lot + 1e-12) * lot)

    def _affordable_shares(
        self, cash: float, price: float, maximum: float, date: pd.Timestamp
    ) -> float:
        shares = self._floor_lot(min(maximum, cash / max(price, 1e-12)))
        lot = self.config.execution.lot_size
        while shares > 0:
            notional = shares * price
            if notional + self._fees(notional, "BUY", date) <= cash + 1e-8:
                return shares
            shares -= lot
        return 0.0

    def _fee_components(
        self, notional: float, side: str, date: pd.Timestamp
    ) -> tuple[float, float, float]:
        if notional <= 0:
            return 0.0, 0.0, 0.0
        schedule = self.config.execution.fee_on(date)
        commission = max(
            self.config.execution.minimum_commission,
            notional * self.config.execution.commission_rate,
        )
        stamp = notional * schedule.stamp_duty_sell if side == "SELL" else 0.0
        transfer = notional * schedule.transfer_fee_rate
        return commission, stamp, transfer

    def _fees(self, notional: float, side: str, date: pd.Timestamp) -> float:
        return float(sum(self._fee_components(notional, side, date)))

    def _next_trading_date(self, date: pd.Timestamp) -> pd.Timestamp:
        location = self.calendar_location.get(date)
        if location is None or location + 1 >= len(self.bundle.calendar):
            return date + pd.Timedelta(1, unit="D")
        return self.bundle.calendar[location + 1]

    def _last_bar_date(self, symbol: str, date: pd.Timestamp) -> pd.Timestamp | None:
        dates = self.bar_dates.get(symbol)
        if dates is None or dates.empty:
            return None
        location = dates.searchsorted(date, side="right") - 1
        return dates[location] if location >= 0 else None

    def _is_stale(self, symbol: str, date: pd.Timestamp) -> bool:
        last_date = self._last_bar_date(symbol, date)
        if last_date is None:
            return True
        current_location = self.calendar_location.get(date)
        last_location = self.calendar_location.get(last_date)
        if current_location is None or last_location is None:
            return False
        return current_location - last_location > self.config.backtest.maximum_stale_trading_days

    def _handle_stale(self, symbol: str, date: pd.Timestamp) -> None:
        if not self._is_stale(symbol, date):
            return
        last_date = self._last_bar_date(symbol, date)
        message = f"{symbol} 截至 {date.date()} 已超过陈旧价格阈值，最后行情为 {last_date}."
        if self.config.backtest.stale_price_policy == "error":
            raise ValueError(message)
        if symbol not in self._stale_warned:
            self.warnings.append(message)
            self._stale_warned.add(symbol)
