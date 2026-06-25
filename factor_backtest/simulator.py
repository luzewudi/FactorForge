# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np

from .lot_rules import LotRuleResult, identify_stock_board, round_buy_shares_by_board
from utils.log_kit import logger


@dataclass
class Simulator:
    initial_cash: float
    commission_rate: float
    stamp_tax_rate: float
    lot_size: int = 100
    cash_buffer_ratio: float = 0.0
    cash: float = field(init=False)
    positions: dict[str, int] = field(default_factory=dict)
    trade_records: list[dict] = field(default_factory=list)
    daily_records: list[dict] = field(default_factory=list)
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0
    last_prices: dict[str, float] = field(default_factory=dict)
    _target_rejections: dict[str, tuple[int, LotRuleResult]] = field(default_factory=dict, init=False)
    _pending_buy_value: float = field(default=0.0, init=False)
    _pending_sell_value: float = field(default=0.0, init=False)
    _pending_commission: float = field(default=0.0, init=False)
    _pending_stamp_tax: float = field(default=0.0, init=False)
    _insufficient_cash_warned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """初始化账户现金；持仓、流水和估值记录由 dataclass 默认工厂创建。"""
        self.cash = float(self.initial_cash)
        self.cash_buffer_ratio = min(max(float(self.cash_buffer_ratio), 0.0), 0.99)

    def total_equity(self, close_prices: dict[str, float]) -> float:
        """按最新收盘价计算账户总权益：现金 + 当前持仓市值。"""
        self._update_last_prices(close_prices)
        value = self.cash
        for stock, shares in self.positions.items():
            price = self.last_prices.get(stock, np.nan)
            if np.isfinite(price):
                value += shares * price
        return float(value)

    def position_value(self, close_prices: dict[str, float]) -> float:
        """按最新收盘价计算持仓总市值，不包含现金。"""
        self._update_last_prices(close_prices)
        value = 0.0
        for stock, shares in self.positions.items():
            price = self.last_prices.get(stock, np.nan)
            if np.isfinite(price):
                value += shares * price
        return float(value)

    def adjust_to_target_weights(
        self,
        target_weights: dict[str, float],
        trade_prices: dict[str, float],
        close_prices: dict[str, float],
        limit_status: dict[str, int],
        trade_date: str,
        scores: dict[str, float] | None = None,
        factor_direction: bool = False,
        slippage: float = 0.0,
    ) -> float:
        """把账户调到目标权重：先卖出、再买入，并返回当日成交额换手率。"""
        before_equity = self.total_equity(close_prices)
        target_positions = self._target_shares(target_weights, trade_prices, before_equity, slippage)
        before_abs_value = self.position_value(close_prices)

        self._sell_to_target(target_positions, trade_prices, limit_status, trade_date, slippage)
        self._buy_to_target(target_positions, trade_prices, limit_status, trade_date, scores or {}, factor_direction, slippage)

        after_abs_value = self.position_value(close_prices)
        traded_value = sum(abs(record["value"]) for record in self.trade_records if record["date"] == trade_date)
        denom = max(before_equity, 1.0)
        turnover = traded_value / denom
        if before_abs_value == 0.0 and after_abs_value > 0:
            turnover = max(turnover, after_abs_value / denom)
        return float(turnover)

    def mark_to_market(self, date: str, close_prices: dict[str, float], turnover: float = 0.0) -> dict:
        """每日收盘估值并记录净值、现金、持仓市值、持仓数量和换手。"""
        equity = self.total_equity(close_prices)
        pos_value = self.position_value(close_prices)
        record = {
            "date": date,
            "cash": float(self.cash),
            "position_value": float(pos_value),
            "total_equity": float(equity),
            "nav": float(equity / self.initial_cash),
            "position_count": int(sum(1 for shares in self.positions.values() if shares > 0)),
            "turnover": float(turnover),
            "commission": float(self.total_commission),
            "stamp_tax": float(self.total_stamp_tax),
            "daily_buy_value": float(self._pending_buy_value),
            "daily_sell_value": float(self._pending_sell_value),
            "daily_commission": float(self._pending_commission),
            "daily_stamp_tax": float(self._pending_stamp_tax),
            "actual_position_ratio": float(pos_value / equity) if equity > 0 else np.nan,
        }
        self.daily_records.append(record)
        self._reset_pending_trade_totals()
        return record

    def _target_shares(
        self,
        target_weights: dict[str, float],
        trade_prices: dict[str, float],
        equity: float,
        slippage: float,
    ) -> dict[str, int]:
        """把目标权重转换成目标股数，并按板块买入单位约束取整。"""
        self._target_rejections.clear()
        total_weight = sum(max(0.0, float(w)) for w in target_weights.values())
        if total_weight <= 0:
            return {}
        out: dict[str, int] = {}
        usable_equity = equity * (1.0 - self.cash_buffer_ratio)
        for stock, weight in target_weights.items():
            stock = normalize_stock(stock)
            price = trade_prices.get(stock, np.nan)
            if not np.isfinite(price) or price <= 0:
                continue
            exec_price = price * (1.0 + slippage)
            target_value = usable_equity * max(0.0, float(weight)) / total_weight
            raw_shares = int(target_value / (exec_price * (1.0 + self.commission_rate)))
            result = round_buy_shares_by_board(stock, raw_shares, self.lot_size)
            if result.shares > 0:
                out[stock] = result.shares
            elif raw_shares > 0:
                self._target_rejections[stock] = (raw_shares, result)
        return out

    def _sell_to_target(
        self,
        target_positions: dict[str, int],
        trade_prices: dict[str, float],
        limit_status: dict[str, int],
        trade_date: str,
        slippage: float,
    ) -> None:
        """执行卖出计划；跌停或价格无效时不成交，并写入交易流水。"""
        for stock in sorted(list(self.positions.keys())):
            board = identify_stock_board(stock)
            current = self.positions.get(stock, 0)
            target = target_positions.get(stock, 0)
            shares = current - target
            if shares <= 0:
                continue
            raw_price = trade_prices.get(stock, np.nan)
            if not np.isfinite(raw_price) or raw_price <= 0:
                self._record(trade_date, stock, "SELL", 0, raw_price, 0.0, 0.0, 0.0, "价格无效", "price is nan", requested_shares=shares)
                continue
            if int(limit_status.get(stock, 0)) == -1:
                self._record(trade_date, stock, "SELL", 0, raw_price, 0.0, 0.0, 0.0, "跌停", "limit down", requested_shares=shares)
                continue
            exec_price = raw_price * (1.0 - slippage)
            value = shares * exec_price
            commission = value * self.commission_rate
            stamp = value * self.stamp_tax_rate
            self.cash += value - commission - stamp
            self.positions[stock] = current - shares
            if self.positions[stock] <= 0:
                self.positions.pop(stock, None)
            self.total_commission += commission
            self.total_stamp_tax += stamp
            self._record(
                trade_date,
                stock,
                "SELL",
                shares,
                exec_price,
                value,
                commission,
                stamp,
                "成交",
                "",
                requested_shares=shares,
                executed_shares=shares,
                board_type=board.board_type,
                lot_rule=board.lot_rule,
            )

    def _buy_to_target(
        self,
        target_positions: dict[str, int],
        trade_prices: dict[str, float],
        limit_status: dict[str, int],
        trade_date: str,
        scores: dict[str, float],
        factor_direction: bool,
        slippage: float,
    ) -> None:
        """执行买入计划；涨停、价格无效或资金不足时不成交，并写入交易流水。"""
        buy_plan = []
        for stock, target in target_positions.items():
            current = self.positions.get(stock, 0)
            shares = target - current
            if shares > 0:
                buy_plan.append((stock, shares, scores.get(stock, 0.0)))
        buy_plan.sort(key=lambda x: x[2], reverse=not factor_direction)

        for stock, shares, _score in buy_plan:
            requested_shares = int(shares)
            board = identify_stock_board(stock)
            raw_price = trade_prices.get(stock, np.nan)
            if not np.isfinite(raw_price) or raw_price <= 0:
                self._record(trade_date, stock, "BUY", 0, raw_price, 0.0, 0.0, 0.0, "价格无效", "price is nan", requested_shares=requested_shares)
                continue
            if int(limit_status.get(stock, 0)) == 1:
                self._record(trade_date, stock, "BUY", 0, raw_price, 0.0, 0.0, 0.0, "涨停", "limit up", requested_shares=requested_shares)
                continue
            exec_price = raw_price * (1.0 + slippage)
            raw_affordable = int(self.cash / (exec_price * (1.0 + self.commission_rate)))
            affordable = round_buy_shares_by_board(stock, raw_affordable, self.lot_size).shares
            shares = min(shares, affordable)
            if shares < requested_shares:
                self._warn_insufficient_cash(trade_date, stock, requested_shares, shares)
            if shares <= 0:
                self._record(
                    trade_date,
                    stock,
                    "BUY",
                    0,
                    exec_price,
                    0.0,
                    0.0,
                    0.0,
                    "资金不足",
                    "insufficient cash",
                    requested_shares=requested_shares,
                    executed_shares=0,
                    board_type=board.board_type,
                    lot_rule=board.lot_rule,
                )
                continue
            value = shares * exec_price
            commission = value * self.commission_rate
            cost = value + commission
            if cost > self.cash + 1e-6:
                self._warn_insufficient_cash(trade_date, stock, requested_shares, 0)
                self._record(
                    trade_date,
                    stock,
                    "BUY",
                    0,
                    exec_price,
                    0.0,
                    0.0,
                    0.0,
                    "资金不足",
                    "insufficient cash",
                    requested_shares=requested_shares,
                    executed_shares=0,
                    board_type=board.board_type,
                    lot_rule=board.lot_rule,
                )
                continue
            self.cash -= cost
            self.positions[stock] = self.positions.get(stock, 0) + shares
            self.total_commission += commission
            reason = "资金不足导致部分成交" if shares < requested_shares else ""
            self._record(
                trade_date,
                stock,
                "BUY",
                shares,
                exec_price,
                value,
                commission,
                0.0,
                "成交",
                reason,
                requested_shares=requested_shares,
                executed_shares=shares,
                board_type=board.board_type,
                lot_rule=board.lot_rule,
            )

        for stock, (raw_shares, result) in self._target_rejections.items():
            if self.positions.get(stock, 0) <= 0:
                raw_price = trade_prices.get(stock, np.nan)
                self._record(
                    trade_date,
                    stock,
                    "BUY",
                    0,
                    raw_price,
                    0.0,
                    0.0,
                    0.0,
                    "资金不足" if "低于" in result.reason else "不可买入",
                    result.reason,
                    requested_shares=raw_shares,
                    executed_shares=0,
                    board_type=result.board_type,
                    lot_rule=result.lot_rule,
                )

    def _record(
        self,
        date: str,
        stock: str,
        action: str,
        shares: int,
        exec_price: float,
        value: float,
        commission: float,
        stamp_tax: float,
        status: str,
        reason: str,
        requested_shares: int | None = None,
        executed_shares: int | None = None,
        board_type: str | None = None,
        lot_rule: str | None = None,
    ) -> None:
        """记录单笔交易或失败委托，作为后续交易明细 CSV 的原始来源。"""
        board = identify_stock_board(stock)
        requested = int(shares if requested_shares is None else requested_shares)
        executed = int(shares if executed_shares is None else executed_shares)
        if status == "成交":
            if action == "BUY":
                self._pending_buy_value += float(value)
            elif action == "SELL":
                self._pending_sell_value += float(value)
            self._pending_commission += float(commission)
            self._pending_stamp_tax += float(stamp_tax)
        self.trade_records.append(
            {
                "date": date,
                "stock_code": stock,
                "action": action,
                "shares": executed,
                "requested_shares": requested,
                "executed_shares": executed,
                "exec_price": float(exec_price) if np.isfinite(exec_price) else np.nan,
                "value": float(value),
                "commission": float(commission),
                "stamp_tax": float(stamp_tax),
                "status": status,
                "reason": reason,
                "board_type": board_type or board.board_type,
                "lot_rule": lot_rule or board.lot_rule,
            }
        )

    def _update_last_prices(self, close_prices: dict[str, float]) -> None:
        """更新每只股票最近一次有效收盘价，用于停牌或缺价时的估值延续。"""
        for stock, price in close_prices.items():
            if np.isfinite(price) and price > 0:
                self.last_prices[stock] = float(price)

    def _reset_pending_trade_totals(self) -> None:
        """清空本次调仓累计的成交额和费用，供下一条净值记录重新累计。"""
        self._pending_buy_value = 0.0
        self._pending_sell_value = 0.0
        self._pending_commission = 0.0
        self._pending_stamp_tax = 0.0

    def _warn_insufficient_cash(self, date: str, stock: str, requested: int, executed: int) -> None:
        """资金不足导致买入少于目标时输出一次醒目的诊断提示。"""
        if self._insufficient_cash_warned:
            return
        logger.warning(
            "[sim-backtest] 出现可用资金不足，建议提高 cash_buffer_ratio；"
            f"首个样本 date={date}, stock={stock}, 目标买入={requested}, 实际买入={executed}"
        )
        self._insufficient_cash_warned = True


def normalize_stock(stock: str) -> str:
    """把带交易所前缀的股票代码统一成 6 位数字代码。"""
    text = str(stock).strip().upper()
    if len(text) == 8 and text[:2] in {"SH", "SZ", "BJ"}:
        return text[2:]
    return text


def round_lot(shares: int, lot_size: int) -> int:
    """把目标股数按最小交易单位向下取整。"""
    if shares <= 0:
        return 0
    return int(shares // lot_size * lot_size)


def vector_to_price_dict(tickers: Iterable[str], values: np.ndarray) -> dict[str, float]:
    """把某日价格向量转换成 {股票代码: 价格} 字典，并过滤无效价格。"""
    out: dict[str, float] = {}
    for stock, value in zip(tickers, values):
        if np.isfinite(value) and value > 0:
            out[normalize_stock(stock)] = float(value)
    return out


def vector_to_limit_dict(tickers: Iterable[str], values: np.ndarray) -> dict[str, int]:
    """把某日涨跌停状态向量转换成 {股票代码: 状态} 字典，只保留非零状态。"""
    out: dict[str, int] = {}
    for stock, value in zip(tickers, values):
        if np.isfinite(value) and int(value) != 0:
            out[normalize_stock(stock)] = int(value)
    return out
