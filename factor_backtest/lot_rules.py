# -*- coding: utf-8 -*-
"""A 股不同板块的买入交易单位规则。

本模块只处理“目标买入股数如何按交易制度落地”，不判断股票是否
应该进入候选池，也不重算涨跌停状态。卖出侧由账户轧差逻辑处理，
允许卖出零股或非整百差额。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LotRuleResult:
    """买入股数按板块规则取整后的结果。"""

    shares: int
    board_type: str
    lot_rule: str
    reason: str = ""


@dataclass(frozen=True)
class StockBoardInfo:
    """股票代码对应的板块信息和默认买入规则说明。"""

    board_type: str
    lot_rule: str
    buyable: bool


def normalize_stock_code(stock: str) -> str:
    """把股票代码统一成 6 位数字；无法识别时返回原始大写文本。"""
    text = str(stock).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(text) == 6 and text.isdigit():
        return text
    if len(digits) == 6:
        return digits
    return text


def identify_stock_board(stock: str) -> StockBoardInfo:
    """根据 6 位股票代码识别 A 股板块，并给出买入交易单位规则。"""
    code = normalize_stock_code(stock)
    if not (len(code) == 6 and code.isdigit()):
        return StockBoardInfo("未知", "未知代码不可新买入", False)

    if code.startswith(("688", "689")):
        return StockBoardInfo("科创板", "200股起买，超过后按1股递增", True)
    if code.startswith(("600", "601", "603", "605")):
        return StockBoardInfo("沪主板", "100股整数倍", True)
    if code.startswith(("000", "001", "002", "003")):
        return StockBoardInfo("深主板", "100股整数倍", True)
    if code.startswith(("300", "301")):
        return StockBoardInfo("创业板", "100股整数倍", True)
    if code.startswith(("4", "8", "920")):
        return StockBoardInfo("北交所", "暂按100股整数倍", True)

    return StockBoardInfo("未知", "未知代码不可新买入", False)


def round_buy_shares_by_board(stock: str, raw_shares: int, lot_size: int = 100) -> LotRuleResult:
    """按股票板块规则把原始目标买入股数转换为可下单股数。"""
    raw = int(raw_shares)
    board = identify_stock_board(stock)
    if raw <= 0:
        return LotRuleResult(0, board.board_type, board.lot_rule, "目标股数小于等于0")
    if not board.buyable:
        return LotRuleResult(0, board.board_type, board.lot_rule, "未知代码不可新买入")

    if board.board_type == "科创板":
        if raw >= 200:
            return LotRuleResult(raw, board.board_type, board.lot_rule)
        return LotRuleResult(0, board.board_type, board.lot_rule, "低于科创板最低买入200股")

    rounded = raw // int(lot_size) * int(lot_size)
    if rounded > 0:
        return LotRuleResult(rounded, board.board_type, board.lot_rule)
    return LotRuleResult(0, board.board_type, board.lot_rule, f"低于最低买入单位{lot_size}股")
