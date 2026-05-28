# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metrics import drawdown
from utils.path_kit import get_folder_by_root


def write_factor_analysis_html(
    out_path: Path,
    factor_name: str,
    ic: pd.Series,
    rankic: pd.Series,
    rolling_t: pd.Series,
    nav_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> None:
    """生成单因子 Step1 HTML，集中展示 IC、RankIC、t 值、分层净值和统计表。"""
    # 因子研究报告：把 IC、RankIC、滚动 t 值、分层净值、换手和统计表集中到一个 HTML。
    out_path = Path(out_path)
    get_folder_by_root(out_path.parent)

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        row_heights=[0.18, 0.18, 0.14, 0.23, 0.12, 0.15],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{}],
            [{}],
            [{}],
            [{"type": "table"}],
        ],
        subplot_titles=[
            "IC",
            "RankIC",
            "Rolling IC t-stat",
            "Layer NAV",
            "Turnover",
            "Statistics",
        ],
    )

    fig.add_trace(go.Bar(x=ic.index, y=ic.values, name="IC", marker_color="#4E79A7"), row=1, col=1, secondary_y=False)
    fig.add_trace(
        go.Scatter(x=ic.index, y=ic.cumsum().values, name="Cumulative IC", line=dict(color="#F28E2B")),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(x=rankic.index, y=rankic.values, name="RankIC", marker_color="#59A14F"),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=rankic.index, y=rankic.cumsum().values, name="Cumulative RankIC", line=dict(color="#E15759")),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=rolling_t.index, y=rolling_t.values, name="Rolling IC t", line=dict(color="#B07AA1")),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#999999", row=3, col=1)

    for col in nav_df.columns:
        width = 3 if col in {"long_short", "benchmark"} else 1.4
        dash = "dash" if col == "benchmark" else None
        fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], name=col, line=dict(width=width, dash=dash)), row=4, col=1)

    if not turnover_df.empty:
        for col in turnover_df.columns:
            fig.add_trace(go.Scatter(x=turnover_df.index, y=turnover_df[col], name=f"turnover_{col}", line=dict(width=1.3)), row=5, col=1)

    table_df = stats_df.copy()
    if not table_df.empty:
        table_df = table_df.replace([np.inf, -np.inf], np.nan)
        display = table_df.copy()
        for col in display.columns:
            if col not in {"period", "series"}:
                display[col] = display[col].map(_fmt)
        fig.add_trace(
            go.Table(
                header=dict(values=list(display.columns), fill_color="#E8EEF7", align="left"),
                cells=dict(values=[display[col].tolist() for col in display.columns], fill_color="white", align="left"),
            ),
            row=6,
            col=1,
        )

    fig.update_layout(
        title=f"{factor_name} factor analysis",
        template="plotly_white",
        height=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def write_simulation_html(
    out_path: Path,
    factor_name: str,
    nav_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    selections_df: pd.DataFrame,
) -> None:
    """生成单因子 Step2 HTML，集中展示资金曲线、回撤、换手、指标和最近交易。"""
    # 模拟回测报告：展示策略净值、基准、超额、回撤、换手、指标和交易明细。
    out_path = Path(out_path)
    get_folder_by_root(out_path.parent)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[0.28, 0.18, 0.16, 0.18, 0.2],
        specs=[[{}], [{}], [{}], [{"type": "table"}], [{"type": "table"}]],
        subplot_titles=["NAV", "Drawdown", "Daily turnover", "Metrics", "Recent trades / selections"],
    )

    for col in ["strategy_nav", "benchmark_nav", "excess_nav"]:
        if col in nav_df.columns:
            dash = "dash" if col == "benchmark_nav" else None
            fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], name=col, line=dict(dash=dash)), row=1, col=1)

    if "strategy_nav" in nav_df.columns:
        dd = drawdown(nav_df["strategy_nav"])
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="drawdown", fill="tozeroy", line=dict(color="#E15759")), row=2, col=1)

    if "turnover" in nav_df.columns:
        fig.add_trace(go.Bar(x=nav_df.index, y=nav_df["turnover"], name="turnover", marker_color="#4E79A7"), row=3, col=1)

    display_metrics = metrics_df.copy()
    for col in display_metrics.columns:
        display_metrics[col] = display_metrics[col].map(_fmt)
    fig.add_trace(
        go.Table(
            header=dict(values=list(display_metrics.columns), fill_color="#E8EEF7", align="left"),
            cells=dict(values=[display_metrics[col].tolist() for col in display_metrics.columns], fill_color="white", align="left"),
        ),
        row=4,
        col=1,
    )

    trade_cols = ["date", "stock_code", "action", "shares", "exec_price", "value", "status", "reason"]
    trades_show = trades_df[[c for c in trade_cols if c in trades_df.columns]].tail(15).copy() if not trades_df.empty else pd.DataFrame()
    if trades_show.empty and not selections_df.empty:
        trades_show = selections_df.tail(15).copy()
    if not trades_show.empty:
        for col in trades_show.columns:
            trades_show[col] = trades_show[col].map(_fmt)
        fig.add_trace(
            go.Table(
                header=dict(values=list(trades_show.columns), fill_color="#E8EEF7", align="left"),
                cells=dict(values=[trades_show[col].tolist() for col in trades_show.columns], fill_color="white", align="left"),
            ),
            row=5,
            col=1,
        )

    fig.update_layout(
        title=f"{factor_name} simulation backtest",
        template="plotly_white",
        height=1300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def write_summary_html(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    """生成汇总 HTML，用于索引所有因子的单因子报告路径和核心指标。"""
    # 汇总页只承担索引作用，方便从多个因子结果快速跳转到单因子 HTML。
    out_path = Path(out_path)
    get_folder_by_root(out_path.parent)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ddd;padding:6px 10px;}th{background:#eef3fb;}</style>",
        "</head><body>",
        f"<h2>{title}</h2>",
        summary.to_html(index=False, escape=False),
        "</body></html>",
    ]
    out_path.write_text("\n".join(html), encoding="utf-8")


def _fmt(value) -> str:
    """把报告表格中的数值、日期和空值格式化为更适合阅读的字符串。"""
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        if abs(value) < 1:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)
