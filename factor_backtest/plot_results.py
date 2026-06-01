# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
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
    out_path = Path(out_path)
    get_folder_by_root(out_path.parent)

    figs = [
        _ic_figure(
            title="IC",
            series=ic,
            cumulative_name="Cumulative IC",
            bar_color="#4E79A7",
            line_color="#F28E2B",
            stats_text=_stats_line(stats_df, "IC"),
        ),
        _ic_figure(
            title="RankIC",
            series=rankic,
            cumulative_name="Cumulative RankIC",
            bar_color="#2E7D32",
            line_color="#D62728",
            stats_text=_stats_line(stats_df, "RankIC"),
        ),
        _rolling_t_figure(rolling_t),
        _layer_nav_figure(nav_df),
        _final_nav_bar_figure(nav_df),
        _turnover_figure(turnover_df),
        _table_figure(stats_df, title="Statistics", height=520),
    ]
    _write_report_html(
        out_path=out_path,
        title=f"{factor_name} factor analysis",
        figures=[fig for fig in figs if fig is not None],
    )


def write_simulation_html(
    out_path: Path,
    factor_name: str,
    nav_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    selections_df: pd.DataFrame,
    selection_profile_df: pd.DataFrame | None = None,
    offset_nav_df: pd.DataFrame | None = None,
) -> None:
    """生成单因子 Step2 HTML，集中展示资金曲线、回撤、换手、指标和最近交易。"""
    out_path = Path(out_path)
    get_folder_by_root(out_path.parent)

    if selection_profile_df is None:
        selection_profile_df = pd.DataFrame()

    figs = [
        _simulation_equity_figure(factor_name, nav_df, metrics_df),
        _simulation_offset_nav_figure(offset_nav_df),
        _simulation_daily_return_figure(nav_df),
        _simulation_turnover_figure(nav_df),
        _selection_profile_figure(selection_profile_df),
    ]
    _write_report_html(
        out_path=out_path,
        title=f"{factor_name} simulation backtest",
        figures=[fig for fig in figs if fig is not None],
    )


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


def _write_report_html(out_path: Path, title: str, figures: list[go.Figure]) -> None:
    """把多个独立 Plotly 图块写入同一个 HTML，保留每张图自己的图例。"""
    figure_blocks = []
    for idx, fig in enumerate(figures):
        figure_blocks.append(
            to_html(
                fig,
                include_plotlyjs="cdn" if idx == 0 else False,
                full_html=False,
                config={"responsive": True, "displayModeBar": True},
            )
        )
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 24px;
      background: #ffffff;
      color: #10233f;
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }}
    h1 {{
      margin: 0 0 18px 8px;
      font-size: 22px;
      font-weight: 600;
    }}
    .figure-container {{
      width: 1560px;
      max-width: none;
      margin: 0 0 28px 0;
      border-bottom: 1px solid #edf1f7;
      padding-bottom: 18px;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {''.join(f'<div class="figure-container">{block}</div>' for block in figure_blocks)}
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")


def _ic_figure(
    title: str,
    series: pd.Series,
    cumulative_name: str,
    bar_color: str,
    line_color: str,
    stats_text: str,
) -> go.Figure | None:
    """单独生成 IC 或 RankIC 图，左轴日度值，右轴累计值。"""
    s = pd.Series(series, dtype=float).dropna()
    if s.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=s.index,
            y=s.values,
            name=title,
            marker_color=bar_color,
            opacity=0.72,
        ),
        secondary_y=False,
    )
    ma = s.rolling(20, min_periods=5).mean()
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma.values,
            name=f"{title}_mean20",
            line=dict(color=_mean_line_color(title), width=2.4),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.cumsum().values,
            name=cumulative_name,
            line=dict(color=line_color, width=2.2),
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#8A94A6", secondary_y=False)
    _style_time_series_figure(
        fig,
        title=f"{title}<br><sup>{stats_text}</sup>" if stats_text else title,
        height=560,
        legend_x=0.855,
        x_domain=(0.0, 0.82),
    )
    fig.update_yaxes(title_text=title, tickformat=".4f", secondary_y=False)
    fig.update_yaxes(title_text=cumulative_name, tickformat=".2f", secondary_y=True)
    return fig


def _rolling_t_figure(rolling_t: pd.Series) -> go.Figure | None:
    s = pd.Series(rolling_t, dtype=float).dropna()
    if s.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            name="Rolling IC t-stat",
            line=dict(color="#B07AA1", width=2),
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#8A94A6")
    _style_time_series_figure(fig, title="Rolling IC t-stat", height=430, legend_x=0.84, x_domain=(0.0, 0.82))
    fig.update_yaxes(title_text="t-stat", tickformat=".2f")
    return fig


def _layer_nav_figure(nav_df: pd.DataFrame) -> go.Figure | None:
    if nav_df.empty:
        return None
    group_cols = _group_columns(nav_df)
    fig = go.Figure()
    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    for idx, col in enumerate(group_cols):
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df[col],
                name=col,
                line=dict(width=1.35, color=colors[idx % len(colors)]),
            )
        )
    if "long_short" in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df["long_short"],
                name="long_short",
                line=dict(width=3.0, color="#D62728"),
            )
        )
    if "benchmark" in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df["benchmark"],
                name="benchmark",
                line=dict(width=2.2, color="#6B7280", dash="dash"),
            )
        )
    _style_time_series_figure(fig, title="Layer NAV", height=660, legend_x=0.84, x_domain=(0.0, 0.82))
    fig.update_yaxes(title_text="NAV", tickformat=".3f")
    return fig


def _final_nav_bar_figure(nav_df: pd.DataFrame) -> go.Figure | None:
    group_cols = _group_columns(nav_df)
    if not group_cols:
        return None
    final_nav = nav_df[group_cols].tail(1).T.iloc[:, 0].astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=final_nav.index.tolist(),
            y=final_nav.values,
            name="final NAV",
            marker_color="#4E79A7",
            text=[_fmt(value) for value in final_nav.values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#8A94A6")
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=430,
        title={"text": "Layer Final NAV", "x": 0.5, "xanchor": "center"},
        margin=dict(l=70, r=70, t=70, b=55),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_yaxes(title_text="Final NAV", tickformat=".3f", gridcolor="#E5ECF6")
    fig.update_xaxes(title_text="Layer", gridcolor="#EEF2F7")
    return fig


def _turnover_figure(turnover_df: pd.DataFrame) -> go.Figure | None:
    if turnover_df.empty:
        return None
    fig = go.Figure()
    group_cols = _group_columns(turnover_df)
    for col in group_cols:
        fig.add_trace(
            go.Scatter(
                x=turnover_df.index,
                y=turnover_df[col],
                name=f"turnover_{col}",
                line=dict(width=1.0),
                opacity=0.55,
            )
        )
    if "average" in turnover_df.columns:
        fig.add_trace(
            go.Scatter(
                x=turnover_df.index,
                y=turnover_df["average"],
                name="turnover_average",
                line=dict(width=2.8, color="#D62728"),
            )
        )
    _style_time_series_figure(fig, title="Turnover", height=470, legend_x=0.84, x_domain=(0.0, 0.82))
    return fig


def _simulation_equity_figure(factor_name: str, nav_df: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure | None:
    if nav_df.empty or "strategy_nav" not in nav_df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_df.index,
            y=nav_df["strategy_nav"],
            name="strategy_nav",
            line=dict(color="#1F77B4", width=2.4),
        )
    )
    if "benchmark_nav" in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df["benchmark_nav"],
                name="benchmark_nav",
                line=dict(color="#6B7280", width=2.0, dash="dash"),
            )
        )
    if "excess_nav" in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df["excess_nav"],
                name="excess_nav",
                line=dict(color="#F28E2B", width=2.0),
            )
        )

    dd = drawdown(nav_df["strategy_nav"]).reindex(nav_df.index)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            name="drawdown",
            yaxis="y2",
            fill="tozeroy",
            line=dict(width=0, color="rgba(214, 39, 40, 0.16)"),
            fillcolor="rgba(214, 39, 40, 0.16)",
            hovertemplate="drawdown: %{y:.2%}<extra></extra>",
        )
    )
    peak, trough, max_dd = _max_drawdown_points(nav_df["strategy_nav"])
    if peak is not None and trough is not None:
        peak_value = float(nav_df.loc[peak, "strategy_nav"])
        trough_value = float(nav_df.loc[trough, "strategy_nav"])
        fig.add_shape(
            type="line",
            x0=peak,
            y0=peak_value,
            x1=trough,
            y1=trough_value,
            xref="x",
            yref="y",
            line=dict(color="#D62728", width=1.6, dash="dot"),
        )
        fig.add_annotation(
            x=trough,
            y=trough_value,
            xref="x",
            yref="y",
            text=f"Max DD {max_dd:.2%}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=42,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#D62728",
            font=dict(color="#D62728", size=12),
        )

    display_metrics = _metrics_display_frame(metrics_df)
    if not display_metrics.empty:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(display_metrics.columns),
                    fill_color="#2F3B52",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[display_metrics[col].tolist() for col in display_metrics.columns],
                    fill_color=[_striped_rows(len(display_metrics))] * len(display_metrics.columns),
                    align="left",
                    height=24,
                    font=dict(size=11),
                ),
                domain=dict(x=[0.785, 1.0], y=[0.0, 0.94]),
            )
        )

    dd_min = float(dd.min()) if not dd.dropna().empty else -0.01
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=720,
        title={"text": "Equity / benchmark / drawdown", "x": 0.37, "xanchor": "center"},
        xaxis=dict(domain=[0.0, 0.72], showspikes=True, spikemode="across+marker", gridcolor="#EEF2F7"),
        yaxis=dict(title="NAV", gridcolor="#E5ECF6", tickformat=".3f"),
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            position=0.72,
            tickformat=".0%",
            range=[min(dd_min * 1.15, -0.02), 0.02],
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.035,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D8DEE9",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=70, r=35, t=105, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _simulation_offset_nav_figure(offset_nav_df: pd.DataFrame | None) -> go.Figure | None:
    if offset_nav_df is None or offset_nav_df.empty:
        return None
    strategy_cols = [col for col in offset_nav_df.columns if str(col) != "benchmark_nav"]
    if not strategy_cols:
        return None

    fig = go.Figure()
    colors = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ]
    for idx, col in enumerate(strategy_cols):
        fig.add_trace(
            go.Scatter(
                x=offset_nav_df.index,
                y=offset_nav_df[col],
                name=str(col),
                line=dict(width=1.45, color=colors[idx % len(colors)]),
                opacity=0.82,
            )
        )
    if "benchmark_nav" in offset_nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=offset_nav_df.index,
                y=offset_nav_df["benchmark_nav"],
                name="benchmark_nav",
                line=dict(color="#111827", width=2.6, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=560,
        title={"text": "所有 offset 净值对比", "x": 0.38, "xanchor": "center"},
        xaxis=dict(domain=[0.0, 0.78], showspikes=True, spikemode="across+marker", gridcolor="#EEF2F7"),
        yaxis=dict(title="NAV", tickformat=".3f", gridcolor="#E5ECF6"),
        legend=dict(
            x=0.805,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#D8DEE9",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        margin=dict(l=70, r=35, t=85, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _simulation_daily_return_figure(nav_df: pd.DataFrame) -> go.Figure | None:
    if nav_df.empty or "strategy_nav" not in nav_df.columns:
        return None
    ret = nav_df["strategy_nav"].pct_change().replace([np.inf, -np.inf], np.nan)
    active = ret.notna()
    if "position_value" in nav_df.columns:
        active &= pd.Series(nav_df["position_value"], index=nav_df.index).fillna(0) > 0
    else:
        active &= ret.ne(0)
    summary = _return_summary(ret[active], ret)
    colors = np.where(ret.fillna(0).to_numpy() >= 0, "#C81D25", "#178C4E")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=nav_df.index,
            y=ret,
            name="daily_return",
            marker_color=colors,
            opacity=0.9,
        )
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#8A94A6")
    fig.add_trace(_summary_table_trace(summary, x_domain=[0.80, 1.0], y_domain=[0.08, 0.88]))
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=420,
        title={"text": "Daily return", "x": 0.36, "xanchor": "center"},
        xaxis=dict(domain=[0.0, 0.74], showspikes=True, spikemode="across+marker", gridcolor="#EEF2F7"),
        yaxis=dict(title="Return", tickformat=".2%", gridcolor="#E5ECF6"),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=70, r=35, t=80, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _simulation_turnover_figure(nav_df: pd.DataFrame) -> go.Figure | None:
    if nav_df.empty or "turnover" not in nav_df.columns:
        return None
    turnover = pd.Series(nav_df["turnover"], index=nav_df.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    active_turnover = turnover[turnover.notna() & (turnover > 0)]
    summary = {
        "Active Mean": _fmt_pct(active_turnover.mean()) if not active_turnover.empty else "",
        "All-Day Mean": _fmt_pct(turnover.dropna().mean()) if not turnover.dropna().empty else "",
        "Active Days": f"{len(active_turnover):,}",
    }
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=nav_df.index,
            y=turnover,
            name="turnover",
            marker_color="#2F6DAE",
            opacity=0.88,
        )
    )
    if not active_turnover.empty:
        fig.add_hline(y=float(active_turnover.mean()), line_dash="dash", line_color="#D62728")
    fig.add_trace(_summary_table_trace(summary, x_domain=[0.80, 1.0], y_domain=[0.08, 0.88]))
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=420,
        title={"text": "Daily turnover", "x": 0.36, "xanchor": "center"},
        xaxis=dict(domain=[0.0, 0.74], showspikes=True, spikemode="across+marker", gridcolor="#EEF2F7"),
        yaxis=dict(title="Turnover", tickformat=".2%", gridcolor="#E5ECF6"),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=70, r=35, t=80, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_yaxes(title_text="Turnover", tickformat=".2f")
    return fig


def _selection_profile_figure(selection_profile_df: pd.DataFrame) -> go.Figure | None:
    if selection_profile_df.empty or not {"category", "segment", "weight_pct"}.issubset(selection_profile_df.columns):
        return None

    industry = (
        selection_profile_df[selection_profile_df["category"] == "industry"]
        .sort_values("weight_pct", ascending=False)
        .head(15)
        .sort_values("weight_pct", ascending=True)
    )
    market_cap = selection_profile_df[selection_profile_df["category"] == "market_cap"].copy()
    cap_order = ["<50亿", "50-100亿", "100-200亿", "200-500亿", "500-1000亿", ">=1000亿", "Unknown"]
    market_cap["order"] = market_cap["segment"].map({name: idx for idx, name in enumerate(cap_order)}).fillna(999)
    market_cap = market_cap.sort_values("order", ascending=False)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.16,
        subplot_titles=["Selected industry exposure", "Selected market-cap exposure"],
    )
    if not industry.empty:
        fig.add_trace(
            go.Bar(
                x=industry["weight_pct"],
                y=industry["segment"],
                orientation="h",
                name="industry",
                marker_color="#2F6DAE",
                text=industry["weight_pct"].map(lambda value: f"{value:.1%}"),
                textposition="outside",
                cliponaxis=False,
            ),
            row=1,
            col=1,
        )
    if not market_cap.empty:
        fig.add_trace(
            go.Bar(
                x=market_cap["weight_pct"],
                y=market_cap["segment"],
                orientation="h",
                name="market_cap",
                marker_color="#8E5EA2",
                text=market_cap["weight_pct"].map(lambda value: f"{value:.1%}"),
                textposition="outside",
                cliponaxis=False,
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=max(520, 28 * max(len(industry), len(market_cap)) + 180),
        title={"text": "Selected stock profile", "x": 0.5, "xanchor": "center"},
        showlegend=False,
        margin=dict(l=160, r=80, t=90, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(tickformat=".0%", gridcolor="#E5ECF6")
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


def _table_figure(table_df: pd.DataFrame, title: str, height: int = 500) -> go.Figure | None:
    if table_df.empty:
        return None
    display = table_df.copy()
    display = display.rename(
        columns={
            "ir": "ICIR/RankICIR",
            "annualized_ir": "Annualized ICIR/RankICIR",
            "win_rate": "Directional Win Rate",
            "t_value": "Directional t",
            "p_value": "Directional p",
        }
    )
    for col in display.columns:
        display[col] = display[col].map(_fmt)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display.columns),
                    fill_color="#2F3B52",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[display[col].tolist() for col in display.columns],
                    fill_color=[_striped_rows(len(display))] * len(display.columns),
                    align="left",
                    height=24,
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=height,
        title={"text": title, "x": 0.0, "xanchor": "left"},
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#FFFFFF",
    )
    return fig


def _style_time_series_figure(
    fig: go.Figure,
    title: str,
    height: int,
    legend_x: float,
    x_domain: tuple[float, float],
) -> None:
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=height,
        title={"text": title, "x": 0.5, "xanchor": "center"},
        hovermode="x unified",
        legend=dict(
            x=legend_x,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#D8DEE9",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=70, r=70, t=80, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(domain=list(x_domain), showspikes=True, spikemode="across+marker", gridcolor="#EEF2F7")
    fig.update_yaxes(showspikes=True, spikemode="across", gridcolor="#E5ECF6")


def _group_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("G") and str(col)[1:].isdigit()]
    return sorted(cols, key=lambda col: int(str(col)[1:]))


def _mean_line_color(title: str) -> str:
    if str(title).lower() == "rankic":
        return "#145A32"
    if str(title).lower() == "ic":
        return "#1F4E79"
    return "#27364A"


def _stats_line(stats_df: pd.DataFrame, series_name: str) -> str:
    if stats_df.empty:
        return ""
    mask = (stats_df["period"].astype(str) == "TOTAL") & (stats_df["series"].astype(str) == series_name)
    if not mask.any():
        return ""
    row = stats_df.loc[mask].iloc[0]
    ir_name = "年化 RankICIR" if str(series_name).lower() == "rankic" else "年化 ICIR"
    parts = [
        f"mean: {_fmt(row.get('mean'))}",
        f"std: {_fmt(row.get('std'))}",
        f"{ir_name}: {_fmt(row.get('annualized_ir', row.get('ir')))}",
        f"win: {_fmt_pct(row.get('win_rate'))}",
        f"t: {_fmt(row.get('t_value'))}",
        f"count: {_fmt(row.get('count'))}",
    ]
    return " | ".join(parts)


def _metrics_display_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or not {"metric", "value"}.issubset(metrics_df.columns):
        return pd.DataFrame()
    name_map = {
        "start": "Start",
        "end": "End",
        "total_return": "Total Return",
        "annual_return": "Annual Return",
        "annual_volatility": "Annual Vol",
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown",
        "calmar": "Calmar",
        "win_rate": "Win Rate",
        "daily_mean": "Daily Mean",
        "daily_std": "Daily Std",
        "excess_return": "Excess Return",
        "total_commission": "Commission",
        "total_stamp_tax": "Stamp Tax",
        "avg_turnover": "Avg Turnover",
        "trade_count": "Trade Count",
        "rankic_mean_from_step1": "Step1 RankIC Mean",
        "auto_factor_direction": "Auto Direction",
        "benchmark_status": "Benchmark",
        "simulation_universe": "Universe",
        "weight_method": "Weight Method",
        "period": "Period",
        "offset_count": "Offset Count",
    }
    rows = []
    for _, row in metrics_df.iterrows():
        metric = str(row["metric"])
        rows.append(
            {
                "Metric": name_map.get(metric, metric),
                "Value": _fmt_metric(metric, row["value"]),
            }
        )
    return pd.DataFrame(rows)


def _return_summary(active_ret: pd.Series, all_ret: pd.Series) -> dict[str, str]:
    active = pd.Series(active_ret, dtype=float).dropna()
    all_valid = pd.Series(all_ret, dtype=float).dropna()
    return {
        "Active Mean": _fmt_pct(active.mean()) if not active.empty else "",
        "All-Day Mean": _fmt_pct(all_valid.mean()) if not all_valid.empty else "",
        "Active Days": f"{len(active):,}",
        "Win Rate": _fmt_pct((active > 0).mean()) if not active.empty else "",
    }


def _summary_table_trace(summary: dict[str, str], x_domain: list[float], y_domain: list[float]) -> go.Table:
    display = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    return go.Table(
        header=dict(
            values=list(display.columns),
            fill_color="#2F3B52",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[display[col].tolist() for col in display.columns],
            fill_color=[_striped_rows(len(display))] * len(display.columns),
            align="left",
            height=26,
            font=dict(size=11),
        ),
        domain=dict(x=x_domain, y=y_domain),
    )


def _fmt_metric(metric: str, value) -> str:
    if metric in {"start", "end", "auto_factor_direction", "benchmark_status", "weight_method", "period"}:
        return "" if value is None or pd.isna(value) else str(value)
    if metric in {"simulation_universe", "trade_count", "offset_count"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return f"{int(numeric):,}"
        return "" if value is None or pd.isna(value) else str(value)
    pct_metrics = {
        "total_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "win_rate",
        "daily_mean",
        "daily_std",
        "excess_return",
        "avg_turnover",
    }
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if metric in pct_metrics:
            return f"{float(numeric):.2%}"
        if abs(float(numeric)) >= 1000:
            return f"{float(numeric):,.2f}"
        return _fmt(float(numeric))
    return "" if value is None or pd.isna(value) else str(value)


def _fmt_pct(value) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.2%}"


def _max_drawdown_points(nav: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float]:
    s = pd.Series(nav, dtype=float).dropna()
    if s.empty:
        return None, None, 0.0
    dd = drawdown(s)
    trough = dd.idxmin()
    peak = s.loc[:trough].idxmax()
    return peak, trough, float(dd.loc[trough])


def _striped_rows(n_rows: int) -> list[str]:
    return ["#F8FAFD" if idx % 2 == 0 else "#FFFFFF" for idx in range(n_rows)]


def _fmt(value) -> str:
    """把报告表格中的数值、日期和空值格式化为更适合阅读的字符串。"""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return ""
        if abs(numeric) < 1:
            return f"{numeric:.6f}"
        if abs(numeric) >= 1000:
            return f"{numeric:,.2f}"
        return f"{numeric:.4f}"
    return str(value)
