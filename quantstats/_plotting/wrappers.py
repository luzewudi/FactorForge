#!/usr/bin/env python
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as _plt
from matplotlib.ticker import (
    StrMethodFormatter as _StrMethodFormatter,
    FuncFormatter as _FuncFormatter,
)

import numpy as _np
import pandas as _pd
from .._compat import safe_resample
import seaborn as _sns

# Lazy imports to avoid circular dependency during package initialization
# These modules are imported when first accessed via _get_stats() and _get_utils()
_stats = None
_utils = None


def _get_stats():
    global _stats
    if _stats is None:
        from .. import stats
        _stats = stats
    return _stats


def _get_utils():
    global _utils
    if _utils is None:
        from .. import utils
        _utils = utils
    return _utils

from . import core as _core

if TYPE_CHECKING:
    from matplotlib.figure import Figure as _Figure

# Type alias for return data (Series or DataFrame)
Returns = _pd.Series | _pd.DataFrame

_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64", "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = (len(_FLATUI_COLORS) * ["black"]) + ["white"]

# Check if plotly is available for optional conversion functionality
_HAS_PLOTLY = False
try:
    import plotly

    _HAS_PLOTLY = True
except ImportError:
    pass


def _benchmark_label(benchmark) -> str:
    """Return the display label attached to a benchmark Series/DataFrame."""
    if isinstance(benchmark, _pd.Series):
        return str(benchmark.name) if benchmark.name else "Benchmark"
    if isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns):
        return str(benchmark.columns[0])
    return "Benchmark"


def to_plotly(fig: _Figure) -> _Figure:
    """
    Convert a matplotlib figure to a Plotly interactive plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert to Plotly format.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive Plotly figure if plotly is available, otherwise original figure.

    Notes
    -----
    This function requires the plotly library to be installed. If plotly is not
    available, the original matplotlib figure is returned unchanged.
    """
    # Return original figure if plotly not available
    if not _HAS_PLOTLY:
        return fig

    # Suppress warnings during conversion to avoid noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Convert matplotlib figure to plotly format
        fig = plotly.tools.mpl_to_plotly(fig)
        # Upload and display the interactive plot
        return plotly.plotly.iplot(fig, filename="quantstats-plot", overwrite=True)  # type: ignore


def snapshot(
    returns: Returns,
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 8),
    title: str = "投资组合摘要",
    fontname: str = "Arial",
    lw: float = 1.5,
    mode: str = "comp",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    log_scale: bool = False,
    **kwargs,
) -> _Figure | None:
    """
    生成带有多个子图的投资组合绩效综合快照。

    参数
    ----------
    returns : pandas.Series or pandas.DataFrame
        日收益率数据。如果 DataFrame 包含多列，则使用均值或特定列。
    grayscale : bool, optional
        如果为 True，使用灰度颜色而不是默认配色方案（默认：False）。
    figsize : tuple, optional
        图形尺寸，为 (宽度, 高度)，单位为英寸（默认：(10, 8)）。
    title : str, optional
        图表的主标题（默认："投资组合摘要"）。
    fontname : str, optional
        文本元素的字体（默认："Arial"）。
    lw : float, optional
        图表的线宽（默认：1.5）。
    mode : str, optional
        计算模式："comp" 用于复合收益率，"sum" 用于简单求和。
    subtitle : bool, optional
        是否显示带有日期范围和夏普比率的副标题（默认：True）。
    savefig : str or dict, optional
        保存图形的路径或包含 matplotlib savefig 参数的字典。
    show : bool, optional
        是否显示图表（默认：True）。
    log_scale : bool, optional
        是否对 y 轴使用对数刻度（默认：False）。
    **kwargs : dict
        其他关键字参数，包括用于列选择的 strategy_col。

    返回
    -------
    matplotlib.figure.Figure or None
        如果 show=False 则返回图形对象，否则返回 None。

    注意
    -----
    创建一个三面板图表，显示：
    1. 累计收益率随时间的变化
    2. 回撤期间
    3. 日收益率分布
    """
    fontname = _core._resolve_font(fontname)

    # 从 kwargs 中提取策略列名
    strategy_colname = kwargs.get("strategy_col", "策略")

    # 处理多列 DataFrame 输入
    multi_column = False
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            # 检查特定策略列是否存在
            if strategy_colname in returns.columns:
                returns = returns[strategy_colname]
            else:
                # 如果找不到特定列，使用所有列的均值
                multi_column = True
                returns = returns.mean(axis=1)
                title = title + " (日等权*)"
        returns.columns = strategy_colname

    # 根据灰度偏好选择配色方案
    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    # 转换为投资组合格式并计算百分比变化
    returns = _get_utils().make_portfolio(returns.dropna(), 1, mode).pct_change(fill_method=None).fillna(0)

    # 如果未指定，使用当前图形尺寸
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.75)

    # 创建三子图：累计收益率、回撤、日收益率
    fig, axes = _plt.subplots(
        3, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # 为多列 DataFrame 说明添加脚注
    if multi_column:
        _plt.figtext(
            0,
            -0.05,
            "            * 当传入多列 DataFrame 时，所有列的均值将作为收益率使用。\n"
            "              要更改此行为，请使用 pandas Series 或在 "
            "`strategy_col` 参数中传入列名。",
            ha="left",
            fontsize=11,
            color="black",
            alpha=0.6,
            linespacing=1.5,
        )

    # 为更整洁的外观移除所有坐标轴的边框
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # 设置主标题
    fig.suptitle(
        title, fontsize=14, y=0.97, fontname=fontname, fontweight="bold", color="black"
    )

    fig.set_facecolor("white")

    # 添加带有日期范围和夏普比率的副标题
    if subtitle:
        if isinstance(returns, _pd.Series):
            axes[0].set_title(
                "%s - %s ;  夏普: %.2f                      \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),  # type: ignore
                    returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                    _get_stats().sharpe(returns),
                ),
                fontsize=12,
                color="gray",
            )
        elif isinstance(returns, _pd.DataFrame):
            axes[0].set_title(
                "\n%s - %s ;  "
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),  # type: ignore
                    returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                ),
                fontsize=12,
                color="gray",
            )

    # 配置第一个子图：累计收益率
    axes[0].set_ylabel(
        "累计收益率", fontname=fontname, fontweight="bold", fontsize=12
    )

    # 为 Series 或 DataFrame 绘制累计收益率
    if isinstance(returns, _pd.Series):
        # 根据模式计算累计收益率
        if mode.lower() in ["cumsum", "sum"]:
            cum_ret = returns.cumsum() * 100
        else:
            cum_ret = _get_stats().compsum(returns) * 100
        # Plot cumulative returns line
        axes[0].plot(
            cum_ret,
            color=colors[1],
            lw=1 if grayscale else lw,
            zorder=1,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Plot each column separately for DataFrame
        for col in returns.columns:
            if mode.lower() in ["cumsum", "sum"]:
                cum_ret = returns[col].cumsum() * 100
            else:
                cum_ret = _get_stats().compsum(returns[col]) * 100
            axes[0].plot(
                cum_ret,
                label=col,
                lw=1 if grayscale else lw,
                zorder=1,
            )
    # Add horizontal line at zero
    axes[0].axhline(0, color="silver", lw=1, zorder=0)

    # Set y-axis scale based on log_scale parameter
    axes[0].set_yscale("symlog" if log_scale else "linear")
    # axes[0].legend(fontsize=12)

    # Configure second subplot: Drawdown
    dd = _get_stats().to_drawdown_series(returns) * 100
    # Calculate appropriate tick spacing for drawdown
    ddmin = _get_utils()._round_to_closest(abs(dd.min()), 5)
    ddmin_ticks = 5
    if ddmin > 50:
        ddmin_ticks = ddmin / 4
    elif ddmin > 20:
        ddmin_ticks = ddmin / 3
    ddmin_ticks = int(_get_utils()._round_to_closest(ddmin_ticks, 5))

    # ddmin_ticks = int(_get_utils()._round_to_closest(ddmin, 5))
    axes[1].set_ylabel("Drawdown", fontname=fontname, fontweight="bold", fontsize=12)
    axes[1].set_yticks(_np.arange(-ddmin, 0, step=ddmin_ticks))

    # Plot drawdown series
    if isinstance(dd, _pd.Series):
        axes[1].plot(dd, color=colors[2], lw=1 if grayscale else lw, zorder=1)
    elif isinstance(dd, _pd.DataFrame):
        for col in dd.columns:
            axes[1].plot(dd[col], label=col, lw=1 if grayscale else lw, zorder=1)
    axes[1].axhline(0, color="silver", lw=1, zorder=0)

    # Add filled area under drawdown curve if not grayscale
    if not grayscale:
        if isinstance(dd, _pd.Series):
            axes[1].fill_between(dd.index, 0, dd, color=colors[2], alpha=0.25)
        elif isinstance(dd, _pd.DataFrame):
            for i, col in enumerate(dd.columns):
                axes[1].fill_between(
                    dd[col].index, 0, dd[col], color=colors[i + 1], alpha=0.25
                )

    axes[1].set_yscale("symlog" if log_scale else "linear")
    # axes[1].legend(fontsize=12)

    # Configure third subplot: Daily Returns
    axes[2].set_ylabel(
        "Daily Return", fontname=fontname, fontweight="bold", fontsize=12
    )

    # Plot daily returns
    if isinstance(returns, _pd.Series):
        axes[2].plot(
            returns * 100, color=colors[0], label=returns.name, lw=0.5, zorder=1
        )
    elif isinstance(returns, _pd.DataFrame):
        for i, col in enumerate(returns.columns):
            axes[2].plot(
                returns[col] * 100, color=colors[i], label=col, lw=0.5, zorder=1
            )
    # Add horizontal lines at zero
    axes[2].axhline(0, color="silver", lw=1, zorder=0)
    axes[2].axhline(0, color=colors[-1], linestyle="--", lw=1, zorder=2)

    axes[2].set_yscale("symlog" if log_scale else "linear")
    # axes[2].legend(fontsize=12)

    # Calculate appropriate tick spacing for daily returns
    retmax = _get_utils()._round_to_closest(returns.max() * 100, 5)
    retmin = _get_utils()._round_to_closest(returns.min() * 100, 5)
    retdiff = retmax - retmin
    steps = 5
    if retdiff > 50:
        steps = retdiff / 5
    elif retdiff > 30:
        steps = retdiff / 4
    steps = _get_utils()._round_to_closest(steps, 5)
    axes[2].set_yticks(_np.arange(retmin, retmax, step=steps))

    # Apply common formatting to all axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_major_formatter(_StrMethodFormatter("{x:,.0f}%"))

    # Adjust layout
    _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    fig.autofmt_xdate()

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def earnings(
    returns: Returns,
    start_balance: float = 1e5,
    mode: str = "comp",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Portfolio Earnings",
    fontname: str = "Arial",
    lw: float = 1.5,
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Plot portfolio earnings over time showing absolute dollar value growth.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    start_balance : float, optional
        Starting portfolio balance in dollars (default: 100000).
    mode : str, optional
        Calculation mode: "comp" for compound returns or "sum" for simple sum.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    title : str, optional
        Main title for the plot (default: "Portfolio Earnings").
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for the earnings line (default: 1.5).
    subtitle : bool, optional
        Whether to show subtitle with date range and P&L (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows portfolio value over time starting from the specified balance.
    Highlights the maximum portfolio value achieved during the period.
    """
    fontname = _core._resolve_font(fontname)

    # Select color scheme and transparency based on grayscale preference
    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    alpha = 0.5 if grayscale else 0.8

    # Convert returns to portfolio dollar values
    returns = _get_utils().make_portfolio(returns, start_balance, mode)

    # Use current figure size if not specified
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.55)

    # Create single subplot figure
    fig, ax = _plt.subplots(figsize=figsize)

    # Remove spines for cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        f"    {title}",
        fontsize=12,
        y=0.95,
        fontname=fontname,
        fontweight="bold",
        color="black",
    )

    # Add subtitle with date range and P&L information
    if subtitle:
        ax.set_title(
            "\n%s - %s ;  P&L: %s (%s)                "
            % (
                returns.index.date[1:2][0].strftime("%e %b '%y"),  # type: ignore
                returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                _get_utils()._score_str(
                    "${:,}".format(round(returns.values[-1] - returns.values[0], 2))
                ),
                _get_utils()._score_str(
                    "{:,}%".format(
                        round((returns.values[-1] / returns.values[0] - 1) * 100, 2)
                    )
                ),
            ),
            fontsize=10,
            color="gray",
        )

    # Find and highlight maximum portfolio value
    mx = returns.max()
    returns_max = returns[returns == mx]
    ix = returns_max[~_np.isnan(returns_max)].index[0]
    returns_max = _np.where(returns.index == ix, mx, _np.nan)

    # Plot maximum value point as a marker
    ax.plot(
        returns.index,
        returns_max,
        marker="o",
        lw=0,
        alpha=alpha,
        markersize=12,
        color=colors[0],
    )

    # Plot main earnings line
    ax.plot(returns.index, returns, color=colors[1], lw=1 if grayscale else lw)

    # Set y-axis label showing starting balance
    ax.set_ylabel(
        "Value of  ${:,.0f}".format(start_balance),
        fontname=fontname,
        fontweight="bold",
        fontsize=11,
    )

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(_FuncFormatter(_core.format_cur_axis))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    _plt.xticks(fontsize=11)
    _plt.yticks(fontsize=11)

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    fig.autofmt_xdate()

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def returns(
    returns: Returns,
    benchmark: Returns | str | None = None,
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 6),
    fontname: str = "Arial",
    lw: float = 1.5,
    match_volatility: bool = False,
    compound: bool = True,
    resample: str | None = None,
    ylabel: str = "Cumulative Returns",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot cumulative returns over time, optionally compared to a benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 1.5).
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    compound : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    resample : str, optional
        Resampling frequency (e.g., 'M' for monthly, 'Q' for quarterly).
    ylabel : str, optional
        Y-axis label (default: "Cumulative Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Creates a time series plot of cumulative returns. If benchmark is provided,
    both series are plotted for comparison.
    """
    # Build title based on parameters
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += " vs %s" % benchmark.upper()
        else:
            title += " vs %s" % _benchmark_label(benchmark)
        if match_volatility:
            title += " (Volatility Matched)"

        # Prepare benchmark data to match returns index
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)

    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Use core plotting function for time series
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=False,
        resample=resample,
        compound=compound,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def log_returns(
    returns: Returns,
    benchmark: Returns | str | None = None,
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 5),
    fontname: str = "Arial",
    lw: float = 1.5,
    match_volatility: bool = False,
    compound: bool = True,
    resample: str | None = None,
    ylabel: str = "Cumulative Returns",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot cumulative returns on a logarithmic scale for better trend visualization.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 1.5).
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    compound : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    resample : str, optional
        Resampling frequency (e.g., 'M' for monthly, 'Q' for quarterly).
    ylabel : str, optional
        Y-axis label (default: "Cumulative Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Similar to returns() but uses logarithmic scale which is better for visualizing
    exponential growth and making percentage changes more comparable across time.
    """
    # Build title with log scale indication
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += " vs %s (Log Scaled" % benchmark.upper()
        else:
            title += " vs %s (Log Scaled" % _benchmark_label(benchmark)
        if match_volatility:
            title += ", Volatility Matched"
    else:
        title += " (Log Scaled"
    title += ")"

    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Prepare benchmark data to match returns index
    benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)  # type: ignore

    # Use core plotting function with log scale enabled
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=True,
        resample=resample,
        compound=compound,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def daily_returns(
    returns: Returns,
    benchmark: Returns | str | None,
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 4),
    fontname: str = "Arial",
    lw: float = 0.5,
    log_scale: bool = False,
    ylabel: str = "Returns",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
    active: bool = False,
) -> _Figure | None:
    """
    Plot daily returns over time, optionally as active returns vs benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str
        Benchmark returns data or ticker symbol.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 4)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 0.5).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    ylabel : str, optional
        Y-axis label (default: "Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).
    active : bool, optional
        If True, plots active returns (returns - benchmark) (default: False).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows daily return variations over time. If active=True, displays the difference
    between portfolio returns and benchmark returns.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)
        # Calculate active returns if requested
        if active and benchmark is not None:
            benchmark = _get_utils()._prepare_returns(benchmark)
            returns = returns - benchmark

    # Set plot title based on active returns setting
    plot_title = "Daily Active Returns" if active else "Daily Returns"
    plot_title += " (Cumulative Sum)"

    # Use core plotting function for daily time series
    fig = _core.plot_timeseries(
        returns,
        None,  # No benchmark for daily returns plot
        plot_title,
        ylabel=ylabel,
        match_volatility=False,
        log_scale=log_scale,
        resample="D",  # Daily resampling
        compound=False,  # No compounding for daily returns
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def yearly_returns(
    returns: Returns,
    benchmark: Returns | str | None = None,
    fontname: str = "Arial",
    grayscale: bool = False,
    hlw: float = 1.5,
    hlcolor: str = "red",
    hllabel: str = "",
    match_volatility: bool = False,
    log_scale: bool = False,
    figsize: tuple[float, float] = (10, 5),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot end-of-year returns as a bar chart, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    hlw : float, optional
        Horizontal line width for mean line (default: 1.5).
    hlcolor : str, optional
        Color for horizontal mean line (default: "red").
    hllabel : str, optional
        Label for horizontal mean line (default: "").
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple sum (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Aggregates returns by year and displays as bars. Shows mean return as horizontal line.
    """
    # Set plot title
    title = "EOY Returns"
    if benchmark is not None:
        title += "  vs %s" % _benchmark_label(benchmark)
        # Prepare and resample benchmark data
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)
        benchmark = safe_resample(benchmark, "YE", _get_stats().comp)
        benchmark = safe_resample(benchmark, "YE", "last")

    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Resample returns to year-end based on compounding preference
    if compounded:
        returns = safe_resample(returns, "YE", _get_stats().comp)
    else:
        returns = safe_resample(returns, "YE", "sum")
    returns = safe_resample(returns, "YE", "last")

    # Use core plotting function for bar chart
    fig = _core.plot_returns_bars(
        returns,
        benchmark,
        fontname=fontname,
        hline=returns.mean(),  # Show mean as horizontal line
        hlw=hlw,
        hllabel=hllabel,
        hlcolor=hlcolor,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample="YE",
        title=title,
        figsize=figsize,
        grayscale=grayscale,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def distribution(
    returns: Returns,
    fontname: str = "Arial",
    grayscale: bool = False,
    ylabel: bool = True,
    figsize: tuple[float, float] = (10, 6),
    subtitle: bool = True,
    compounded: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    title: str | None = None,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot the distribution of returns using histogram and density curves.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    subtitle : bool, optional
        Whether to show subtitle with distribution statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    title : str, optional
        Custom title for the plot (default: None).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows the distribution of returns with histogram bars and overlaid density curve.
    Helpful for understanding return characteristics and identifying outliers.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Use core plotting function for distribution
    fig = _core.plot_distribution(
        returns,
        fontname=fontname,
        grayscale=grayscale,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        title=title,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def histogram(
    returns: Returns,
    benchmark: Returns | str | None = None,
    resample: str = "ME",
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 5),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot histogram of returns resampled to specified frequency.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    resample : str, optional
        Resampling frequency: 'W' for weekly, 'ME' for monthly, 'QE' for quarterly,
        'YE' for yearly (default: 'ME').
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with distribution statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Resamples returns to specified frequency and plots distribution histogram.
    Useful for analyzing return patterns at different time horizons.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)
        if benchmark is not None:
            benchmark = _get_utils()._prepare_returns(benchmark)

    # Determine title based on resampling frequency
    if resample == "W":
        title = "Weekly "
    elif resample == "ME":
        title = "Monthly "
    elif resample == "QE":
        title = "Quarterly "
    elif resample == "YE":
        title = "Annual "
    else:
        title = ""

    # Use core plotting function for histogram
    return _core.plot_histogram(
        returns,
        benchmark,
        resample=resample,
        grayscale=grayscale,
        fontname=fontname,
        title="Distribution of %sReturns" % title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )


def drawdown(
    returns: Returns,
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 5),
    fontname: str = "Arial",
    lw: float = 1,
    log_scale: bool = False,
    match_volatility: bool = False,
    compound: bool = False,
    ylabel: str = "Drawdown",
    resample: str | None = None,
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Plot drawdown series over time showing periods of loss from peak values.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for drawdown plot (default: 1).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    match_volatility : bool, optional
        Not used in drawdown plot (default: False).
    compound : bool, optional
        Not used in drawdown plot (default: False).
    ylabel : str, optional
        Y-axis label (default: "Drawdown").
    resample : str, optional
        Resampling frequency for data aggregation.
    subtitle : bool, optional
        Whether to show subtitle with drawdown statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows underwater plot of drawdowns with filled area. Includes average drawdown
    line as reference. Useful for understanding portfolio risk and recovery periods.
    """
    # Convert returns to drawdown series
    dd = _get_stats().to_drawdown_series(returns)

    # Use core plotting function for drawdown time series
    fig = _core.plot_timeseries(
        dd,
        title="Underwater Plot",
        hline=dd.mean(),  # Show average drawdown as horizontal line
        hlw=2,
        hllabel="Average",
        returns_label="Drawdown",
        compound=compound,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=resample,
        fill=True,  # Fill area under drawdown curve
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
        raw_data=True,  # Skip cumulative transformation for drawdown data
    )
    if not show:
        return fig


def drawdowns_periods(
    returns: Returns,
    periods: int = 5,
    lw: float = 1.5,
    log_scale: bool = False,
    fontname: str = "Arial",
    grayscale: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot the longest drawdown periods as separate lines for detailed analysis.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    periods : int, optional
        Number of longest drawdown periods to display (default: 5).
    lw : float, optional
        Line width for drawdown lines (default: 1.5).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    title : str, optional
        Custom title for the plot (default: None).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with drawdown statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Identifies and plots the longest drawdown periods separately. Each period is
    shown as a different colored line for easy comparison of severity and duration.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Use core plotting function for longest drawdown periods
    fig = _core.plot_longest_drawdowns(
        returns,
        periods=periods,
        lw=lw,
        log_scale=log_scale,
        fontname=fontname,
        grayscale=grayscale,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_beta(
    returns: Returns,
    benchmark: Returns | str,
    window1: int = 126,
    window1_label: str = "6-Months",
    window2: int = 252,
    window2_label: str = "12-Months",
    lw: float = 1.5,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: bool = True,
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    prepare_returns: bool = True,
) -> _Figure | None:
    """
    Plot rolling beta coefficients over time using multiple window sizes.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str
        Benchmark returns data or ticker symbol.
    window1 : int, optional
        First rolling window size in days (default: 126).
    window1_label : str, optional
        Label for first window (default: "6-Months").
    window2 : int, optional
        Second rolling window size in days (default: 252).
    window2_label : str, optional
        Label for second window (default: "12-Months").
    lw : float, optional
        Line width for beta lines (default: 1.5).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with beta statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows how portfolio beta (systematic risk) changes over time relative to benchmark.
    Uses two different window sizes to show short-term and long-term beta trends.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Prepare benchmark data to match returns index
    benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)  # type: ignore

    # Use core plotting function for rolling beta
    fig = _core.plot_rolling_beta(
        returns,
        benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title="Rolling Beta to Benchmark",
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_volatility(
    returns: Returns,
    benchmark: Returns | str | None = None,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = 252,
    lw: float = 1.5,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Volatility",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Plot rolling volatility over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for volatility lines (default: 1.5).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Volatility").
    subtitle : bool, optional
        Whether to show subtitle with volatility statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling volatility (standard deviation) over time. Includes mean volatility
    as horizontal reference line. Useful for understanding risk patterns over time.
    """
    # Calculate rolling volatility for returns
    returns = _get_stats().rolling_volatility(returns, period, periods_per_year)

    # Calculate rolling volatility for benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)
        benchmark = _get_stats().rolling_volatility(
            benchmark, period, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean volatility as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Volatility (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sharpe(
    returns: Returns,
    benchmark: Returns | str | None = None,
    rf: float = 0.0,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = 252,
    lw: float = 1.25,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Sharpe",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Plot rolling Sharpe ratio over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    rf : float, optional
        Risk-free rate for Sharpe calculation (default: 0.0).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for Sharpe lines (default: 1.25).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Sharpe").
    subtitle : bool, optional
        Whether to show subtitle with Sharpe statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling Sharpe ratio (risk-adjusted returns) over time. Higher values
    indicate better risk-adjusted performance. Includes mean Sharpe as reference.
    """
    # Calculate rolling Sharpe ratio for returns
    returns = _get_stats().rolling_sharpe(
        returns,
        rf,
        period,
        True,  # prepare_returns
        periods_per_year,
    )

    # Calculate rolling Sharpe ratio for benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _get_stats().rolling_sharpe(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean Sharpe as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sharpe (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sortino(
    returns: Returns,
    benchmark: Returns | str | None = None,
    rf: float = 0.0,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = 252,
    lw: float = 1.25,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Sortino",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Plot rolling Sortino ratio over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    rf : float, optional
        Risk-free rate for Sortino calculation (default: 0.0).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for Sortino lines (default: 1.25).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Sortino").
    subtitle : bool, optional
        Whether to show subtitle with Sortino statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling Sortino ratio (downside deviation adjusted returns) over time.
    Similar to Sharpe but only considers downside volatility. Higher values indicate
    better downside-adjusted performance.
    """
    # Calculate rolling Sortino ratio for returns
    returns = _get_stats().rolling_sortino(returns, rf, period, True, periods_per_year)

    # Calculate rolling Sortino ratio for benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _get_stats().rolling_sortino(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean Sortino as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sortino (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def monthly_heatmap(
    returns: Returns,
    benchmark: Returns | str | None = None,
    annot_size: int = 10,
    figsize: tuple[float, float] = (8, 5),
    cbar: bool = True,
    square: bool = False,
    returns_label: str = "Strategy",
    compounded: bool = True,
    eoy: bool = False,
    grayscale: bool = False,
    fontname: str = "Arial",
    ylabel: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    active: bool = False,
) -> _Figure | None:
    """
    Create a heatmap of monthly returns showing performance across years and months.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    annot_size : int, optional
        Font size for annotations in heatmap cells (default: 10).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (8, 5)).
    cbar : bool, optional
        Whether to show color bar (default: True).
    square : bool, optional
        Whether to make heatmap cells square (default: False).
    returns_label : str, optional
        Label for the returns series (default: "Strategy").
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    eoy : bool, optional
        Whether to include end-of-year column (default: False).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    active : bool, optional
        If True, shows active returns (returns - benchmark) (default: False).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Creates a color-coded heatmap where each cell represents a month's performance.
    Green indicates positive returns, red indicates negative returns. Useful for
    identifying seasonal patterns and performance consistency.
    """
    fontname = _core._resolve_font(fontname)

    # colors, ls, alpha = _core._get_colors(grayscale)
    # Select color map based on grayscale preference
    cmap = "gray" if grayscale else "RdYlGn"

    # Convert to monthly returns and convert to percentage
    returns = _get_stats().monthly_returns(returns, eoy=eoy, compounded=compounded) * 100
    returns = returns.replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)

    # Calculate figure height based on number of years
    fig_height = len(returns) / 2.5

    # Use current figure size if not specified
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    # Adjust figure size based on data and color bar
    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.051, max([fig_height, figsize[1]]))

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)

    # Remove spines for cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # _sns.set(font_scale=.9)
    # Create heatmap for active returns vs benchmark
    if active and benchmark is not None:
        ax.set_title(
            f"{returns_label} - Monthly Active Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        # Calculate benchmark monthly returns
        benchmark = (
            _get_stats().monthly_returns(benchmark, eoy=eoy, compounded=compounded) * 100
        )
        benchmark = benchmark.replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
        # Calculate active returns (strategy - benchmark)
        active_returns = (returns - benchmark).replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
        annot = _heatmap_annot_frame(active_returns)

        # Create heatmap with active returns
        ax = _sns.heatmap(
            active_returns,
            ax=ax,
            annot=annot,
            center=0,  # Center colormap at zero
            annot_kws={"size": annot_size, "weight": "bold"},
            fmt="",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    else:
        # Create standard monthly returns heatmap
        ax.set_title(
            f"{returns_label} - Monthly Returns (%)\n",
            fontsize=12,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )

        annot = _heatmap_annot_frame(returns)

        # Create heatmap with monthly returns
        ax = _sns.heatmap(
            returns,
            ax=ax,
            annot=annot,
            center=0,  # Center colormap at zero
            annot_kws={"size": annot_size, "weight": "bold"},
            fmt="",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )

    _style_heatmap_annotations(ax)

    # Format color bar if present
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=annot_size)

    # Set y-axis label
    if ylabel:
        ax.set_ylabel("Years", fontname=fontname, fontweight="bold", fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format tick labels
    ax.tick_params(colors="#808080")
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def _heatmap_annot_frame(data: _pd.DataFrame) -> _pd.DataFrame:
    """为热力图生成显式注释文本，保证 0.00 等值也会被写进格子。"""
    return data.map(_heatmap_annot_value) if hasattr(data, "map") else data.applymap(_heatmap_annot_value)


def _heatmap_annot_value(value) -> str:
    numeric = _pd.to_numeric(_pd.Series([value]), errors="coerce").iloc[0]
    if _pd.isna(numeric) or not _np.isfinite(float(numeric)):
        return ""
    return f"{float(numeric):.2f}"


def _style_heatmap_annotations(ax) -> None:
    """根据格子底色自动切换注释文字颜色，避免深绿/深红里数字看不见。"""
    if not getattr(ax, "collections", None):
        return
    facecolors = ax.collections[0].get_facecolors()
    for text, rgba in zip(ax.texts, facecolors):
        red, green, blue = rgba[:3]
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text.set_color("#111827" if luminance >= 0.58 else "#FFFFFF")
        text.set_fontweight("bold")


def monthly_returns(
    returns: Returns,
    annot_size: int = 9,
    figsize: tuple[float, float] = (10, 5),
    cbar: bool = True,
    square: bool = False,
    compounded: bool = True,
    eoy: bool = False,
    grayscale: bool = False,
    fontname: str = "Arial",
    ylabel: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> _Figure | None:
    """
    Create a heatmap of monthly returns (wrapper function for monthly_heatmap).

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    annot_size : int, optional
        Font size for annotations in heatmap cells (default: 9).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    cbar : bool, optional
        Whether to show color bar (default: True).
    square : bool, optional
        Whether to make heatmap cells square (default: False).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    eoy : bool, optional
        Whether to include end-of-year column (default: False).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    This is a convenience wrapper around monthly_heatmap() with commonly used
    default parameters for displaying monthly returns.
    """
    # Call monthly_heatmap with provided parameters
    return monthly_heatmap(
        returns=returns,
        annot_size=annot_size,
        figsize=figsize,
        cbar=cbar,
        square=square,
        compounded=compounded,
        eoy=eoy,
        grayscale=grayscale,
        fontname=fontname,
        ylabel=ylabel,
        savefig=savefig,
        show=show,
    )
