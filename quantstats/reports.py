#!/usr/bin/env python
#
# QuantStats: 量化投资者的投资组合分析工具
# https://github.com/ranaroussi/quantstats
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

import pandas as _pd
import numpy as _np
from math import sqrt as _sqrt, ceil as _ceil
from datetime import datetime as _dt
from base64 import b64encode as _b64encode
import re as _regex
from tabulate import tabulate as _tabulate
from . import __version__
from ._zh import zh_columns as _zh_columns
from ._zh import zh_html as _zh_html
from ._zh import zh_index as _zh_index
from ._zh import zh_metric as _zh_metric
from ._zh import zh_text as _zh_text
from utils.log_kit import logger

# 延迟导入以避免包初始化期间的循环依赖
_stats = None
_utils = None
_plots = None


def _get_stats():
    """获取 stats 模块的延迟导入。"""
    global _stats
    if _stats is None:
        from . import stats
        _stats = stats
    return _stats


def _get_utils():
    """获取 utils 模块的延迟导入。"""
    global _utils
    if _utils is None:
        from . import utils
        _utils = utils
    return _utils


def _get_plots():
    """获取 plots 模块的延迟导入。"""
    global _plots
    if _plots is None:
        from . import plots
        _plots = plots
    return _plots
from dateutil.relativedelta import relativedelta
from io import StringIO
from pathlib import Path
import tempfile
import webbrowser

try:
    from IPython.display import display as iDisplay, HTML as iHTML # type: ignore
except ImportError:
    pass  # IPython 不可用，显示函数将不会使用


def _get_trading_periods(periods_per_year=252):
    """
    根据年化周期数计算全年和半年的滚动窗口长度。

    此辅助函数计算全年和半年期间的交易周期数，
    这些在金融计算中常用于年化和滚动窗口分析。

    参数
    ----------
    periods_per_year : int, default 252
        一年中的交易周期数（例如，日数据为 252，月数据为 12）

    返回
    -------
    tuple
        包含 (periods_per_year, half_year_periods) 的元组

    示例
    --------
    >>> _get_trading_periods(252)  # 日数据
    (252, 126)
    >>> _get_trading_periods(12)   # 月数据
    (12, 6)
    """
    # 使用向上取整计算半年周期数以确保至少达到一半
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _print_parameters_table(
    benchmark_title=None,
    periods_per_year=252,
    rf=0.0,
    compounded=True,
    match_dates=True,
):
    """
    用中文输出报告参数表，统一替代原版英文 print 输出。

    参数
    ----------
    benchmark_title : str or None
        基准名称/代码
    periods_per_year : int
        每年交易周期数
    rf : float
        无风险利率
    compounded : bool
        是否复合收益率
    match_dates : bool
        是否与基准对齐日期
    """
    width = 40
    logger.debug("=" * width)
    logger.debug("                 参数")
    logger.debug("-" * width)
    if benchmark_title:
        logger.debug(f"{'基准':<25}{str(benchmark_title).upper():>15}")
    logger.debug(f"{'年化周期数':<25}{periods_per_year:>15}")
    logger.debug(f"{'无风险利率':<25}{rf:>14.1%}")
    logger.debug(f"{'复利计算':<25}{'是' if compounded else '否':>15}")
    if benchmark_title:
        logger.debug(f"{'对齐日期':<25}{'是' if match_dates else '否':>15}")
    logger.debug("=" * width)
    logger.debug("")


def _match_dates(returns, benchmark):
    """
    把策略和基准裁剪到共同起点，确保后续绩效比较口径一致。

    此函数确保收益率和基准序列从同一日期开始，
    通过找到两个序列都有非零值的最晚开始日期。
    这对于准确的绩效比较至关重要。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据，可以是 Series 或包含多列的 DataFrame
    benchmark : pd.Series
        基准收益率数据

    返回
    -------
    tuple
        包含 (aligned_returns, aligned_benchmark) 的元组，两者从同一日期开始

    示例
    --------
    >>> returns_aligned, bench_aligned = _match_dates(returns, benchmark)
    """
    # 处理不同类型的收益率数据（Series vs DataFrame）
    if isinstance(returns, _pd.DataFrame):
        # 对于 DataFrame，使用第一列找到开始日期
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        # 对于 Series，找到两个序列开始日期的最大值
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())

    # 将两个序列切片到最新的共同开始日期
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark


def html(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    title="策略绩效报告",
    output=None,
    compounded=True,
    periods_per_year=252,
    download_filename="quantstats-绩效报告.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    **kwargs,
):
    """
    生成投资组合绩效分析的 HTML  tearsheet 报告。

    本函数创建包含绩效指标、可视化图表和投资收益率分析的全面 HTML 报告。
    报告包括与基准的比较、回撤分析以及各种绩效图表。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        策略/投资组合的日收益率数据
    benchmark : pd.Series, str, or None, default None
        用于比较的基准收益率。可以是收益率 Series、代码字符串或 None（无基准）
    rf : float, default 0.0
        计算用的无风险利率（以小数表示，例如 0.02 表示 2%）
    grayscale : bool, default False
        是否生成灰度图表而不是彩色
    title : str, default "策略绩效报告"
        HTML 报告顶部显示的标题
    output : str or None, default None
        保存 HTML 报告的文件路径。如果为 None，则在浏览器中下载
    compounded : bool, default True
        是否复合收益率进行计算
    periods_per_year : int, default 252
        年化的每年交易周期数
    download_filename : str, default "quantstats-绩效报告.html"
        如果 output 为 None 时浏览器下载的文件名
    figfmt : str, default "svg"
        嵌入图表的格式（'svg'、'png'、'jpg'）
    template_path : str or None, default None
        自定义 HTML 模板文件的路径。如果为 None 则使用默认模板
    match_dates : bool, default True
        是否对齐收益率和基准的开始日期
    **kwargs
        用于自定义的其他关键字参数：
        - strategy_title: 策略的自定义名称
        - benchmark_title: 基准的自定义名称
        - active_returns: 是否显示相对于基准的主动收益率

    返回
    -------
    None
        生成 HTML 文件，可以下载或保存到指定路径

    示例
    --------
    >>> html(returns, benchmark='^GSPC', title='我的策略')
    >>> html(returns, output='report.html', grayscale=True)

    异常
    ------
    FileNotFoundError
        如果自定义 template_path 不存在
    """
    _get_utils().require_local_returns(returns, "returns")
    if benchmark is not None:
        _get_utils().require_local_returns(benchmark, "benchmark")

    # Clean returns data by removing NaN values if date matching is enabled
    if match_dates:
        returns = returns.dropna()

    # Get trading periods for calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Secure file path handling for HTML template
    if template_path is None:
        # Use default template path - report.html in same directory
        template_path = Path(__file__).parent / 'report.html'
    else:
        template_path = Path(template_path)

    # Resolve to absolute path and validate template file existence
    template_path = template_path.resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")

    # Read template securely with UTF-8 encoding
    tpl = template_path.read_text(encoding='utf-8')

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data for analysis
    returns = _get_utils()._prepare_returns(returns)

    # Handle strategy title - can be single string or list for multiple columns
    strategy_title = kwargs.get("strategy_title", "策略")
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "基准")
        # Auto-determine benchmark title if not provided
        if kwargs.get("benchmark_title") is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name if benchmark.name else "基准"
            elif isinstance(benchmark, _pd.DataFrame):
                col_name = benchmark[benchmark.columns[0]].name
                benchmark_title = col_name if col_name else "基准"

        # Ensure benchmark_title is a string for .upper() call
        if benchmark_title is None:
            benchmark_title = "基准"
        # Store original benchmark before any alignment for accurate EOY calculations
        # This preserves the full benchmark data including non-trading days
        if isinstance(benchmark, str):
            _get_utils().require_local_returns(benchmark, "benchmark")
        elif isinstance(benchmark, _pd.Series):
            benchmark_original = benchmark.copy()
        else:
            benchmark_original = benchmark
        # Prepare benchmark data to match returns index and risk-free rate
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        # Align dates between returns and benchmark if requested
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
    else:
        benchmark_title = None
        benchmark_original = None

    # Format date range for display in template
    date_range = returns.index.strftime("%e %b, %Y")
    tpl = tpl.replace("{{date_range}}", date_range[0] + " - " + date_range[-1])

    # Build title with compounding indicator (only show if compounded)
    full_title = f"{title}（复利）" if compounded else title
    tpl = tpl.replace("{{title}}", full_title)
    tpl = tpl.replace("{{v}}", __version__)

    # Build parameters string for subtitle
    params_parts = []

    # Add user-provided parameters first if present
    user_params = kwargs.get("parameters", {})
    if user_params:
        for key, value in user_params.items():
            params_parts.append(f"{key}: {value}")

    # Add auto-detected parameters (always show key params)
    if benchmark_title:
        params_parts.append(f"基准: {str(benchmark_title).upper()}")
    params_parts.append(f"年化周期数: {periods_per_year}")
    params_parts.append(f"无风险利率: {rf:.1%}")

    params_str = " &bull; ".join(params_parts)
    if params_str:
        params_str += " | "
    tpl = tpl.replace("{{params}}", params_str)

    # Add matched dates indicator
    matched_dates_str = "（已对齐日期）" if match_dates and benchmark is not None else ""
    tpl = tpl.replace("{{matched_dates}}", matched_dates_str)

    # Set names for data series to be used in charts and tables
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    # Generate comprehensive performance metrics table
    mtrx = metrics(
        returns=returns,
        benchmark=benchmark,
        rf=rf,
        display=False,
        mode="full",
        sep=True,
        internal="True",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
    )[2:]

    # Format metrics table for HTML display
    mtrx.index.name = "指标"
    tpl = tpl.replace("{{metrics}}", _html_table(_zh_index(_zh_columns(mtrx))))

    # Handle table formatting for multiple columns
    if isinstance(returns, _pd.DataFrame):
        num_cols = len(returns.columns)
        # Replace empty table rows with horizontal rule separators
        for i in reversed(range(num_cols + 1, num_cols + 3)):
            str_td = "<td></td>" * i
            tpl = tpl.replace(
                f"<tr>{str_td}</tr>", '<tr><td colspan="{}"><hr></td></tr>'.format(i)
            )

    # Clean up table formatting with horizontal rules
    tpl = tpl.replace(
        "<tr><td></td><td></td><td></td></tr>", '<tr><td colspan="3"><hr></td></tr>'
    )
    tpl = tpl.replace(
        "<tr><td></td><td></td></tr>", '<tr><td colspan="2"><hr></td></tr>'
    )

    # Generate end-of-year (EOY) returns comparison table
    if benchmark is not None:
        # Use original benchmark for EOY comparison to preserve accurate yearly returns
        # This prevents loss of benchmark returns on non-trading days
        benchmark_for_eoy = benchmark_original if benchmark_original is not None else benchmark
        yoy = _get_stats().compare(
            returns, benchmark_for_eoy, "YE", compounded=compounded, prepare_returns=False
        )
        # Set appropriate column names based on data type
        if isinstance(returns, _pd.Series):
            yoy.columns = [benchmark_title, strategy_title, "倍数", "胜出"]
        elif isinstance(returns, _pd.DataFrame):
            yoy.columns = list(
                _pd.core.common.flatten([benchmark_title, strategy_title])
            )
        yoy.index.name = "年份"
        tpl = tpl.replace("{{eoy_title}}", "<h3>年度收益 vs 基准</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(_zh_columns(yoy)))
    else:
        # Generate EOY returns table without benchmark comparison
        # pct multiplier
        yoy = _pd.DataFrame(_get_utils().group_returns(returns, returns.index.year) * 100)
        if isinstance(returns, _pd.Series):
            yoy.columns = ["收益"]
            yoy["累计"] = _get_utils().group_returns(returns, returns.index.year, True) * 100
            # Don't add "%" here - the CSS in report.html handles it via :after pseudo-element
            # Adding "%" in Python causes double "%" display (bug #475)
        elif isinstance(returns, _pd.DataFrame):
            # Don't show cumulative for multiple strategy portfolios
            # just show compounded like when we have a benchmark
            yoy.columns = list(_pd.core.common.flatten(strategy_title))

        yoy.index.name = "年份"
        tpl = tpl.replace("{{eoy_title}}", "<h3>年度收益</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(_zh_columns(yoy)))

    # Generate drawdown analysis table
    if isinstance(returns, _pd.Series):
        # Calculate drawdown series and get worst drawdown periods
        dd = _get_stats().to_drawdown_series(returns)
        dd_info = _get_stats().drawdown_details(dd).sort_values(
            by="max drawdown", ascending=True
        )[:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["开始", "修复", "回撤", "天数"]
        tpl = tpl.replace("{{dd_info}}", _html_table(dd_info, False))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        dd_info_list = []
        for col in returns.columns:
            dd = _get_stats().to_drawdown_series(returns[col])
            dd_info = _get_stats().drawdown_details(dd).sort_values(
                by="max drawdown", ascending=True
            )[:10]
            dd_info = dd_info[["start", "end", "max drawdown", "days"]]
            dd_info.columns = ["开始", "修复", "回撤", "天数"]
            dd_info_list.append(_html_table(dd_info, False))

        # Combine all drawdown tables with headers
        dd_html_table = ""
        for html_str, col in zip(dd_info_list, returns.columns):
            dd_html_table = (
                dd_html_table + f"<h3>{col}</h3><br>" + StringIO(html_str).read()
            )
        tpl = tpl.replace("{{dd_info}}", dd_html_table)

    # Get active returns setting for plots
    active = kwargs.get("active_returns", False)

    # Generate all the performance plots and embed them in the HTML
    # plots
    figfile = _get_utils()._file_stream()
    _get_plots().returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 5),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{returns}}", _embed_figure(figfile, figfmt))

    # Log returns plot for better visualization of performance
    figfile = _get_utils()._file_stream()
    _get_plots().log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{log_returns}}", _embed_figure(figfile, figfmt))

    # Volatility-matched returns plot (only if benchmark exists)
    if benchmark is not None:
        figfile = _get_utils()._file_stream()
        _get_plots().returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compound=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{vol_returns}}", _embed_figure(figfile, figfmt))

    # Yearly returns comparison chart
    figfile = _get_utils()._file_stream()
    _get_plots().yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{eoy_returns}}", _embed_figure(figfile, figfmt))

    # Returns distribution histogram
    figfile = _get_utils()._file_stream()
    _get_plots().histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(7, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{monthly_dist}}", _embed_figure(figfile, figfmt))

    # Daily returns scatter plot
    figfile = _get_utils()._file_stream()
    _get_plots().daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        prepare_returns=False,
        active=active,
    )
    tpl = tpl.replace("{{daily_returns}}", _embed_figure(figfile, figfmt))

    # Rolling beta analysis (only if benchmark exists)
    if benchmark is not None:
        figfile = _get_utils()._file_stream()
        _get_plots().rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 3),
            subtitle=False,
            window1=win_half_year,
            window2=win_year,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            prepare_returns=False,
        )
        tpl = tpl.replace("{{rolling_beta}}", _embed_figure(figfile, figfmt))

    # Rolling volatility analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_vol}}", _embed_figure(figfile, figfmt))

    # Rolling Sharpe ratio analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sharpe}}", _embed_figure(figfile, figfmt))

    # Rolling Sortino ratio analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sortino}}", _embed_figure(figfile, figfmt))

    # Drawdown periods analysis
    figfile = _get_utils()._file_stream()
    if isinstance(returns, _pd.Series):
        _get_plots().drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(embed, figfmt))

    # Underwater (drawdown) plot
    figfile = _get_utils()._file_stream()
    _get_plots().drawdown(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
    )
    tpl = tpl.replace("{{dd_plot}}", _embed_figure(figfile, figfmt))

    # Monthly returns heatmap
    figfile = _get_utils()._file_stream()
    if isinstance(returns, _pd.Series):
        _get_plots().monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 4),
            cbar=False,
            returns_label=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            active=active,
        )
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(8, 4),
                cbar=False,
                returns_label=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                active=active,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(embed, figfmt))

    # Returns distribution analysis
    figfile = _get_utils()._file_stream()

    if isinstance(returns, _pd.Series):
        _get_plots().distribution(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(embed, figfmt))

    # Clean up any remaining template placeholders
    tpl = _regex.sub(r"\{\{(.*?)\}\}", "", tpl)
    tpl = tpl.replace("white-space:pre;", "")
    tpl = _zh_html(tpl)

    # Handle output - either download in browser or save to file
    if output is None:
        if _get_utils()._in_notebook():
            _download_html(tpl, download_filename)
        else:
            # Save to temp file and open in browser
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(tpl)
                temp_path = f.name
            webbrowser.open("file://" + temp_path)
        return

    # Write HTML content to specified output file
    with open(output, "w", encoding="utf-8") as f:
        f.write(tpl)


def full(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    生成全面的绩效分析报告。

    本函数创建包含指标、最差回撤分析和完整可视化套件的全面绩效分析。
    它专为详细的投资组合分析而设计，可处理单个策略和多个策略的比较。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        策略/投资组合的日收益率数据
    benchmark : pd.Series, str, or None, default None
        用于比较的基准收益率
    rf : float, default 0.0
        计算用的无风险利率（以小数表示）
    grayscale : bool, default False
        是否生成灰度图表
    figsize : tuple, default (8, 5)
        图表的尺寸，为 (宽度, 高度)
    display : bool, default True
        是否在 notebook/控制台显示结果
    compounded : bool, default True
        是否复合收益率进行计算
    periods_per_year : int, default 252
        每年交易周期数
    match_dates : bool, default True
        是否对齐收益率和基准的开始日期
    **kwargs
        其他关键字参数：
        - strategy_title: 策略的自定义名称
        - benchmark_title: 基准的自定义名称
        - active_returns: 是否显示相对于基准的主动收益率

    返回
    -------
    None
        显示包含指标、回撤和图表的全面分析

    示例
    --------
    >>> full(returns, benchmark='^GSPC', rf=0.02)
    >>> full(returns, figsize=(10, 6), grayscale=True)
    """
    _get_utils().require_local_returns(returns, "returns")
    if benchmark is not None:
        _get_utils().require_local_returns(benchmark, "benchmark")

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _get_utils()._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "基准")
    strategy_title = kwargs.get("strategy_title", "策略")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Set names for display purposes
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    # Calculate drawdown analysis for worst periods display
    dd = _get_stats().to_drawdown_series(returns)

    # Process drawdown details based on data type
    if isinstance(dd, _pd.Series):
        col = _get_stats().drawdown_details(dd).columns[4]
        dd_info = _get_stats().drawdown_details(dd).sort_values(by=col, ascending=True)[:5]
        if not dd_info.empty:
            dd_info.index = range(1, min(6, len(dd_info) + 1))
            dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    elif isinstance(dd, _pd.DataFrame):
        # Handle multiple strategy columns
        col = _get_stats().drawdown_details(dd).columns.get_level_values(1)[4]
        dd_info_dict = {}
        for ptf in dd.columns:
            dd_info = _get_stats().drawdown_details(dd[ptf]).sort_values(
                by=col, ascending=True
            )[:5]
            if not dd_info.empty:
                dd_info.index = range(1, min(6, len(dd_info) + 1))
                dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
            dd_info_dict[ptf] = dd_info

    # Display results based on environment (notebook vs console)
    if _get_utils()._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>绩效指标</h4>"))
        iDisplay(
            metrics(
                returns=returns,
                benchmark=benchmark,
                rf=rf,
                display=display,
                mode="full",
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False,
                benchmark_title=benchmark_title,
                strategy_title=strategy_title,
            )
        )

        # Display worst drawdowns analysis
        if isinstance(dd, _pd.Series):
            iDisplay(iHTML('<h4 style="margin-bottom:20px">最差 5 次回撤</h4>'))
            if dd_info.empty:
                iDisplay(iHTML("<p>无回撤</p>"))
            else:
                iDisplay(_zh_columns(dd_info))
        elif isinstance(dd, _pd.DataFrame):
            # Display drawdowns for each strategy
            for ptf, dd_info in dd_info_dict.items():
                iDisplay(
                    iHTML(
                        '<h4 style="margin-bottom:20px">%s - 最差 5 次回撤</h4>'
                        % ptf
                    )
                )
                if dd_info.empty:
                    iDisplay(iHTML("<p>无回撤</p>"))
                else:
                    iDisplay(_zh_columns(dd_info))

        iDisplay(iHTML("<h4>策略图表</h4>"))
    else:
        # Display in console/terminal environment
        _print_parameters_table(
            benchmark_title=benchmark_title,
            periods_per_year=periods_per_year,
            rf=rf,
            compounded=compounded,
            match_dates=match_dates,
        )
        logger.debug("[绩效指标]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="full",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        logger.debug("\n\n")
        logger.debug("[最差 5 次回撤]\n")

        # Display drawdowns in tabular format
        if isinstance(dd, _pd.Series):
            if dd_info.empty:
                logger.debug("无回撤")
            else:
                logger.debug(
                    _tabulate(
                        _zh_columns(dd_info), headers="keys", tablefmt="simple", floatfmt=".2f"
                    )
                )
        elif isinstance(dd, _pd.DataFrame):
            for ptf, dd_info in dd_info_dict.items():
                if dd_info.empty:
                    logger.debug("无回撤")
                else:
                    logger.debug(f"{ptf}\n")
                    logger.debug(
                        _tabulate(
                            _zh_columns(dd_info), headers="keys", tablefmt="simple", floatfmt=".2f"
                        )
                    )

        logger.debug("\n\n")
        logger.debug("[策略图表]\n通过 Matplotlib 生成")

    # Generate comprehensive plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def basic(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    生成基本绩效分析报告。

    本函数创建包含基本指标和基本可视化的简化绩效分析。
    它专为不需要详细分析时的快速投资组合分析而设计。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        策略/投资组合的日收益率数据
    benchmark : pd.Series, str, or None, default None
        用于比较的基准收益率
    rf : float, default 0.0
        计算用的无风险利率（以小数表示）
    grayscale : bool, default False
        是否生成灰度图表
    figsize : tuple, default (8, 5)
        图表的尺寸，为 (宽度, 高度)
    display : bool, default True
        是否在 notebook/控制台显示结果
    compounded : bool, default True
        是否复合收益率进行计算
    periods_per_year : int, default 252
        每年交易周期数
    match_dates : bool, default True
        是否对齐收益率和基准的开始日期
    **kwargs
        其他关键字参数：
        - strategy_title: 策略的自定义名称
        - benchmark_title: 基准的自定义名称
        - active_returns: 是否显示相对于基准的主动收益率

    返回
    -------
    None
        显示包含基本指标和图表的分析

    示例
    --------
    >>> basic(returns, benchmark='^GSPC')
    >>> basic(returns, figsize=(10, 6), display=False)
    """
    _get_utils().require_local_returns(returns, "returns")
    if benchmark is not None:
        _get_utils().require_local_returns(benchmark, "benchmark")

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _get_utils()._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "基准")
    strategy_title = kwargs.get("strategy_title", "策略")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Display results based on environment (notebook vs console)
    if _get_utils()._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>绩效指标</h4>"))
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        iDisplay(iHTML("<h4>策略图表</h4>"))
    else:
        # Display in console/terminal environment
        _print_parameters_table(
            benchmark_title=benchmark_title,
            periods_per_year=periods_per_year,
            rf=rf,
            compounded=compounded,
            match_dates=match_dates,
        )
        logger.debug("[绩效指标]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )

        logger.debug("\n\n")
        logger.debug("[策略图表]\n通过 Matplotlib 生成")

    # Generate basic plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="basic",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    计算投资组合分析的全面绩效指标。

    本函数计算包括收益率、风险度量、比率和统计度量在内的广泛绩效指标。
    它可以处理单个策略和带有可选基准分析的多策略比较。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        策略/投资组合的日收益率数据
    benchmark : pd.Series, str, or None, default None
        用于比较的基准收益率
    rf : float, default 0.0
        计算用的无风险利率（以小数表示）
    display : bool, default True
        是否以格式化表格显示结果
    mode : str, default "basic"
        分析模式 - "basic" 用于基本指标，"full" 用于全面分析
    sep : bool, default False
        是否在输出中包含分隔符行
    compounded : bool, default True
        是否复合收益率进行计算
    periods_per_year : int, default 252
        每年交易周期数
    prepare_returns : bool, default True
        是否准备/清理收益率数据
    match_dates : bool, default True
        是否对齐收益率和基准的开始日期
    **kwargs
        其他关键字参数：
        - strategy_title: 策略的自定义名称
        - benchmark_title: 基准的自定义名称
        - as_pct: 是否返回百分比
        - internal: 内部计算标志

    返回
    -------
    pd.DataFrame or None
        如果 display=False 则返回带有绩效指标的 DataFrame，否则返回 None

    示例
    --------
    >>> metrics_df = metrics(returns, benchmark='^GSPC', display=False)
    >>> metrics(returns, mode="full", rf=0.02)
    """
    _get_utils().require_local_returns(returns, "returns")
    if benchmark is not None:
        _get_utils().require_local_returns(benchmark, "benchmark")

    # Clean returns data if date matching is enabled
    if match_dates:
        returns = returns.dropna()
    # Remove timezone information from index for consistent processing
    returns.index = returns.index.tz_localize(None)

    # Get trading periods for annualization calculations
    win_year, _ = _get_trading_periods(periods_per_year)

    # Extract column names from kwargs or use defaults
    benchmark_colname = kwargs.get("benchmark_title", "基准")
    strategy_colname = kwargs.get("strategy_title", "策略")

    # Handle benchmark column naming
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"基准 ({benchmark.upper()})"
        elif isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    # Handle strategy column naming for multiple strategies
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    # if isinstance(returns, _pd.DataFrame):
    #     if len(returns.columns) > 1:
    #         raise ValueError("`returns` needs to be a Pandas Series or one column DataFrame. "
    #                          "multi colums DataFrame was passed")
    #     returns = returns[returns.columns[0]]

    # Prepare returns data if requested
    if prepare_returns:
        df = _get_utils()._prepare_returns(returns)

    # Create main DataFrame for calculations
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame({"returns": returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
            # Truncate df to the aligned date range to exclude leading zeros
            df = df.loc[returns.index]
        df["benchmark"] = benchmark
        # Update blank list for proper formatting
        if isinstance(returns, _pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, _pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    # Calculate start and end dates for each series
    if isinstance(returns, _pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, _pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    # Add benchmark dates if present
    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    # Fill missing values with zeros for calculations
    df = df.fillna(0)

    # Determine percentage multiplier for display
    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # Initialize metrics DataFrame with basic information
    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate %"] = _pd.Series(s_rf) * 100
    metrics["Time in Market %"] = _get_stats().exposure(df, prepare_returns=False) * pct

    # Add separator row
    metrics["~"] = blank

    # Calculate return metrics based on compounding preference
    if compounded:
        metrics["Cumulative Return %"] = (_get_stats().comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    # Calculate annualized return (CAGR)
    metrics["CAGR﹪%"] = _get_stats().cagr(df, rf, compounded, win_year) * pct

    # Add separator row
    metrics["~~~~~~~~~~~~~~"] = blank

    # Calculate risk-adjusted return ratios
    metrics["Sharpe"] = _get_stats().sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _get_stats().probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )

    # Add advanced Sharpe metrics for full mode
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _get_stats().smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _get_stats().probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    # Calculate Sortino ratio (downside deviation-based)
    metrics["Sortino"] = _get_stats().sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _get_stats().probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _get_stats().smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _get_stats().probabilistic_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate adjusted Sortino ratio
    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        # metrics['Prob. Sortino/√2 Ratio %'] = _get_stats().probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _get_stats().probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate Omega ratio (probability-weighted ratio)
    if isinstance(returns, _pd.Series):
        if "benchmark" in df:
            metrics["Omega"] = [
                _get_stats().omega(df["returns"], rf, 0.0, win_year),
                _get_stats().omega(df["benchmark"], rf, 0.0, win_year),
            ]
        else:
            metrics["Omega"] = _get_stats().omega(df["returns"], rf, 0.0, win_year)
    elif isinstance(returns, _pd.DataFrame):
        omega_values = [
            _get_stats().omega(df[strategy_col], rf, 0.0, win_year)
            for strategy_col in df_strategy_columns
        ]
        if "benchmark" in df:
            omega_values.append(_get_stats().omega(df["benchmark"], rf, 0.0, win_year))
        metrics["Omega"] = omega_values

    # Add separator and prepare for drawdown metrics
    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Max DD Date"] = blank
    metrics["Max DD Period Start"] = blank
    metrics["Max DD Period End"] = blank
    metrics["Longest DD Days"] = blank

    # Add detailed volatility and risk metrics for full mode
    if mode.lower() == "full":
        # Calculate annualized volatility
        if isinstance(returns, _pd.Series):
            ret_vol = (
                _get_stats().volatility(df["returns"], win_year, True, prepare_returns=False)
                * pct
            )
        elif isinstance(returns, _pd.DataFrame):
            ret_vol = [
                _get_stats().volatility(
                    df[strategy_col], win_year, True, prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]

        # Add benchmark volatility if present
        if "benchmark" in df:
            bench_vol = (
                _get_stats().volatility(
                    df["benchmark"], win_year, True, prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(_pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            # Calculate benchmark-relative metrics
            if isinstance(returns, _pd.Series):
                metrics["R^2"] = _get_stats().r_squared(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
                metrics["Information Ratio"] = _get_stats().information_ratio(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
            elif isinstance(returns, _pd.DataFrame):
                metrics["R^2"] = (
                    [
                        _get_stats().r_squared(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        _get_stats().information_ratio(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            # No benchmark case
            if isinstance(returns, _pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, _pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        # Additional risk and return metrics
        metrics["Calmar"] = _get_stats().calmar(df, prepare_returns=False, periods=win_year)
        metrics["Skew"] = _get_stats().skew(df, prepare_returns=False)
        metrics["Kurtosis"] = _get_stats().kurtosis(df, prepare_returns=False)

        # Additional ratios
        metrics["Ulcer Performance Index"] = _get_stats().ulcer_performance_index(df, rf)
        metrics["Risk-Adjusted Return %"] = _get_stats().rar(df, rf) * pct
        metrics["Risk-Return Ratio"] = _get_stats().risk_return_ratio(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~"] = blank

        # Average return metrics
        metrics["Avg. Return %"] = _get_stats().avg_return(df, prepare_returns=False) * pct
        metrics["Avg. Win %"] = _get_stats().avg_win(df, prepare_returns=False) * pct
        metrics["Avg. Loss %"] = _get_stats().avg_loss(df, prepare_returns=False) * pct
        metrics["Win/Loss Ratio"] = _get_stats().win_loss_ratio(df, prepare_returns=False)
        metrics["Profit Ratio"] = _get_stats().profit_ratio(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~~"] = blank

        # Expected returns at different frequencies
        metrics["Expected Daily %%"] = (
            _get_stats().expected_return(df, compounded=compounded, prepare_returns=False)
            * pct
        )
        metrics["Expected Monthly %%"] = (
            _get_stats().expected_return(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Expected Yearly %%"] = (
            _get_stats().expected_return(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Risk management metrics
        metrics["Kelly Criterion %"] = (
            _get_stats().kelly_criterion(df, prepare_returns=False) * pct
        )
        metrics["Risk of Ruin %"] = _get_stats().risk_of_ruin(df, prepare_returns=False)

        # Value at Risk metrics
        metrics["Daily Value-at-Risk %"] = -abs(
            _get_stats().var(df, prepare_returns=False) * pct
        )
        metrics["Expected Shortfall (cVaR) %"] = -abs(
            _get_stats().cvar(df, prepare_returns=False) * pct
        )

    # Add separator
    metrics["~~~~~~"] = blank

    # Consecutive wins/losses analysis (full mode only)
    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _get_stats().consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _get_stats().consecutive_losses(df)

    # Pain-based metrics (Gain/Pain ratio)
    metrics["Gain/Pain Ratio"] = _get_stats().gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _get_stats().gain_to_pain_ratio(df, rf, "ME")
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _get_stats().gain_to_pain_ratio(df, rf, "QE")
    #     metrics['GPR (6M)'] = _get_stats().gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _get_stats().gain_to_pain_ratio(df, rf, "YE")

    # Add separator
    metrics["~~~~~~~"] = blank

    # Trading-based performance metrics
    metrics["Payoff Ratio"] = _get_stats().payoff_ratio(df, prepare_returns=False)
    metrics["Profit Factor"] = _get_stats().profit_factor(df, prepare_returns=False)
    metrics["Common Sense Ratio"] = _get_stats().common_sense_ratio(df, prepare_returns=False)
    metrics["CPC Index"] = _get_stats().cpc_index(df, prepare_returns=False)
    metrics["Tail Ratio"] = _get_stats().tail_ratio(df, prepare_returns=False)
    metrics["Outlier Win Ratio"] = _get_stats().outlier_win_ratio(df, prepare_returns=False)
    metrics["Outlier Loss Ratio"] = _get_stats().outlier_loss_ratio(df, prepare_returns=False)

    # # returns
    metrics["~~"] = blank

    # Time-based return analysis
    today = df.index[-1]  # _dt.today()
    m3 = today - relativedelta(months=3)
    m6 = today - relativedelta(months=6)
    y1 = today - relativedelta(years=1)

    # Calculate period returns based on compounding preference
    if compounded:
        metrics["MTD %"] = (
            _get_stats().comp(df[df.index >= _dt(today.year, today.month, 1)]) * pct
        )
        metrics["3M %"] = _get_stats().comp(df[df.index >= m3]) * pct
        metrics["6M %"] = _get_stats().comp(df[df.index >= m6]) * pct
        metrics["YTD %"] = _get_stats().comp(df[df.index >= _dt(today.year, 1, 1)]) * pct
        metrics["1Y %"] = _get_stats().comp(df[df.index >= y1]) * pct
    else:
        metrics["MTD %"] = (
            _np.sum(df[df.index >= _dt(today.year, today.month, 1)], axis=0) * pct
        )
        metrics["3M %"] = _np.sum(df[df.index >= m3], axis=0) * pct
        metrics["6M %"] = _np.sum(df[df.index >= m6], axis=0) * pct
        metrics["YTD %"] = _np.sum(df[df.index >= _dt(today.year, 1, 1)], axis=0) * pct
        metrics["1Y %"] = _np.sum(df[df.index >= y1], axis=0) * pct

    # Multi-year annualized returns
    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    metrics["All-time (ann.) %"] = _get_stats().cagr(df, 0.0, compounded, win_year) * pct

    # Best/worst period analysis (full mode only)
    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = (
            _get_stats().best(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Worst Day %"] = _get_stats().worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _get_stats().best(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Month %"] = (
            _get_stats().worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _get_stats().best(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Year %"] = (
            _get_stats().worst(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

    # Calculate and integrate drawdown metrics
    # return drawdown (dd) df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    # Add drawdown metrics to main metrics DataFrame
    # drawdown (dd) detail
    metrics["~~~~"] = blank
    # Properly integrate drawdown data into metrics
    for metric_name in dd.index:
        metrics[metric_name] = dd.loc[metric_name].values

    # Additional drawdown-based metrics
    metrics["Recovery Factor"] = _get_stats().recovery_factor(df)
    metrics["Ulcer Index"] = _get_stats().ulcer_index(df)
    metrics["Serenity Index"] = _get_stats().serenity_index(df, rf)

    # Win rate analysis (full mode only)
    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            _get_stats().avg_win(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Avg. Down Month %"] = (
            _get_stats().avg_loss(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Days %%"] = _get_stats().win_rate(df, prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Quarter %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="QE", prepare_returns=False
            )
            * pct
        )
        metrics["Win Year %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Greek letters and correlation analysis (if benchmark exists)
        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, _pd.Series):
                # Calculate Greek letters (Beta, Alpha) for single strategy
                greeks = _get_stats().greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _get_stats().treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, _pd.DataFrame):
                # Calculate Greek letters for multiple strategies
                greeks = [
                    _get_stats().greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _get_stats().treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # Format metrics for display
    # prepare for display
    for col in metrics.columns:
        try:
            # Try to convert to float and round
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except (ValueError, TypeError, AttributeError):
            pass
        # Handle integer columns (marked with *int)
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        # Add percentage signs to percentage columns
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    # Format drawdown days as integers
    try:
        metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = _pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    # Clean up column names (remove separators and percentage signs)
    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    # Set appropriate column names
    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(_pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # Final data cleaning
    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # Reorder columns to put benchmark first if present
    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]

    # Handle display vs return
    if display:
        # Build and display parameters table (feature #472)
        metrics_to_show = _zh_index(_zh_columns(metrics))
        params_data = {
            "参数": ["无风险利率", "年化周期数", "复利计算", "对齐日期"],
            "取值": [
                f"{rf:.1%}" if rf != 0 else "0.0%",
                str(periods_per_year),
                "是" if compounded else "否",
                "是" if match_dates else "否",
            ],
        }
        if benchmark is not None:
            params_data["参数"].insert(0, "基准")
            params_data["取值"].insert(0, benchmark_colname)
        params_df = _pd.DataFrame(params_data)
        logger.debug("\n" + _tabulate(params_df, headers="keys", tablefmt="simple", showindex=False))
        logger.debug("\n")
        logger.debug(_tabulate(metrics_to_show, headers="keys", tablefmt="simple"))
        return None

    # Remove separator rows if not requested
    if not sep:
        metrics = metrics[metrics.index != ""]

    # Final formatting for programmatic use
    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return _zh_index(_zh_columns(metrics))


def plots(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(8, 5),
    mode="basic",
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    生成投资组合绩效的综合可视化图表。

    本函数创建包含收益率、回撤、分布和滚动指标的完整绩效可视化图表集。
    它可以生成基本图表或完整的综合套件。

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        策略/投资组合的日收益率数据
    benchmark : pd.Series, str, or None, default None
        用于比较的基准收益率
    grayscale : bool, default False
        是否生成灰度图表
    figsize : tuple, default (8, 5)
        图表的尺寸，为 (宽度, 高度)
    mode : str, default "basic"
        绘图模式 - "basic" 用于基本图表，"full" 用于综合套件
    compounded : bool, default True
        是否复合收益率进行计算
    periods_per_year : int, default 252
        每年交易周期数
    prepare_returns : bool, default True
        是否准备/清理收益率数据
    match_dates : bool, default True
        是否对齐收益率和基准的开始日期
    **kwargs
        其他关键字参数：
        - strategy_title: 策略的自定义名称
        - benchmark_title: 基准的自定义名称
        - active: 是否显示相对于基准的主动收益率

    返回
    -------
    None
        显示各种绩效图表

    示例
    --------
    >>> plots(returns, benchmark='^GSPC', mode="full")
    >>> plots(returns, grayscale=True, figsize=(10, 6))
    """
    # Extract title parameters from kwargs
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)

    # Get trading periods for rolling window calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Clean returns data if date matching is enabled
    if match_dates is True:
        returns = returns.dropna()

    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Set names for display in plots
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_colname

    # Generate basic plots (snapshot and heatmap)
    if mode.lower() != "full":
        # Performance snapshot plot
        _get_plots().snapshot(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0]),
            show=True,
            mode=("comp" if compounded else "sum"),
            benchmark_title=benchmark_colname,
            strategy_title=strategy_colname,
        )

        # Monthly returns heatmap
        if isinstance(returns, _pd.Series):
            _get_plots().monthly_heatmap(
                returns,
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                compounded=compounded,
                active=active,
            )
        elif isinstance(returns, _pd.DataFrame):
            # Generate heatmap for each strategy column
            for col in returns.columns:
                _get_plots().monthly_heatmap(
                    returns[col].dropna(),
                    benchmark,
                    grayscale=grayscale,
                    figsize=(figsize[0], figsize[0] * 0.5),
                    show=True,
                    ylabel="",
                    returns_label=col,
                    compounded=compounded,
                    active=active,
                )

        return

    # prepare timeseries
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)
        benchmark.name = benchmark_colname
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Generate comprehensive plot suite
    # Cumulative returns plot
    _get_plots().returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.6),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Log returns plot for better visualization
    _get_plots().log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Volatility-matched returns (if benchmark exists)
    if benchmark is not None:
        _get_plots().returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compound=compounded,
        )

    # Yearly returns comparison
    _get_plots().yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Returns distribution histogram
    _get_plots().histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Calculate figure size for smaller plots
    small_fig_size = (figsize[0], figsize[0] * 0.35)
    if isinstance(returns, _pd.DataFrame) and len(returns.columns) > 1:
        small_fig_size = (
            figsize[0],
            figsize[0] * (0.33 * (len(returns.columns) * 0.66)),
        )

    # Daily returns scatter plot
    _get_plots().daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        prepare_returns=False,
        active=active,
    )

    # Rolling beta analysis (if benchmark exists)
    if benchmark is not None:
        _get_plots().rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            window1=win_half_year,
            window2=win_year,
            figsize=small_fig_size,
            show=True,
            ylabel="",
            prepare_returns=False,
        )

    # Rolling volatility analysis
    _get_plots().rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sharpe ratio analysis
    _get_plots().rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sortino ratio analysis
    _get_plots().rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Drawdown periods analysis
    if isinstance(returns, _pd.Series):
        _get_plots().drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                title=col,
                prepare_returns=False,
                compounded=compounded,
            )

    # Underwater (drawdown) plot
    _get_plots().drawdown(
        returns,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.4),
        show=True,
        ylabel="",
        compound=compounded,
    )

    # Monthly returns heatmap
    if isinstance(returns, _pd.Series):
        _get_plots().monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            returns_label=returns.name,
            show=True,
            ylabel="",
            compounded=compounded,
            active=active,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                returns_label=col,
                compounded=compounded,
                active=active,
            )

    # Returns distribution analysis
    if isinstance(returns, _pd.Series):
        _get_plots().distribution(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            title=returns.name,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                title=col,
                ylabel="",
                prepare_returns=False,
                compounded=compounded,
            )


def _calc_dd(df, display=True, as_pct=False):
    """
    计算绩效分析的回撤统计信息。

    此辅助函数计算包括最大回撤、回撤日期、恢复期间和平均回撤在内的综合回撤统计信息。
    它处理单个策略和多个策略分析。

    参数
    ----------
    df : pd.DataFrame
        包含收益率数据的 DataFrame，列为策略和可选的基准
    display : bool, default True
        输出是否用于显示（影响格式）
    as_pct : bool, default False
        是否返回百分比而不是小数

    返回
    -------
    pd.DataFrame
        包含回撤统计信息的 DataFrame，包括：
        - Max Drawdown %: 最大回撤百分比
        - Max DD Date: 最大回撤日期
        - Max DD Period Start: 最差回撤期间的开始日期
        - Max DD Period End: 最差回撤期间的结束日期
        - Longest DD Days: 最长回撤持续天数
        - Avg. Drawdown %: 平均回撤百分比
        - Avg. Drawdown Days: 平均回撤持续天数

    示例
    --------
    >>> dd_stats = _calc_dd(returns_df, display=False)
    >>> dd_stats = _calc_dd(returns_df, as_pct=True)
    """
    # Convert returns to drawdown series
    dd = _get_stats().to_drawdown_series(df)
    dd_info = _get_stats().drawdown_details(dd)

    # Return empty DataFrame if no drawdowns found
    if dd_info.empty:
        return _pd.DataFrame()

    # Handle different column structures based on data type
    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    # Calculate drawdown statistics based on data structure
    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        # Multiple strategy columns case
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Max DD Date": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["valley"]
                .values[0],
                "Max DD Period Start": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["start"]
                .values[0],
                "Max DD Period End": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["end"]
                .values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        # Single strategy case
        max_dd = ret_dd.sort_values(by="max drawdown", ascending=True)
        dd_stats = {
            "returns": {
                "Max Drawdown %": max_dd["max drawdown"].values[0] / 100,
                "Max DD Date": max_dd["valley"].values[0],
                "Max DD Period Start": max_dd["start"].values[0],
                "Max DD Period End": max_dd["end"].values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
            }
        }

    # Add benchmark drawdown statistics if present
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Max DD Date": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["valley"].values[0],
            "Max DD Period Start": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["start"].values[0],
            "Max DD Period End": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["end"].values[0],
            "Longest DD Days": str(
                _np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(bench_dd["days"].mean())),
        }

    # Apply percentage multiplier based on display settings
    # pct multiplier
    pct = 100 if display or as_pct else 1

    # Convert to DataFrame and apply percentage formatting
    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T


def _html_table(obj, showindex="default"):
    """
    将 DataFrame 转换为用于报告生成的 HTML 表格格式。

    此辅助函数将 pandas DataFrame 转换为适合嵌入 HTML 报告的简洁 HTML 表格格式。
    它移除默认的 tabulate 样式并清理间距以获得更好的展示效果。

    参数
    ----------
    obj : pd.DataFrame
        要转换为 HTML 表格的 DataFrame
    showindex : str or bool, default "default"
        是否在 HTML 表格中显示 DataFrame 索引。
        "default" 使用 tabulate 的默认行为

    返回
    -------
    str
        包含格式化表格的 HTML 字符串

    示例
    --------
    >>> html_str = _html_table(metrics_df)
    >>> html_str = _html_table(metrics_df, showindex=False)
    """
    # Convert DataFrame to HTML table using tabulate
    obj = _tabulate(
        obj, headers="keys", tablefmt="html", floatfmt=".2f", showindex=showindex
    )

    # Remove default tabulate styling attributes
    obj = obj.replace(' style="text-align: right;"', "")
    obj = obj.replace(' style="text-align: left;"', "")
    obj = obj.replace(' style="text-align: center;"', "")

    # Clean up spacing in table cells
    obj = _regex.sub("<td> +", "<td>", obj)
    obj = _regex.sub(" +</td>", "</td>", obj)
    obj = _regex.sub("<th> +", "<th>", obj)
    obj = _regex.sub(" +</th>", "</th>", obj)

    return obj


def _download_html(html, filename="quantstats-tearsheet.html"):
    """
    生成在浏览器中下载 HTML 内容的 JavaScript 代码。

    此辅助函数创建触发浏览器中 HTML 内容下载的 JavaScript 代码。
    它用于直接从 Jupyter notebooks 下载 tearsheet 报告。

    参数
    ----------
    html : str
        要下载的 HTML 内容
    filename : str, default "quantstats-tearsheet.html"
        下载文件的文件名

    返回
    -------
    None
        在 notebook 中显示 JavaScript 代码以触发下载

    示例
    --------
    >>> _download_html(html_content, "my_report.html")
    """
    # 创建用于文件下载的 JavaScript 代码
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var bl=new Blob(['{{html}}'],{type:"text/html"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(bl);
    a.download="{{filename}}";
    a.hidden=true;document.body.appendChild(a);
    a.innerHTML="download report";
    a.click();</script>""".replace(
            "\n", ""
        ),
    )

    # 插入 HTML 内容并清理格式
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # 如果在 notebook 环境中则执行 JavaScript
    if _get_utils()._in_notebook():
        iDisplay(iHTML(jscode.replace("{{filename}}", filename)))


def _open_html(html):
    """
    生成在新浏览器窗口中打开 HTML 内容的 JavaScript 代码。

    此辅助函数创建在新浏览器窗口中打开 HTML 内容的 JavaScript 代码。
    它用于直接从 Jupyter notebooks 在浏览器中显示 tearsheet 报告。

    参数
    ----------
    html : str
        要在新窗口中显示的 HTML 内容

    返回
    -------
    None
        在 notebook 中显示 JavaScript 代码以打开新窗口

    示例
    --------
    >>> _open_html(html_content)
    """
    # 创建用于打开带有 HTML 内容的新窗口的 JavaScript 代码
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var win=window.open();win.document.body.innerHTML='{{html}}';
    </script>""".replace(
            "\n", ""
        ),
    )

    # 插入 HTML 内容并清理格式
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # 如果在 notebook 环境中则执行 JavaScript
    if _get_utils()._in_notebook():
        iDisplay(iHTML(jscode))


def _embed_figure(figfiles, figfmt):
    """
    将 matplotlib 图形嵌入 HTML 格式用于报告。

    此辅助函数将 matplotlib 图形对象转换为适合包含在 HTML 报告中的嵌入 HTML 格式。
    它处理 SVG 和 base64 编码的图像格式。

    参数
    ----------
    figfiles : io.StringIO or list of io.StringIO
        包含图形数据的文件类对象。可以是单个图形
        或用于多图表的图形列表
    figfmt : str
        图形的格式（'svg'、'png'、'jpg' 等）

    返回
    -------
    str
        准备好包含在报告中的嵌入图形的 HTML 字符串

    示例
    --------
    >>> embed_str = _embed_figure(figfile, 'svg')
    >>> embed_str = _embed_figure([fig1, fig2], 'png')
    """
    # 处理多个图形
    if isinstance(figfiles, list):
        embed_string = "\n"
        for figfile in figfiles:
            figbytes = figfile.getvalue()
            if figfmt == "svg":
                # SVG 可以直接作为文本嵌入
                return figbytes.decode()
            # 对于其他格式，编码为 base64 data URI
            data_uri = _b64encode(figbytes).decode()
            embed_string.join(
                '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)
            )
    else:
        # 处理单个图形
        figbytes = figfiles.getvalue()
        if figfmt == "svg":
            # SVG 可以直接作为文本嵌入
            return figbytes.decode()
        # 对于其他格式，编码为 base64 data URI
        data_uri = _b64encode(figbytes).decode()
        embed_string = '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)

    return embed_string
