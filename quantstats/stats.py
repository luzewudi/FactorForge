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

"""
投资组合统计模块

本模块提供全面的投资组合绩效评估、风险评估和基准比较的统计分析函数。
它包含用于计算各种收益率指标、风险比率、回撤分析以及与基准进行比较的函数。

本模块设计用于处理包含收益率数据、价格数据或绩效指标的 pandas Series 和 DataFrame。
"""

from warnings import warn
from typing import Literal
import pandas as _pd
import numpy as _np
from numpy.typing import NDArray
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import norm as _norm, linregress as _linregress

from . import utils as _utils
from ._compat import safe_concat
from .utils import validate_input

# 常用类型的类型别名（Python 3.10+ 语法）
Returns = _pd.Series | _pd.DataFrame
"""收益率数据的类型别名：可以是 pandas Series 或 DataFrame。"""

# ======== 统计函数 ========


def pct_rank(prices: _pd.Series, window: int = 60) -> _pd.Series:
    """
    计算滚动窗口内价格的百分位排名。

    本函数计算每个价格在滚动窗口内的百分位排名（0-100），
    用于识别当前价格在最近历史中的相对位置。

    参数:
        prices (pd.Series): 价格数据序列
        window (int): 排名计算的滚动窗口大小（默认：60）

    返回:
        pd.Series: 百分位排名（0-100 刻度）

    示例:
        >>> prices = pd.Series([100, 105, 110, 95, 120])
        >>> ranks = pct_rank(prices, window=3)
        >>> print(ranks)
    """
    # 创建滚动窗口移位并转置以进行排名
    rank = _utils.multi_shift(prices, window).T.rank(pct=True).T
    # 提取第一列并转换为百分比刻度
    return rank.iloc[:, 0] * 100.0


def compsum(returns: Returns) -> Returns:
    """
    计算滚动复合收益率（累计乘积）。

    本函数通过将每个收益率加 1、取累计乘积、再减 1 来计算累计复合收益率。

    参数:
        returns: 收益率的 Series 或 DataFrame

    返回:
        累计复合收益率（与输入类型相同）

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> cumulative = compsum(returns)
        >>> print(cumulative)
    """
    # 加 1 将收益率转换为增长因子，然后取累计乘积
    return returns.add(1).cumprod(axis=0) - 1


def comp(returns: Returns) -> _pd.Series | float:
    """
    计算总复合收益率（最终累计收益率）。

    本函数通过将收益率转换为增长因子并取其乘积来计算整个期间的总复合收益率。

    参数:
        returns (pd.Series): 收益率序列

    返回:
        float: 总复合收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> total_return = comp(returns)
        >>> print(total_return)
    """
    # 将收益率转换为增长因子，取乘积，减 1
    return returns.add(1).prod(axis=0) - 1


def distribution(
    returns: Returns,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> dict:
    """
    分析不同时间周期的收益率分布。

    本函数计算日、周、月、季、年各周期的收益率分布（包括异常值）。
    使用 IQR 方法识别异常值（Q1/Q3 之外 1.5 * IQR）。

    参数:
        returns (pd.Series): 要分析的收益率序列
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        dict: 包含每个周期分布数据的字典

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01],
        ...                    index=pd.date_range('2023-01-01', periods=3))
        >>> dist = distribution(returns)
        >>> print(dist['Daily']['values'])
    """
    def get_outliers(data):
        """
        使用 IQR 方法识别异常值。

        使用 1.5 * IQR 规则：超出 Q1 - 1.5*IQR 或 Q3 + 1.5*IQR 的值被视为异常值。
        """
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)  # 第一四分位数
        Q3 = data.quantile(0.75)  # 第三四分位数
        IQR = Q3 - Q1  # 四分位距

        # 为非异常值创建过滤器
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)

        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    # 处理 DataFrame 输入：选择适当的列
    if isinstance(returns, _pd.DataFrame):
        warn(
            "传入了 Pandas DataFrame（期望 Series）。"
            "只会使用第一列。"
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    # 根据 compounded 参数选择聚合函数
    apply_fnc = comp if compounded else _np.sum
    daily = returns.dropna()

    # 如果请求则准备收益率
    if prepare_returns:
        daily = _utils._prepare_returns(daily)

    # 计算不同时间周期的分布
    return {
        "Daily": get_outliers(daily),
        "Weekly": get_outliers(daily.resample("W-MON").apply(apply_fnc)),
        "Monthly": get_outliers(daily.resample("ME").apply(apply_fnc)),
        "Quarterly": get_outliers(daily.resample("QE").apply(apply_fnc)),
        "Yearly": get_outliers(daily.resample("YE").apply(apply_fnc)),
    }


def expected_return(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    计算给定周期的预期收益率（几何平均收益率）。

    本函数计算几何持有期收益率，它代表基于历史数据的每周期预期收益。
    计算方式为 (1 + 收益率) 乘积的 n 次方根再减 1。

    参数:
        returns (pd.Series): 收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 每周期的预期收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> expected = expected_return(returns)
        >>> print(f"预期收益率: {expected:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果指定了周期，则聚合收益率
    returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # 计算几何平均：(1 + 收益率) 的乘积的 1/n 次方 - 1
    return _np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


def geometric_mean(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float:
    """
    计算收益率的几何平均数。

    这是 expected_return() 函数的简写形式，参数相同。

    参数:
        returns (pd.Series): 收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）

    返回:
        float: 收益率的几何平均数
    """
    return expected_return(returns, aggregate, compounded)


def ghpr(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float:
    """
    计算几何持有期收益率（Geometric Holding Period Return）。

    这是 expected_return() 函数的简写形式，参数相同。
    GHPR 代表每周期的平均收益率。

    参数:
        returns (pd.Series): 收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）

    返回:
        float: 几何持有期收益率
    """
    return expected_return(returns, aggregate, compounded)


def outliers(returns: Returns, quantile: float = 0.95) -> Returns:
    """
    识别并返回超过指定分位数的异常收益。

    本函数筛选收益率，仅显示超过指定分位数阈值的收益，
    用于识别极端正收益期间。

    参数:
        returns (pd.Series): 要分析的收益率序列
        quantile (float): 分位数阈值（默认：0.95，即第95百分位）

    返回:
        pd.Series: 超过分位数阈值的收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, 0.05, -0.01, 0.10])
        >>> outlier_returns = outliers(returns, quantile=0.90)
        >>> print(outlier_returns)
    """
    # 筛选超过指定分位数的收益率并移除 NaN 值
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


def remove_outliers(returns: Returns, quantile: float = 0.95) -> Returns:
    """
    移除超过指定分位数的异常收益率。

    本函数过滤掉超过分位数阈值的极端收益率，
    可用于通过移除极端值进行稳健的统计分析。

    参数:
        returns (pd.Series): 要过滤的收益率序列
        quantile (float): 分位数阈值（默认：0.95，即第95百分位）

    返回:
        pd.Series: 低于分位数阈值的收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, 0.05, -0.01, 0.10])
        >>> filtered = remove_outliers(returns, quantile=0.90)
        >>> print(filtered)
    """
    # 仅保留低于指定分位数阈值的收益率
    return returns[returns < returns.quantile(quantile)]


def best(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    找出给定周期的最佳（最高）收益率。

    本函数识别指定聚合周期内的最大收益率，
    帮助识别数据集中表现最佳的周期。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 该周期的最佳（最大）收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> best_return = best(returns)
        >>> print(f"最佳收益率: {best_return:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 聚合收益率并取最大值
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


def worst(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    找出给定周期的最差（最低）收益率。

    本函数识别指定聚合周期内的最小收益率，
    帮助识别数据集中表现最差的周期。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 该周期的最差（最小）收益率

    示例:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> worst_return = worst(returns)
        >>> print(f"最差收益率: {worst_return:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 聚合收益率并取最小值
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
    """
    计算连续盈利期间的最大数量。

    本函数识别最长连续正收益序列，
    有助于评估正收益的一致性。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        int: 最大连续盈利期间数量

    示例:
        >>> returns = pd.Series([0.01, 0.02, 0.03, -0.01, 0.02])
        >>> max_wins = consecutive_wins(returns)
        >>> print(f"最大连续盈利次数: {max_wins}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 聚合收益率并转换为布尔值（正收益 = True）
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0

    # 计算连续 True 值数量并返回最大值
    return _utils._count_consecutive(returns).max()


def consecutive_losses(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
    """
    计算连续亏损期间的最大数量。

    本函数识别最长连续负收益序列，
    有助于评估潜在长期回撤期间的风险。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        int: 最大连续亏损期间数量

    示例:
        >>> returns = pd.Series([0.01, -0.02, -0.01, -0.01, 0.02])
        >>> max_losses = consecutive_losses(returns)
        >>> print(f"最大连续亏损次数: {max_losses}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 聚合收益率并转换为布尔值（负收益 = True）
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0

    # 计算连续 True 值数量并返回最大值
    return _utils._count_consecutive(returns).max()


def exposure(
    returns: Returns,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    计算市场暴露时间比例（非零收益率期间的百分比）。

    本函数测量策略实际投资（有非零收益率）的时间比例，
    与持有现金或零仓位的对比。

    参数:
        returns (pd.Series or pd.DataFrame): 收益率序列或 DataFrame
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float or pd.Series: 暴露比例（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, 0.00, 0.02, 0.00, 0.03])
        >>> exp = exposure(returns)
        >>> print(f"市场暴露: {exp:.2%}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    def _exposure(ret):
        """
        计算单个收益率序列的暴露比例。

        统计非 NaN、非零收益率数量并除以总期间数。
        向上取整到最近的百分比以避免舍入导致的零暴露。
        """
        # 统计非 NaN 且非零的收益率数量
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / len(ret)
        # 向上取整到最近的百分比
        return _ceil(ex * 100) / 100

    # 处理 DataFrame 输入，为每列计算暴露比例
    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return _pd.Series(_df)

    return _exposure(returns)


def win_rate(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    计算胜率（盈利期间的百分比）。

    本函数计算正收益率与非零总收益率的比率，
    提供策略产生利润频率的度量。

    参数:
        returns (pd.Series or pd.DataFrame): 收益率序列或 DataFrame
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float or pd.Series: 胜率（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> wr = win_rate(returns)
        >>> print(f"胜率: {wr:.2%}")
    """
    def _win_rate(series):
        """
        计算单个收益率序列的胜率。

        处理边缘情况（如没有非零收益率）并提供计算错误处理。
        """
        try:
            # 过滤掉零收益率（无交易期间）
            non_zero_returns = series[series != 0]
            if len(non_zero_returns) == 0:
                warn("未找到非零收益率用于胜率计算，返回 0.0")
                return 0.0

            # 计算正收益率与非零收益率的比率
            return len(series[series > 0]) / len(non_zero_returns)
        except (ValueError, TypeError) as e:
            warn(f"胜率计算错误: {e}，返回 0.0")
            return 0.0

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果指定了周期，则聚合收益率
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # 处理 DataFrame 输入，为每列计算胜率
    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])
        return _pd.Series(_df)

    return _win_rate(returns)


def avg_return(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    计算每周期平均收益率（不含零收益率）。

    本函数计算非零收益率的平均值，
    提供策略活跃时典型收益幅度的洞察。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 每周期平均收益率

    示例:
        >>> returns = pd.Series([0.01, 0.00, 0.02, -0.01, 0.03])
        >>> avg_ret = avg_return(returns)
        >>> print(f"平均收益率: {avg_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果指定了周期，则聚合收益率
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # 计算非零收益率的平均值
    return returns[returns != 0].dropna().mean()


def avg_win(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    计算平均盈利收益率（正收益率的平均值）。

    本函数仅计算正收益率的平均值，
    显示盈利期间的典型幅度。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 平均盈利收益率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> avg_win_ret = avg_win(returns)
        >>> print(f"平均盈利: {avg_win_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果指定了周期，则聚合收益率
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # 仅计算正收益率的平均值
    return returns[returns > 0].dropna().mean()


def avg_loss(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    """
    计算平均亏损收益率（负收益率的平均值）。

    本函数仅计算负收益率的平均值，
    显示亏损期间的典型幅度。

    参数:
        returns (pd.Series): 要分析的收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 平均亏损收益率（负值）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> avg_loss_ret = avg_loss(returns)
        >>> print(f"平均亏损: {avg_loss_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果指定了周期，则聚合收益率
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # 仅计算负收益率的平均值
    return returns[returns < 0].dropna().mean()


def volatility(
    returns: Returns,
    periods: int = 252,
    annualize: bool = True,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    计算收益率的波动率（标准差）。

    本函数计算收益率的波动率，衡量收益率随时间的变异程度。
    更高的波动率表示更大的不确定性和风险。

    参数:
        returns: 要分析的收益率序列或 DataFrame
        periods: 年化周期数（默认：252）
        annualize: 是否年化波动率（默认：True）
        prepare_returns: 是否先准备收益率（默认：True）

    返回:
        波动率（如果 annualize=True 则为年化值）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> vol = volatility(returns)
        >>> print(f"年化波动率: {vol:.4f}")
    """
    validate_input(returns)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算收益率的标准差
    std = returns.std()

    # 通过乘以年周期数的平方根进行年化
    if annualize:
        return std * _np.sqrt(periods)

    return std


def rolling_volatility(
    returns: Returns,
    rolling_period: int = 126,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> _pd.Series:
    """
    计算指定窗口的滚动波动率。

    本函数使用滚动窗口计算波动率，
    提供随时间变化的风险度量，能够适应不断变化的市场条件。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rolling_period (int): 滚动窗口大小（默认：126，约6个月）
        periods_per_year (int): 年化周期数（默认：252）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.Series: 滚动波动率序列（年化）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_vol = rolling_volatility(returns, rolling_period=3)
        >>> print(rolling_vol)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns, rolling_period)

    # 计算滚动标准差并年化
    return returns.rolling(rolling_period).std() * _np.sqrt(periods_per_year)


def implied_volatility(
    returns: Returns,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
    """
    使用对数收益率计算隐含波动率。

    本函数使用对数收益率而非简单收益率计算波动率，
    这对于连续复利的数学处理更为恰当。

    参数:
        returns (pd.Series): 要分析的收益率序列
        periods (int): 滚动计算的周期数（默认：252）
        annualize (bool): 是否年化波动率（默认：True）

    返回:
        float or pd.Series: 隐含波动率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> impl_vol = implied_volatility(returns)
        >>> print(f"隐含波动率: {impl_vol:.4f}")
    """
    # 转换为对数收益率用于连续复利
    logret = _utils.log_returns(returns)

    if annualize:
        # 计算滚动波动率并年化
        return logret.rolling(periods).std() * _np.sqrt(periods)

    # 返回简单标准差
    return logret.std()


def autocorr_penalty(
    returns: Returns,
    prepare_returns: bool = False,
) -> float:
    """
    计算风险调整指标的自相关惩罚因子。

    本函数计算一个惩罚因子，用于修正收益率中的自相关性。
    自相关会夸大风险调整比率。该因子用于调整
    Sharpe 和 Sortino 比率以获得更现实的风险评估。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：False）

    返回:
        float: 自相关惩罚因子（>= 1）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> penalty = autocorr_penalty(returns)
        >>> print(f"自相关惩罚因子: {penalty:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 处理 DataFrame 输入，选择第一列
    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    # returns.to_csv('/Users/ran/Desktop/test.csv')
    num = len(returns)

    # 计算连续收益率之间的自相关系数
    coef = _np.abs(_np.corrcoef(returns[:-1], returns[1:])[0, 1])

    # 使用向量化计算代替列表推导式
    x = _np.arange(1, num)
    # 计算随时间变化的加权相关效应
    corr = ((num - x) / num) * (coef**x)

    # 返回惩罚因子（1 + 2 * 相关性之和的平方根）
    return _np.sqrt(1 + 2 * _np.sum(corr))


# ======= METRICS =======


def sharpe(
    returns: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
    """
    计算超额收益率的夏普比率。

    夏普比率通过将超额收益率（收益率 - 无风险利率）
    除以收益率标准差来衡量风险调整收益。
    数值越高表示风险调整表现越好。

    参数:
        returns: 要分析的收益率序列或 DataFrame
        rf: 无风险利率（如果指定了周期则为年化值，默认：0.0）
        periods: 年化周期数（默认：252）
        annualize: 是否年化比率（默认：True）
        smart: 是否应用自相关惩罚（默认：False）

    返回:
        夏普比率（Series 输入返回 float，DataFrame 输入返回 Series）

    异常:
        ValueError: 如果 rf 非零但 periods 为 None

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sharpe_ratio = sharpe(returns, rf=0.02)
        >>> print(f"夏普比率: {sharpe_ratio:.4f}")
    """
    validate_input(returns)

    # 验证无风险利率参数处理
    if rf != 0 and periods is None:
        raise ValueError("当无风险利率 (rf) 非零时，需要 periods 参数。"
                         "这对于正确年化无风险利率是必需的。")

    # 准备收益率（如果适用则减去无风险利率）
    returns = _utils._prepare_returns(returns, rf, periods)

    # 计算标准差作为分母
    divisor = returns.std(ddof=1)

    # 如果启用智能模式，应用自相关惩罚
    if smart:
        # 用自相关惩罚夏普比率
        divisor = divisor * autocorr_penalty(returns)

    # 计算基础夏普比率
    res = returns.mean() / divisor

    # 如果请求则年化
    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sharpe(
    returns: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
    """
    计算智能夏普比率（带自相关惩罚的夏普比率）。

    这是 sharpe() 函数的包装器，smart=True，
    应用自相关惩罚以对具有自相关收益率的策略
    提供更现实的风险调整收益。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化比率（默认：True）

    返回:
        float: 智能夏普比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> smart_sharpe_ratio = smart_sharpe(returns)
        >>> print(f"智能夏普比率: {smart_sharpe_ratio:.4f}")
    """
    return sharpe(returns, rf, periods, annualize, True)


def rolling_sharpe(
    returns: Returns,
    rf: float = 0.0,
    rolling_period: int = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> _pd.Series:
    """
    计算指定窗口的滚动夏普比率。

    本函数使用滚动窗口计算夏普比率，
    提供随时间变化的风险调整表现度量，
    能够适应不断变化的市场条件。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        rolling_period (int): 滚动窗口大小（默认：126，约6个月）
        annualize (bool): 是否年化比率（默认：True）
        periods_per_year (int): 年化周期数（默认：252）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.Series: 滚动夏普比率序列

    异常:
        Exception: 如果 rf != 0 且 rolling_period 为 None

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sharpe_ratio = rolling_sharpe(returns, rolling_period=3)
        >>> print(rolling_sharpe_ratio)
    """
    # 验证无风险利率参数处理
    if rf != 0 and rolling_period is None:
        raise Exception("如果 rf != 0，必须提供周期数")

    if prepare_returns:
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    # 计算滚动均值和标准差
    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    # 如果请求则年化
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


def sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
    """
    计算超额收益率的索提诺比率。

    索提诺比率类似于夏普比率，但使用下行偏差
    而不是总波动率，仅关注有害的波动率。
    这提供了更准确的风险调整收益度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化比率（默认：True）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 索提诺比率

    异常:
        ValueError: 如果 rf 非零但 periods 为 None

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sortino_ratio = sortino(returns, rf=0.02)
        >>> print(f"索提诺比率: {sortino_ratio:.4f}")

    注意:
        计算基于 Red Rock Capital 的论文：
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    validate_input(returns)

    # 验证无风险利率参数处理
    if rf != 0 and periods is None:
        raise ValueError("当无风险利率 (rf) 非零时，需要 periods 参数。"
                         "这对于正确年化无风险利率是必需的。")

    # 准备收益率（如果适用则减去无风险利率）
    returns = _utils._prepare_returns(returns, rf, periods)

    # 计算下行偏差（仅负收益率）
    downside = _np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    # 如果启用智能模式，应用自相关惩罚
    if smart:
        # 用自相关惩罚索提诺比率
        downside = downside * autocorr_penalty(returns)

    # 计算基础索提诺比率
    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(downside, _pd.Series):
        res = returns.mean() / downside.replace(0, _np.nan)
    else:
        if downside == 0:
            res = _np.nan
        else:
            res = returns.mean() / downside

    # 如果请求则年化
    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
    """
    计算智能索提诺比率（带自相关惩罚的索提诺比率）。

    这是 sortino() 函数的包装器，smart=True，
    应用自相关惩罚以对具有自相关收益率的策略
    提供更现实的风险调整收益。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化比率（默认：True）

    返回:
        float: 智能索提诺比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> smart_sortino_ratio = smart_sortino(returns)
        >>> print(f"智能索提诺比率: {smart_sortino_ratio:.4f}")
    """
    return sortino(returns, rf, periods, annualize, True)


def rolling_sortino(
    returns: Returns,
    rf: float = 0,
    rolling_period: int = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    **kwargs,
) -> _pd.Series:
    """
    计算指定窗口的滚动索提诺比率。

    本函数使用滚动窗口计算索提诺比率，
    提供随时间变化的下行风险调整表现度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        rolling_period (int): 滚动窗口大小（默认：126，约6个月）
        annualize (bool): 是否年化比率（默认：True）
        periods_per_year (int): 年化周期数（默认：252）
        **kwargs: 附加关键字参数（如 prepare_returns）

    返回:
        pd.Series: 滚动索提诺比率序列

    异常:
        Exception: 如果 rf != 0 且 rolling_period 为 None

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sortino_ratio = rolling_sortino(returns, rolling_period=3)
        >>> print(rolling_sortino_ratio)
    """
    # 验证无风险利率参数处理
    if rf != 0 and rolling_period is None:
        raise Exception("如果 rf != 0，必须提供周期数")

    if kwargs.get("prepare_returns", True):
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    # 使用向量化操作优化下行偏差计算
    def calc_downside(x):
        """
        更高效地计算下行方差。

        本函数计算负收益率平方和，
        用于计算下行偏差。
        """
        negative_returns = x[x < 0]
        return (negative_returns**2).sum() if len(negative_returns) > 0 else 0

    # 计算滚动下行偏差
    downside = (
        returns.rolling(rolling_period).apply(calc_downside, raw=True) / rolling_period
    )

    # 计算滚动索提诺比率
    res = returns.rolling(rolling_period).mean() / _np.sqrt(downside)

    # 如果请求则年化
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


def adjusted_sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
    """
    计算 Jack Schwager 的调整索提诺比率。

    此版本的索提诺比率通过除以 sqrt(2) 进行调整，
    以便与夏普比率进行直接比较。此调整
    考虑了计算方法的差异。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化比率（默认：True）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 调整索提诺比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> adj_sortino = adjusted_sortino(returns)
        >>> print(f"调整索提诺比率: {adj_sortino:.4f}")

    注意:
        更多信息请见：https://archive.is/wip/2rwFW
    """
    # 计算标准索提诺比率
    data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)

    # 应用 Schwager 的调整因子
    return data / _sqrt(2)


def probabilistic_ratio(
    series: Returns,
    rf: float = 0.0,
    base: str = "sharpe",
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    """
    计算给定基础指标的概率比率。

    本函数计算风险调整比率的概率版本，
    该版本考虑了比率估计中的统计不确定性。
    它考虑偏度和峰度以提供更稳健的估计。

    参数:
        series (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        base (str): 基础指标 ('sharpe', 'sortino', 'adjusted_sortino')
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化结果（默认：False）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 概率比率（0-1 刻度，表示概率）

    异常:
        ValueError: 如果提供了无效的基础指标

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> prob_ratio = probabilistic_ratio(returns, base="sharpe")
        >>> print(f"概率夏普比率: {prob_ratio:.4f}")
    """
    # 根据选择的指标计算基础比率
    if base.lower() == "sharpe":
        base = sharpe(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "sortino":
        base = sortino(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "adjusted_sortino":
        base = adjusted_sortino(series, periods=periods, annualize=False, smart=smart)
    else:
        raise ValueError(
            f"无效的指标 '{base}'。必须是以下之一：'sharpe', 'sortino', 或 'adjusted_sortino'"
        )

    # 计算调整用的高阶矩
    skew_no = skew(series, prepare_returns=False)
    kurtosis_no = kurtosis(series, prepare_returns=False)

    n = len(series)

    # 计算包含高阶矩的比率标准误差
    # 公式考虑了偏度和峰度对比率分布的影响
    sigma_sr = _np.sqrt(
        (1 + (0.5 * base**2) - (skew_no * base) + (((kurtosis_no - 3) / 4) * base**2))
        / (n - 1)
    )

    # 计算标准化比率并转换为概率
    ratio = (base - rf) / sigma_sr
    psr = _norm.cdf(ratio)

    # 如果请求则年化
    if annualize:
        return psr * (252**0.5)

    return psr


def probabilistic_sharpe_ratio(
    series: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    """
    计算概率夏普比率（PSR）。

    本函数计算 PSR，它表示观察到的夏普比率
    在统计上大于基准的概率。
    它考虑高阶矩以提供更稳健的估计。

    参数:
        series (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化结果（默认：False）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 概率夏普比率（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_sharpe_ratio(returns)
        >>> print(f"概率夏普比率: {psr:.4f}")
    """
    return probabilistic_ratio(
        series, rf, base="sharpe", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    """
    计算概率索提诺比率。

    本函数计算索提诺比率的概率版本，
    该版本考虑了比率估计中的统计不确定性。

    参数:
        series (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化结果（默认：False）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 概率索提诺比率（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_sortino_ratio(returns)
        >>> print(f"概率索提诺比率: {psr:.4f}")
    """
    return probabilistic_ratio(
        series, rf, base="sortino", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_adjusted_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    """
    计算概率调整索提诺比率。

    本函数计算调整索提诺比率的概率版本，
    考虑了比率估计中的统计不确定性。

    参数:
        series (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        periods (int): 年化周期数（默认：252）
        annualize (bool): 是否年化结果（默认：False）
        smart (bool): 是否应用自相关惩罚（默认：False）

    返回:
        float: 概率调整索提诺比率（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_adjusted_sortino_ratio(returns)
        >>> print(f"概率调整索提诺比率: {psr:.4f}")
    """
    return probabilistic_ratio(
        series,
        rf,
        base="adjusted_sortino",
        periods=periods,
        annualize=annualize,
        smart=smart,
    )


def treynor_ratio(returns, benchmark, periods=252.0, rf=0.0):
    """
    计算特雷诺比率。

    特雷诺比率通过系统性风险（贝塔）
    而非总风险（波动率）来衡量风险调整收益。
    计算方式为超额收益除以贝塔，
    用于比较具有不同市场敞口的投资组合。

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于计算贝塔的基准收益率序列
        periods (float): 年化周期数（默认：252.0）
        rf (float): 无风险利率（年化，默认：0.0）

    返回:
        float: 特雷诺比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> treynor = treynor_ratio(returns, benchmark)
        >>> print(f"特雷诺比率: {treynor:.4f}")
    """
    # 处理 DataFrame 输入，选择第一列
    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    # 从希腊值（阿尔法、贝塔分析）计算贝塔
    beta = greeks(returns, benchmark, periods=periods).to_dict().get("beta", 0)

    # 防止除以零
    if beta == 0:
        warn("贝塔为零，无法计算特雷诺比率，返回 0")
        return 0

    # 计算超额收益（除以无风险利率）除以贝塔
    return (comp(returns) - rf) / beta


def omega(
    returns: Returns,
    rf: float = 0.0,
    required_return: float = 0.0,
    periods: int = 252,
) -> float:
    """
    计算策略的 Omega 比率。

    Omega 比率衡量收益与损失的概率加权比率，
    高于和低于阈值收益。它提供了
    收益分布特征的全面视图。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        required_return (float): 阈值收益率（默认：0.0）
        periods (int): 年化周期数（默认：252）

    返回:
        float: Omega 比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> omega_ratio = omega(returns, required_return=0.01)
        >>> print(f"Omega 比率: {omega_ratio:.4f}")

    注意:
        详情请见 https://en.wikipedia.org/wiki/Omega_ratio
    """
    validate_input(returns)

    # 验证最小数据要求
    if len(returns) < 2:
        warn("Omega 比率计算数据不足（需要至少 2 个收益率），返回 NaN")
        return _np.nan

    # 验证必需收益率参数
    if required_return <= -1:
        warn(f"无效的 required_return ({required_return}) 用于 Omega 比率，必须 > -1，返回 NaN")
        return _np.nan

    # 准备收益率（如果适用则减去无风险利率）
    returns = _utils._prepare_returns(returns, rf, periods)

    # 如果需要，将年化必需收益率转换为每周期值
    if periods == 1:
        return_threshold = required_return
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    # 计算与阈值的偏差
    returns_less_thresh = returns - return_threshold

    # 正偏差之和（超过阈值的收益）
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum()

    # 负偏差之和的绝对值（低于阈值的损失）
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()

    # 处理 Series 和标量情况
    if isinstance(denom, _pd.Series):
        result = numer / denom
        # 在分母为零处返回 NaN
        result = result.where(denom > 0.0, _np.nan)
        return result
    else:
        if denom > 0.0:
            return numer / denom
        return _np.nan


def gain_to_pain_ratio(returns, rf=0, resolution="D"):
    """
    计算 Jack Schwager 的收益痛苦比率（GPR）。

    该比率衡量总收益除以总损失，
    提供了每单位损失产生多少利润的简单度量。
    数值越高表示表现越好。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（默认：0）
        resolution (str): 重采样频率（'D', 'W', 'M' 等）

    返回:
        float: 收益痛苦比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> gpr = gain_to_pain_ratio(returns)
        >>> print(f"收益痛苦比率: {gpr:.4f}")

    注意:
        更多信息请见：https://archive.is/wip/2rwFW
    """
    # 准备收益率并重采样到指定频率
    returns = _utils._prepare_returns(returns, rf).resample(resolution).sum()

    # 计算负收益率绝对值之和（痛苦）
    downside = abs(returns[returns < 0].sum())

    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(downside, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return returns.sum() / downside.replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if downside == 0:
            return _np.nan
        return returns.sum() / downside


def cagr(
    returns: Returns,
    rf: float = 0.0,
    compounded: bool = True,
    periods: int = 252,
) -> float | _pd.Series:
    """
    计算超额收益率的复合年增长率（CAGR）。

    CAGR 代表几何平均年增长率，提供
    考虑复利效应的平滑年化收益率。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）
        compounded (bool): 是否复合收益率（默认：True）
        periods (int): 年化周期数（默认：252）

    返回:
        float or pd.Series: CAGR 百分比

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
        ...                    index=pd.date_range('2023-01-01', periods=5))
        >>> cagr_value = cagr(returns)
        >>> print(f"CAGR: {cagr_value:.4f}")
    """
    validate_input(returns)

    # 准备收益率（如果适用则减去无风险利率）
    total = _utils._prepare_returns(returns, rf)

    # 计算总收益率
    if compounded:
        total = comp(total)
    else:
        total = _np.sum(total, axis=0)

    # 使用交易周期计算以年为单位的时间段
    # 这与 quantstats 中夏普、索提诺和其他指标
    # 处理年化的方式一致
    years = len(returns) / periods

    # 使用几何平均公式计算 CAGR
    res = abs(total + 1.0) ** (1.0 / years) - 1

    # 处理 DataFrame 输入
    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.0):
    """
    计算风险调整收益（RAR）。

    RAR 计算为 CAGR 除以暴露比例，
    考虑了策略实际投资的时间。这提供了
    相对于实际市场参与的更准确收益度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（年化，默认：0.0）

    返回:
        float: 风险调整收益

    示例:
        >>> returns = pd.Series([0.01, 0.00, 0.03, 0.00, 0.02])
        >>> rar_value = rar(returns)
        >>> print(f"风险调整收益: {rar_value:.4f}")
    """
    # 准备收益率（如果适用则减去无风险利率）
    returns = _utils._prepare_returns(returns, rf)

    # 计算 CAGR 并除以暴露比例
    return cagr(returns) / exposure(returns)


def skew(returns, prepare_returns=True):
    """
    计算收益率的偏度。

    偏度衡量分布围绕其均值的不对称程度。
    正偏度表示正侧有更长的尾巴，
    而负偏度表示负侧有更长的尾巴。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 偏度值

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> skewness = skew(returns)
        >>> print(f"偏度: {skewness:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 使用 pandas 内置方法计算偏度
    return returns.skew()


def kurtosis(returns, prepare_returns=True):
    """
    计算收益率的峰度。

    峰度衡量分布相对于正态分布的峰度程度。
    更高的峰度表示更极端的收益率（肥尾），
    而更低的峰度表示更少的极端收益率。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 峰度值（超量峰度，正态分布 = 0）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> kurt = kurtosis(returns)
        >>> print(f"峰度: {kurt:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 使用 pandas 内置方法计算峰度（超量峰度）
    return returns.kurtosis()


def calmar(
    returns: Returns,
    prepare_returns: bool = True,
    periods: int = 252,
) -> float:
    """
    计算卡尔马比率（CAGR / 最大回撤）。

    卡尔马比率通过将 CAGR 除以最大回撤的绝对值
    来衡量风险调整收益。它提供了
    相对于最坏情况场景的收益洞察。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）
        periods (int): 年化周期数（默认：252）

    返回:
        float: 卡尔马比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> calmar_ratio = calmar(returns)
        >>> print(f"卡尔马比率: {calmar_ratio:.4f}")
    """
    validate_input(returns)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算 CAGR 和最大回撤
    cagr_ratio = cagr(returns, periods=periods)
    max_dd = max_drawdown(returns)

    # 返回 CAGR 除以最大回撤绝对值的比率
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    """
    计算溃疡指数（下行风险度量）。

    溃疡指数衡量回撤的深度和持续时间，
    提供了下行风险的全面度量。
    计算方式为回撤平方均值的平方根。

    参数:
        returns (pd.Series): 要分析的收益率序列

    返回:
        float: 溃疡指数值

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> ulcer = ulcer_index(returns)
        >>> print(f"溃疡指数: {ulcer:.4f}")
    """
    # 将收益率转换为回撤序列
    dd = to_drawdown_series(returns)

    # 计算回撤的均方根
    return _np.sqrt(_np.divide((dd**2).sum(), returns.shape[0] - 1))


def ulcer_performance_index(returns, rf=0):
    """
    计算溃疡绩效指数（UPI）。

    UPI 使用溃疡指数作为风险度量而非标准差
    来衡量风险调整收益。它为具有
    显著回撤的策略提供了更好的度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（默认：0）

    返回:
        float: 溃疡绩效指数

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> upi_value = ulcer_performance_index(returns)
        >>> print(f"溃疡绩效指数: {upi_value:.4f}")
    """
    # 计算超额收益除以溃疡指数
    ulcer = ulcer_index(returns)
    
    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(ulcer, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return (comp(returns) - rf) / ulcer.replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if ulcer == 0:
            return _np.nan
        return (comp(returns) - rf) / ulcer


def upi(returns, rf=0):
    """
    计算溃疡绩效指数（UPI）。

    这是 ulcer_performance_index() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（默认：0）

    返回:
        float: 溃疡绩效指数
    """
    return ulcer_performance_index(returns, rf)


def serenity_index(returns, rf=0):
    """
    计算宁静指数。

    宁静指数是一种综合风险调整收益度量，
    将溃疡指数与下行风险考虑相结合。
    它提供了策略表现的更全面视图。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（默认：0）

    返回:
        float: 宁静指数

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> serenity = serenity_index(returns)
        >>> print(f"宁静指数: {serenity:.4f}")

    注意:
        基于 KeyQuant 白皮书：
        https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf
    """
    # 将收益率转换为回撤序列
    dd = to_drawdown_series(returns)

    # 使用回撤的条件在险值计算陷阱度量
    std_returns = returns.std()
    
    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(std_returns, _pd.Series):
        # DataFrame 输入 - 逐元素操作
        pitfall = -cvar(dd) / std_returns.replace(0, _np.nan)
        denominator = ulcer_index(returns) * pitfall
        return (returns.sum() - rf) / denominator.replace(0, _np.nan)
    else:
        # Series 输入 - 标量操作
        if std_returns == 0:
            return _np.nan

        cvar_val = cvar(dd)
        ulcer_val = ulcer_index(returns)

        # 处理这些可能返回 Series/数组的情况
        if hasattr(cvar_val, '__len__') and len(cvar_val) == 1:
            cvar_val = float(cvar_val.iloc[0] if hasattr(cvar_val, 'iloc') else cvar_val[0])
        if hasattr(ulcer_val, '__len__') and len(ulcer_val) == 1:
            ulcer_val = float(ulcer_val.iloc[0] if hasattr(ulcer_val, 'iloc') else ulcer_val[0])

        pitfall = -cvar_val / std_returns
        denominator = ulcer_val * pitfall

        if denominator == 0:
            return _np.nan
        return (returns.sum() - rf) / denominator


def risk_of_ruin(returns, prepare_returns=True):
    """
    计算崩溃风险（损失所有资金的概率）。

    本函数基于胜率和交易/期间数量
    估算损失所有投资资金的可能性。
    这对于仓位调整和风险管理很有用。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 崩溃风险概率（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> ror_value = risk_of_ruin(returns)
        >>> print(f"崩溃风险: {ror_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算胜率
    wins = win_rate(returns)

    # 使用赌徒破产公式计算崩溃风险
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    """
    计算崩溃风险（损失所有资金的概率）。

    这是 risk_of_ruin() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列

    返回:
        float: 崩溃风险概率（0-1 刻度）
    """
    return risk_of_ruin(returns)


def value_at_risk(
    returns: Returns,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    计算每日风险价值（VaR）。

    VaR 估算在给定时间范围内和指定置信水平下
    的最大预期损失，使用方差-协方差方法。

    参数:
        returns (pd.Series): 要分析的收益率序列
        sigma (float): 波动率乘数（默认：1）
        confidence (float): 置信水平（0.95 = 95%，默认：0.95）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 风险价值（负值，表示损失）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> var_value = value_at_risk(returns, confidence=0.95)
        >>> print(f"95% VaR: {var_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算均值并调整波动率
    mu = returns.mean()
    sigma *= returns.std()

    # 如果需要，将百分比置信度转换为小数
    if confidence > 1:
        confidence = confidence / 100

    # 使用正态分布逆 CDF 计算 VaR
    return _norm.ppf(1 - confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    计算每日风险价值（VaR）。

    这是 value_at_risk() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        sigma (float): 波动率乘数（默认：1）
        confidence (float): 置信水平（0.95 = 95%，默认：0.95）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 风险价值（负值，表示损失）
    """
    return value_at_risk(returns, sigma, confidence, prepare_returns)


def conditional_value_at_risk(
    returns: Returns,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    计算条件风险价值（CVaR），也称为预期 shortfall。

    CVaR 衡量当损失超过 VaR 阈值时的预期损失。
    它量化了投资面临的尾部风险，提供了比 VaR 本身
    更全面的风险度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        sigma (float): 波动率乘数（默认：1）
        confidence (float): 置信水平（0.95 = 95%，默认：0.95）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 条件风险价值（VaR 之外的预期损失）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> cvar_value = conditional_value_at_risk(returns, confidence=0.95)
        >>> print(f"95% CVaR: {cvar_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 处理 Series 和 DataFrame 输入
    if isinstance(returns, _pd.DataFrame):
        # 对于 DataFrame，分别计算每列的 CVaR
        result = {}
        for col in returns.columns:
            col_returns = returns[col]
            # 计算该特定列的 VaR
            col_var = value_at_risk(col_returns, sigma, confidence, prepare_returns=False)
            below_var = col_returns[col_returns < col_var]
            c_var_col = below_var.mean() if len(below_var) > 0 else _np.nan
            result[col] = c_var_col if not _np.isnan(c_var_col) else col_var
        return _pd.Series(result)
    else:
        # 对于 Series，计算 VaR 阈值
        var = value_at_risk(returns, sigma, confidence)
        c_var = returns[returns < var].values.mean()
        # 如果有效则返回 CVaR，否则返回 VaR
        return c_var if ~_np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    计算条件风险价值（CVaR）。

    这是 conditional_value_at_risk() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        sigma (float): 波动率乘数（默认：1）
        confidence (float): 置信水平（0.95 = 95%，默认：0.95）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 条件风险价值
    """
    return conditional_value_at_risk(returns, sigma, confidence, prepare_returns)


def expected_shortfall(returns, sigma=1, confidence=0.95):
    """
    计算预期 shortfall（ES），也称为 CVaR。

    这是 conditional_value_at_risk() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        sigma (float): 波动率乘数（默认：1）
        confidence (float): 置信水平（0.95 = 95%，默认：0.95）

    返回:
        float: 预期 shortfall
    """
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95, prepare_returns=True):
    """
    计算右尾与左尾之间的尾部比率。

    本函数测量收益率分布右尾（95%）与左尾（5%）
    的比率，提供极端收益不对称性的洞察。
    更高的值表示更有利的尾部特征。

    参数:
        returns (pd.Series): 要分析的收益率序列
        cutoff (float): 尾部分析的百分位分位数（默认：0.95）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 尾部比率（右尾 / 左尾）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> tail_r = tail_ratio(returns)
        >>> print(f"尾部比率: {tail_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算右尾与左尾的比率
    upper_quantile = returns.quantile(cutoff)
    lower_quantile = returns.quantile(1 - cutoff)
    
    # 处理边缘情况：NaN 值或零分母
    # 检查结果是否为 Series（DataFrame 输入）或标量（Series 输入）
    if isinstance(upper_quantile, _pd.Series):
        # 处理 DataFrame 输入 - 逐元素应用
        result = _pd.Series(index=upper_quantile.index, dtype=float)
        for col in upper_quantile.index:
            if _pd.isna(upper_quantile[col]) or _pd.isna(lower_quantile[col]) or lower_quantile[col] == 0:
                result[col] = _np.nan
            else:
                result[col] = abs(upper_quantile[col] / lower_quantile[col])
        return result
    else:
        # 处理 Series 输入 - 标量值
        if _pd.isna(upper_quantile) or _pd.isna(lower_quantile) or lower_quantile == 0:
            return _np.nan
        return abs(upper_quantile / lower_quantile)


def payoff_ratio(returns, prepare_returns=True):
    """
    计算盈亏比（平均盈利 / 平均亏损）。

    本函数衡量平均盈利收益率与平均亏损收益率的比率，
    提供个别交易或期间的风险回报特征洞察。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 盈亏比（平均盈利 / 平均亏损绝对值）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> payoff_r = payoff_ratio(returns)
        >>> print(f"盈亏比: {payoff_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算平均盈利与平均亏损绝对值的比率
    avg_loss_val = avg_loss(returns)
    avg_win_val = avg_win(returns)
    
    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(avg_loss_val, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        # 在 replace 之前使用 abs() 正确处理负值
        return avg_win_val / abs(avg_loss_val).replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if avg_loss_val == 0:
            return _np.nan
        return avg_win_val / abs(avg_loss_val)


def win_loss_ratio(returns, prepare_returns=True):
    """
    计算胜负比（平均盈利 / 平均亏损）。

    这是 payoff_ratio() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 胜负比
    """
    return payoff_ratio(returns, prepare_returns)


def profit_ratio(returns, prepare_returns=True):
    """
    计算利润比（胜率比 / 亏损比）。

    本函数衡量盈利频率与亏损频率的比率，
    提供盈利期间一致性的洞察。

    参数:
        returns (pd.Series or pd.DataFrame): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float or pd.Series: 利润比

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> profit_r = profit_ratio(returns)
        >>> print(f"利润比: {profit_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    def _profit_ratio(ret):
        # 分隔盈利和亏损
        wins = ret[ret >= 0]
        loss = ret[ret < 0]

        # 处理边缘情况
        win_count = len(wins)
        loss_count = len(loss)

        if win_count == 0:
            return 0.0
        if loss_count == 0:
            return _np.nan

        # 计算盈亏比率
        win_ratio = abs(wins.mean() / win_count) if win_count > 0 else 0
        loss_ratio = abs(loss.mean() / loss_count) if loss_count > 0 else 0

        if loss_ratio == 0:
            return _np.nan
        return win_ratio / loss_ratio

    # 处理 DataFrame，为每列应用函数
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(_profit_ratio)

    return _profit_ratio(returns)


def profit_factor(returns, prepare_returns=True):
    """
    计算利润因子（总盈利 / 总亏损）。

    本函数衡量总盈利收益率与总亏损收益率的比率，
    提供策略整体盈利能力的洞察。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 利润因子（总盈利 / 总亏损）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> pf = profit_factor(returns)
        >>> print(f"利润因子: {pf:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算总盈利和总亏损
    wins_sum = returns[returns >= 0].sum()
    losses_sum = abs(returns[returns < 0].sum())

    # 处理 Series 和标量情况
    if isinstance(losses_sum, _pd.Series):
        result = wins_sum / losses_sum
        # 将无穷值替换为 0
        result = result.replace([_np.inf, -_np.inf], 0)
        return result
    else:
        # 处理除以零的情况
        if losses_sum == 0:
            return 0.0 if wins_sum == 0 else float('inf')
        return wins_sum / losses_sum


def cpc_index(returns, prepare_returns=True):
    """
    计算 CPC 指数（利润因子 * 胜率 * 胜负比）。

    CPC 指数是一种综合表现度量，
    将利润因子、胜率和胜负比相结合，
    为策略评估提供单一指标。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: CPC 指数

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> cpc = cpc_index(returns)
        >>> print(f"CPC 指数: {cpc:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算复合指标
    return profit_factor(returns) * win_rate(returns) * win_loss_ratio(returns)


def common_sense_ratio(returns, prepare_returns=True):
    """
    计算常识比率（利润因子 * 尾部比率）。

    该比率将利润因子与尾部比率相结合，
    提供同时考虑盈利能力和尾部风险特征的度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 常识比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> csr = common_sense_ratio(returns)
        >>> print(f"常识比率: {csr:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算复合指标
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=0.99, prepare_returns=True):
    """
    计算异常盈利比率。

    本函数计算收益率的第99百分位
    与平均正收益率的比率，
    显示异常盈利对整体表现的贡献程度。

    参数:
        returns (pd.Series): 要分析的收益率序列
        quantile (float): 异常值阈值分位数（默认：0.99）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 异常盈利比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> outlier_win_r = outlier_win_ratio(returns)
        >>> print(f"异常盈利比率: {outlier_win_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算高分位数与平均正收益率的比率
    positive_mean = returns[returns >= 0].mean()
    quantile_val = returns.quantile(quantile)  # DataFrame 返回 Series，Series 返回标量

    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(positive_mean, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return quantile_val / positive_mean.replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if _pd.isna(positive_mean) or positive_mean == 0:
            return _np.nan
        return quantile_val / positive_mean


def outlier_loss_ratio(returns, quantile=0.01, prepare_returns=True):
    """
    计算异常亏损比率。

    本函数计算收益率的第1百分位
    与平均负收益率的比率，
    显示异常亏损对整体风险的贡献程度。

    参数:
        returns (pd.Series): 要分析的收益率序列
        quantile (float): 异常值阈值分位数（默认：0.01）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 异常亏损比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> outlier_loss_r = outlier_loss_ratio(returns)
        >>> print(f"异常亏损比率: {outlier_loss_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算低分位数与平均负收益率的比率
    negative_mean = returns[returns < 0].mean()
    quantile_val = returns.quantile(quantile)  # DataFrame 返回 Series，Series 返回标量

    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(negative_mean, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return quantile_val / negative_mean.replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if _pd.isna(negative_mean) or negative_mean == 0:
            return _np.nan
        return quantile_val / negative_mean


def recovery_factor(returns, rf=0.0, prepare_returns=True):
    """
    计算恢复因子（总收益率 / 最大回撤）。

    本函数通过将总收益率与经历的最大回撤进行比较，
    衡量策略从回撤中恢复的速度。
    更高的值表示更好的恢复特征。

    参数:
        returns (pd.Series): 要分析的收益率序列
        rf (float): 无风险利率（默认：0.0）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 恢复因子

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rf_value = recovery_factor(returns)
        >>> print(f"恢复因子: {rf_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算总超额收益率
    total_returns = returns.sum() - rf

    # 计算最大回撤
    max_dd = max_drawdown(returns)

    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(max_dd, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return abs(total_returns) / abs(max_dd).replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if max_dd == 0:
            return _np.nan
        return abs(total_returns) / abs(max_dd)


def risk_return_ratio(returns, prepare_returns=True):
    """
    计算风险回报比率（平均收益率 / 标准差）。

    本函数计算不考虑无风险利率的夏普比率，
    提供每单位风险的简单收益度量。

    参数:
        returns (pd.Series): 要分析的收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 风险回报比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rrr = risk_return_ratio(returns)
        >>> print(f"风险回报比率: {rrr:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 计算平均收益率除以标准差
    std = returns.std()
    
    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(std, _pd.Series):
        # DataFrame 输入 - 逐元素除法，带零保护
        return returns.mean() / std.replace(0, _np.nan)
    else:
        # Series 输入 - 标量除法
        if std == 0:
            return _np.nan
        return returns.mean() / std


def _get_baseline_value(prices):
    """
    确定用于回撤计算的适当基准值。

    本函数分析价格序列以确定应代表"无回撤"
    （即起始权益）的正确基准值。

    参数:
        prices (pd.Series): 价格序列

    返回:
        float: 用于回撤计算的基准值
    """
    if len(prices) == 0:
        return 1.0

    # 处理 Series 和 DataFrame 情况
    if isinstance(prices, _pd.DataFrame):
        # 如果 prices 是 DataFrame，确保它至少有一列
        if prices.shape[1] == 0:
            return 1.0  # 没有列的空 DataFrame 的默认基准
        # 获取第一列的第一个值
        first_price = prices.iat[0, 0]
    else:
        # 如果 prices 是 Series，直接获取第一个值
        first_price = prices.iloc[0]

    # 如果第一个价格远大于 1，它可能来自 to_prices 转换
    # to_prices 函数使用 base * (1 + compsum)，因此我们确定适当的基准
    if first_price > 1000:
        # 这表明它来自带有大基准的 to_prices（默认 1e5）
        # 但是，我们应该使用更合理的基准进行回撤计算
        # 我们将使用与价格相同的比例，但表示"无损失"基准
        return 1e5
    elif first_price > 10:
        # 较小的基准值比例
        return 100.0
    else:
        # 正常价格比例，使用 1.0 作为基准
        return 1.0


def max_drawdown(prices: Returns) -> float:
    """
    计算从峰值到谷值的最大回撤。

    本函数计算从峰值到后续谷值观察到的最大损失，
    表示为百分比。它通过建立适当的基准
    来处理第一个收益率为负的边缘情况。

    参数:
        prices (pd.Series): 价格序列或累计收益率

    返回:
        float: 最大回撤（负值）

    示例:
        >>> prices = pd.Series([100, 110, 105, 120, 115])
        >>> max_dd = max_drawdown(prices)
        >>> print(f"最大回撤: {max_dd:.4f}")
    """
    validate_input(prices)

    # 准备价格（如需要从收益率转换）
    prices = _utils._prepare_prices(prices)

    if len(prices) == 0:
        return 0.0

    # 处理边缘情况：如果第一个值代表相对于基准的损失
    # 添加一个虚基线值以确保正确的回撤计算
    try:
        time_delta = prices.index.freq or _pd.Timedelta(days=1)
    except Exception:
        time_delta = _pd.Timedelta(days=1)

    phantom_date = prices.index[0] - time_delta

    # 确定适当的基准值
    baseline_value = _get_baseline_value(prices)

    # 创建带虚基线的扩展序列
    extended_prices = prices.copy()
    extended_prices.loc[phantom_date] = baseline_value
    extended_prices = extended_prices.sort_index()

    # 使用虚基线计算回撤
    return (extended_prices / extended_prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    """
    将收益率序列转换为回撤序列。

    本函数将收益率序列转换为回撤序列，
    显示每个时间点相对于峰值的下降。
    它通过建立适当的基准
    来处理第一个收益率为负的边缘情况。

    参数:
        returns (pd.Series): 要转换的收益率序列

    返回:
        pd.Series: 回撤序列（负值显示相对于峰值的下降）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> dd_series = to_drawdown_series(returns)
        >>> print(dd_series)
    """
    validate_input(returns)

    # 将收益率转换为价格
    prices = _utils._prepare_prices(returns)

    if len(prices) == 0:
        return _pd.Series([], dtype=float, index=returns.index)

    # 处理边缘情况：如果第一个值代表相对于基准的损失
    # 添加一个虚基线值以确保正确的回撤计算
    try:
        time_delta = prices.index.freq or _pd.Timedelta(days=1)
    except Exception:
        time_delta = _pd.Timedelta(days=1)

    phantom_date = prices.index[0] - time_delta

    # 确定适当的基准值
    baseline_value = _get_baseline_value(prices)

    # 创建带虚基线的扩展序列
    extended_prices = prices.copy()
    extended_prices.loc[phantom_date] = baseline_value
    extended_prices = extended_prices.sort_index()

    # 使用虚基线计算回撤序列
    dd = extended_prices / _np.maximum.accumulate(extended_prices) - 1.0

    # 移除虚点并返回原始时间序列
    dd = dd.drop(phantom_date)

    # 清理无穷大和零值
    return dd.replace([_np.inf, -_np.inf, -0], 0)  # type: ignore[attr-defined]


def kelly_criterion(returns, prepare_returns=True):
    """
    根据凯利准则计算建议投入给定策略的最大资本比例。

    基于凯利准则（http://en.wikipedia.org/wiki/Kelly_criterion）
    计算应分配给该策略的最大资本比例。
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    # 处理 DataFrame 输入的 Series 情况和 Series 输入的标量情况
    if isinstance(win_loss_ratio, _pd.Series):
        # DataFrame 输入 - 逐元素操作，带零/NaN 保护
        # 将 0 和 NaN 值替换为 NaN 以避免除法问题
        win_loss_ratio_safe = win_loss_ratio.replace(0, _np.nan)
        return ((win_loss_ratio_safe * win_prob) - lose_prob) / win_loss_ratio_safe
    else:
        # Series 输入 - 标量操作
        if win_loss_ratio == 0 or _pd.isna(win_loss_ratio):
            return _np.nan
        return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====


def r_squared(returns, benchmark, prepare_returns=True):
    """
    计算相对于基准的 R 平方（决定系数）。

    R 平方衡量收益率与基准的直线关系拟合程度。
    接近 1 的值表示与基准高度相关，
    而接近 0 的值表示更独立的表现。

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: R 平方值（0-1 刻度）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> r_sq = r_squared(returns, benchmark)
        >>> print(f"R 平方: {r_sq:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 准备基准以匹配收益率索引
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # 执行线性回归并提取相关系数
    _, _, r_val, _, _ = _linregress(
        returns, _utils._prepare_benchmark(benchmark, returns.index)
    )

    # 对相关系数平方以获得 R 平方
    return r_val**2


def r2(returns, benchmark):
    """
    计算相对于基准的 R 平方（决定系数）。

    这是 r_squared() 的简写形式，
    具有相同的参数和功能。

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列

    返回:
        float: R 平方值（0-1 刻度）
    """
    return r_squared(returns, benchmark)


def information_ratio(returns, benchmark, prepare_returns=True):
    """
    计算信息比率。

    信息比率衡量投资组合相对于基准的
    风险调整超额收益。计算方式为主动收益
    （收益 - 基准）除以跟踪误差（主动收益的标准差）。

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        float: 信息比率

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> info_ratio = information_ratio(returns, benchmark)
        >>> print(f"信息比率: {info_ratio:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 准备基准以匹配收益率索引
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # 计算主动收益（收益率 - 基准）
    diff_rets = returns - _utils._prepare_benchmark(benchmark, returns.index)

    # 计算跟踪误差（主动收益的标准差）
    std = diff_rets.std()

    # 返回信息比率（主动收益 / 跟踪误差）
    if std != 0:
        return diff_rets.mean() / diff_rets.std()
    return 0


def greeks(returns, benchmark, periods=252.0, prepare_returns=True):
    """
    计算相对于基准的投资组合希腊值（阿尔法和贝塔）。

    本函数计算基准比较的关键投资组合指标：
    - 阿尔法：调整系统性风险（贝塔）后的超额收益
    - 贝塔：对基准变动的敏感度（系统性风险）

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列
        periods (float): 年化阿尔法的周期数（默认：252.0）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.Series: 包含 'alpha' 和 'beta' 值的序列

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> portfolio_greeks = greeks(returns, benchmark)
        >>> print(f"阿尔法: {portfolio_greeks['alpha']:.4f}")
        >>> print(f"贝塔: {portfolio_greeks['beta']:.4f}")
    """
    # 数据准备
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)
    # ----------------------------

    # 计算收益率与基准之间的协方差矩阵
    matrix = _np.cov(returns, benchmark)

    # 计算贝塔（对基准变动的敏感度）
    if matrix[1, 1] == 0:
        beta = _np.nan
    else:
        beta = matrix[0, 1] / matrix[1, 1]

    # 计算阿尔法（调整贝塔后的超额收益）
    alpha = returns.mean() - beta * benchmark.mean()

    # 年化阿尔法
    alpha = alpha * periods

    # 以序列形式返回结果
    return _pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
            # "vol": _np.sqrt(matrix[0, 0]) * _np.sqrt(periods)
        }
    ).fillna(0)


def rolling_greeks(returns, benchmark, periods=252, prepare_returns=True):
    """
    计算随时间变化的滚动希腊值（阿尔法和贝塔）。

    本函数使用滚动窗口计算随时间变化的阿尔法和贝塔，
    显示投资组合对基准的敏感度如何随时间变化。
    可用于分析策略稳定性和状态变化。

    参数:
        returns (pd.Series): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列
        periods (int): 滚动窗口大小（默认：252，约1年）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.DataFrame: 包含随时间变化的 'alpha' 和 'beta' 列的 DataFrame

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> rolling_greeks_df = rolling_greeks(returns, benchmark, periods=3)
        >>> print(rolling_greeks_df)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 创建用于滚动计算的组合 DataFrame
    df = _pd.DataFrame(
        data={
            "returns": returns,
            "benchmark": _utils._prepare_benchmark(benchmark, returns.index),
        }
    )

    # 用 0 填充 NaN 值以确保计算稳定性
    df = df.fillna(0)

    # 计算滚动相关性和标准差
    corr = df.rolling(int(periods)).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(int(periods)).std()

    # 计算滚动贝塔（防止除以零）
    beta = corr * std["returns"] / std["benchmark"].replace(0, _np.nan)

    # 计算滚动阿尔法（滚动版本不年化）
    alpha = df["returns"].mean() - beta * df["benchmark"].mean()

    # 返回带有滚动希腊值的 DataFrame
    return _pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})


def compare(
    returns,
    benchmark,
    aggregate=None,
    compounded=True,
    round_vals=None,
    prepare_returns=True,
):
    """
    比较不同周期的收益率与基准。

    本函数提供投资组合收益率与基准表现
    在各种聚合周期（日、周、月、季、年）
    的综合比较。

    参数:
        returns (pd.Series or pd.DataFrame): 要分析的收益率序列
        benchmark (pd.Series): 用于比较的基准收益率序列
        aggregate (str): 聚合周期 ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): 是否复合收益率（默认：True）
        round_vals (int): 四舍五入的小数位数（默认：None）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.DataFrame: 比较 DataFrame，包含列：
            - Benchmark: 每个周期的基准收益率
            - Returns: 每个周期的投资组合收益率
            - Multiplier: 投资组合收益率 / 基准收益率
            - Won: 如果投资组合跑赢基准则 '+'，否则 '-'

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> comparison = compare(returns, benchmark)
        >>> print(comparison)
    """
    _utils.require_local_returns(returns, "returns")
    if benchmark is not None:
        _utils.require_local_returns(benchmark, "benchmark")

    # 规范化收益率的时区以确保一致的比较
    # 如果是时区感知的则转换为 UTC，然后使其 naive
    # 这必须在 prepare_returns 之前发生以避免问题
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns = returns.tz_convert('UTC').tz_localize(None)
    
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 如果基准不是字符串，首先规范化其时区
    if benchmark is not None and not isinstance(benchmark, str):
        if hasattr(benchmark.index if isinstance(benchmark, _pd.Series) else benchmark[benchmark.columns[0]].index, 'tz'):
            if isinstance(benchmark, _pd.Series) and benchmark.index.tz is not None:
                benchmark = benchmark.tz_convert('UTC').tz_localize(None)
            elif isinstance(benchmark, _pd.DataFrame) and benchmark[benchmark.columns[0]].index.tz is not None:
                for col in benchmark.columns:
                    benchmark[col] = benchmark[col].tz_convert('UTC').tz_localize(None)
    
    # 存储原始基准以进行正确的聚合
    # 这保留了可能在非交易日的收益率
    if isinstance(benchmark, str):
        _utils.require_local_returns(benchmark, "benchmark")
    elif isinstance(benchmark, _pd.DataFrame):
        benchmark_original = benchmark[benchmark.columns[0]].copy()
    else:
        benchmark_original = benchmark.copy() if benchmark is not None else None
    
    # 也为 benchmark_original 规范化时区（如果它被下载了的话）
    if benchmark_original is not None and hasattr(benchmark_original.index, 'tz') and benchmark_original.index.tz is not None:
        benchmark_original = benchmark_original.tz_convert('UTC').tz_localize(None)
    
    # 准备基准以匹配收益率索引用于其他计算
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # 处理 Series 输入
    if isinstance(returns, _pd.Series):
        # 聚合收益率并使用原始基准进行聚合
        # 这确保我们不会丢失非交易日的基准收益率
        if benchmark_original is not None:
            benchmark_agg = _utils.aggregate_returns(benchmark_original, aggregate, compounded) * 100
        else:
            benchmark_agg = _utils.aggregate_returns(benchmark, aggregate, compounded) * 100
        returns_agg = _utils.aggregate_returns(returns, aggregate, compounded) * 100

        # 创建比较 DataFrame
        data = _pd.DataFrame(
            data={
                "Benchmark": benchmark_agg,
                "Returns": returns_agg,
            }
        )

        # 计算表现乘数和胜负指示器
        # 防止基准中的除以零
        data["Multiplier"] = data["Returns"] / data["Benchmark"].replace(0, _np.nan)
        data["Won"] = _np.where(data["Returns"] >= data["Benchmark"], "+", "-")

    # 处理 DataFrame 输入（多个策略）
    elif isinstance(returns, _pd.DataFrame):
        # 使用原始数据聚合基准以保留非交易日收益率
        if benchmark_original is not None:
            bench = {
                "Benchmark": _utils.aggregate_returns(benchmark_original, aggregate, compounded) * 100
            }
        else:
            bench = {
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded) * 100
            }

        # 聚合每个策略列
        strategy = {
            "Returns_" + str(i): _utils.aggregate_returns(returns[col], aggregate, compounded) * 100
            for i, col in enumerate(returns.columns)
        }

        # 合并为单个 DataFrame
        data = _pd.DataFrame(data={**bench, **strategy})

    # 如果指定则应用四舍五入
    if round_vals is not None:
        return _np.round(data, round_vals)

    return data


def monthly_returns(returns, eoy=True, compounded=True, prepare_returns=True):
    """
    以透视表格式计算月收益率。

    本函数创建一个矩阵，显示不同年份的每月收益率，
    可轻松识别季节性模式并比较不同时间段的表现。

    参数:
        returns (pd.Series or pd.DataFrame): 要分析的收益率序列
        eoy (bool): 是否包含年末总计（默认：True）
        compounded (bool): 是否复合收益率（默认：True）
        prepare_returns (bool): 是否先准备收益率（默认：True）

    返回:
        pd.DataFrame: 月收益率矩阵，以年份为行、月份为列。
                     如果 eoy=True，包含 'EOY' 列的年度收益率。

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
        ...                    index=pd.date_range('2023-01-01', periods=5, freq='M'))
        >>> monthly_rets = monthly_returns(returns)
        >>> print(monthly_rets)
    """
    # 处理 DataFrame 输入，选择适当的列
    if isinstance(returns, _pd.DataFrame):
        warn(
            "传入了 Pandas DataFrame（期望 Series）。"
            "只会使用第一列。"
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # 存储原始收益率用于年末计算
    original_returns = returns.copy()

    # 按年月分组并聚合收益率
    returns = _pd.DataFrame(
        _utils.group_returns(returns, returns.index.strftime("%Y-%m-01"), compounded)
    )

    # 设置 DataFrame 结构
    returns.columns = ["Returns"]
    returns.index = _pd.to_datetime(returns.index)

    # 提取年份和月份用于透视表
    returns["Year"] = returns.index.strftime("%Y")
    returns["Month"] = returns.index.strftime("%b")

    # 创建以年份为行、月份为列的透视表
    returns = returns.pivot(index="Year", columns="Month", values="Returns").fillna(0)

    # 确保所有月份都出现在 DataFrame 中
    for month in [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # 按日历月份排序列
    returns = returns[
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    ]

    # 如果请求则添加年末总计
    if eoy:
        returns["eoy"] = _utils.group_returns(
            original_returns, original_returns.index.year, compounded=compounded  # type: ignore
        ).values

    # 格式化列名为大写
    returns.columns = map(lambda x: str(x).upper(), returns.columns)  # type: ignore
    returns.index.name = None

    return returns


def drawdown_details(drawdown):
    """
    计算每个回撤期间的详细回撤统计。

    本函数分析回撤序列以提供每个单独回撤期间的
    全面统计，包括开始/结束日期、持续时间、
    最大回撤和第99百分位回撤。

    参数:
        drawdown (pd.Series or pd.DataFrame): 要分析的回撤序列

    返回:
        pd.DataFrame: 详细回撤统计，包含列：
            - start: 回撤期间的开始日期
            - valley: 最大回撤日期
            - end: 回撤期间的结束日期
            - days: 持续天数
            - max drawdown: 最大回撤百分比
            - 99% max drawdown: 第99百分位回撤（排除异常值）

    示例:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> dd_series = to_drawdown_series(returns)
        >>> dd_details = drawdown_details(dd_series)
        >>> print(dd_details)
    """

    def _drawdown_details(drawdown):
        """
        计算单个回撤序列的回撤详情。

        本内部函数处理单个回撤序列以提取
        每个回撤期间的详细统计。
        """
        # 标记无回撤期间（回撤 = 0）
        no_dd = drawdown == 0

        # 提取回撤开始日期（每个回撤期间的第一天）
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.values].index)

        # 提取回撤结束日期（每个回撤期间的最后一天）
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.values].index)

        # 如果未找到回撤则返回空 DataFrame
        if not starts:
            return _pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # 处理边缘情况：回撤序列从回撤开始
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # 处理边缘情况：序列以回撤结束
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # 为每个回撤期间构建详细统计
        data = []
        for i, _ in enumerate(starts):
            # 提取该期间的回撤
            dd = drawdown[starts[i]:ends[i]]

            # 计算 99% 回撤（排除异常值）
            clean_dd = -remove_outliers(-dd, 0.99)

            # 编译该回撤期间的统计
            data.append(
                (
                    starts[i],                          # 开始日期
                    dd.idxmin(),                       # 谷值日期（最大回撤）
                    ends[i],                           # 结束日期
                    (ends[i] - starts[i]).days + 1,   # 持续天数
                    dd.min() * 100,                    # 最大回撤 %
                    clean_dd.min() * 100,              # 99% 最大回撤 %
                )
            )

        # 使用结果创建 DataFrame
        df = _pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )

        # 格式化数据类型
        df["days"] = df["days"].astype(int)
        df["max drawdown"] = df["max drawdown"].astype(float)
        df["99% max drawdown"] = df["99% max drawdown"].astype(float)

        # 将日期格式化为字符串
        df["start"] = df["start"].dt.strftime("%Y-%m-%d")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d")
        df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

        return df

    # 处理 DataFrame 输入，分别处理每列
    if isinstance(drawdown, _pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return safe_concat(_dfs, axis=1)

    return _drawdown_details(drawdown)

