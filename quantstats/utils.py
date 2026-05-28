#!/usr/bin/env python
#
# QuantStats: 量化投资者的投资组合分析工具
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
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

import io as _io
import datetime as _dt
import warnings as _warnings
import pandas as _pd
import numpy as _np
from ._compat import safe_concat, safe_resample
import inspect
import threading

# 收益率数据的类型别名
Returns = _pd.Series | _pd.DataFrame
"""收益率数据的类型别名：可以是 pandas Series 或 DataFrame。"""


# QuantStats 自定义异常类
class QuantStatsError(Exception):
    """QuantStats 的基异常类"""

    pass


class DataValidationError(QuantStatsError):
    """当输入数据验证失败时引发"""

    pass


class CalculationError(QuantStatsError):
    """当计算失败时引发"""

    pass


class PlottingError(QuantStatsError):
    """当绘图操作失败时引发"""

    pass


class BenchmarkError(QuantStatsError):
    """当基准数据出现问题时引发"""

    pass


class LocalDataRequiredError(QuantStatsError):
    """本地化版本只允许传入本地收益率/净值数据，禁止自动联网下载行情。"""

    pass


def _local_data_only(message: str) -> None:
    """输出本地化输入提示并抛错，统一替代原 QuantStats 的在线下载行为。"""
    try:
        from utils.log_kit import logger

        logger.debug(message)
    except Exception:
        _warnings.warn(message, RuntimeWarning, stacklevel=2)
    raise LocalDataRequiredError(message)


def require_local_returns(data, name: str = "returns") -> None:
    """检查收益率输入是否为本地 pandas Series/DataFrame，不允许传 ticker 字符串。"""
    if isinstance(data, str):
        _local_data_only(
            f"QuantStats 本地化版本不支持用字符串自动下载 {name}: {data!r}。"
            f"请传入本地 pandas Series/DataFrame 收益率序列。"
        )
    if not isinstance(data, (_pd.Series, _pd.DataFrame)):
        _local_data_only(
            f"QuantStats 本地化版本要求 {name} 是 pandas Series/DataFrame，"
            f"当前收到 {type(data).__name__}。"
        )


def validate_input(data, allow_empty=False):
    """
    验证 QuantStats 函数的输入数据

    参数
    ----------
    data : pd.Series or pd.DataFrame
        要验证的输入数据
    allow_empty : bool, default False
        是否允许空数据集

    引发
    ------
    DataValidationError
        如果数据验证失败
    """
    if data is None:
        raise DataValidationError("输入数据不能为 None")

    if not isinstance(data, (_pd.Series, _pd.DataFrame)):
        raise DataValidationError(
            f"输入数据必须是 pandas Series 或 DataFrame，当前类型为 {type(data)}"
        )

    if not allow_empty and len(data) == 0:
        raise DataValidationError("输入数据不能为空")

    if not allow_empty and data.dropna().empty:
        raise DataValidationError("输入数据只包含 NaN 值")

    # 检查有效的日期索引
    if not isinstance(data.index, (_pd.DatetimeIndex, _pd.RangeIndex)):
        try:
            data.index = _pd.to_datetime(data.index)
        except Exception:
            raise DataValidationError("输入数据必须具有有效的日期时间索引")

    return True


# _prepare_returns 函数的缓存，带线程安全
_PREPARE_RETURNS_CACHE = {}
_CACHE_MAX_SIZE = 100
_cache_lock = threading.Lock()


def _generate_cache_key(data, rf, nperiods):
    """
    为 _prepare_returns 函数生成缓存键

    参数
    ----------
    data : pd.Series or pd.DataFrame
        用于生成哈希的输入数据
    rf : float
        无风险利率参数
    nperiods : int
        期间数量参数

    返回
    -------
    str or None
        缓存键字符串；如果哈希失败则返回 None
    """
    try:
        # 从数据创建哈希
        if isinstance(data, _pd.Series):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        elif isinstance(data, _pd.DataFrame):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        else:
            data_hash = hash(str(data))

        # 在键中包含参数
        key = f"{data_hash}_{rf}_{nperiods}"
        return key
    except (ValueError, TypeError, AttributeError, MemoryError):
        # 如果哈希失败，返回 None 以跳过缓存
        return None


def _clear_cache_if_full():
    """
    当缓存超过最大大小时清空缓存

    使用简单的 FIFO 策略，当缓存大小超过限制时保留最近一半的条目。
    """
    with _cache_lock:
        if len(_PREPARE_RETURNS_CACHE) >= _CACHE_MAX_SIZE:
            # 移除最旧的条目（简单 FIFO）- 保留最近一半
            keys_to_remove = list(_PREPARE_RETURNS_CACHE.keys())[:-(_CACHE_MAX_SIZE // 2)]
            for key in keys_to_remove:
                del _PREPARE_RETURNS_CACHE[key]


def _mtd(df):
    """
    筛选数据框至当月数据

    参数
    ----------
    df : pd.DataFrame or pd.Series
        具有日期时间索引的输入数据

    返回
    -------
    pd.DataFrame or pd.Series
        从当月开始筛选的数据
    """
    # 获取当月第一天作为字符串
    return df[df.index >= _dt.datetime.now().strftime("%Y-%m-01")]


def _qtd(df):
    """
    筛选数据框至当季数据

    参数
    ----------
    df : pd.DataFrame or pd.Series
        具有日期时间索引的输入数据

    返回
    -------
    pd.DataFrame or pd.Series
        从当季开始筛选的数据
    """
    date = _dt.datetime.now()
    # 检查当前是哪个季度（Q1: 1-3月，Q2: 4-6月，Q3: 7-9月，Q4: 10-12月）
    for q in [1, 4, 7, 10]:  # 每个季度的第一天月份
        if date.month <= q:
            return df[df.index >= _dt.datetime(date.year, q, 1).strftime("%Y-%m-01")]
    # 如果没有匹配的季度，默认返回当月数据
    return df[df.index >= date.strftime("%Y-%m-01")]


def _ytd(df):
    """
    筛选数据框至年初至今数据

    参数
    ----------
    df : pd.DataFrame or pd.Series
        具有日期时间索引的输入数据

    返回
    -------
    pd.DataFrame or pd.Series
        从今年开始筛选的数据
    """
    # 获取今年第一天作为字符串
    return df[df.index >= _dt.datetime.now().strftime("%Y-01-01")]


def _pandas_date(df, dates):
    """
    按指定日期筛选数据框

    参数
    ----------
    df : pd.DataFrame or pd.Series
        具有日期时间索引的输入数据
    dates : list or single date
        用于筛选的日期

    返回
    -------
    pd.DataFrame or pd.Series
        筛选后指定日期的数据
    """
    # 确保 dates 是列表以保持一致处理
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    """
    筛选数据框至当月数据

    参数
    ----------
    df : pd.DataFrame or pd.Series
        具有日期时间索引的输入数据

    返回
    -------
    pd.DataFrame or pd.Series
        当月筛选后的数据
    """
    n = _dt.datetime.now()
    # 从当月第一天到当前日期创建日期范围
    daterange = _pd.date_range(_dt.date(n.year, n.month, 1), n)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    """获取 pandas 中相对于另一行的最后 N 行 - 针对内存使用优化"""
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    # 使用字典推导式的更节省内存的方法
    # 直接列赋值
    result = df.copy()

    # 创建数据的滞后版本
    for i in range(1, shift):
        shifted = df.shift(i)
        # 重命名列以避免冲突
        shifted.columns = [f"{col}{i}" for col in shifted.columns]
        result = safe_concat([result, shifted], axis=1, sort=True)

    return result


def to_returns(prices: Returns, rf: float = 0.0) -> Returns:
    """
    从价格序列计算简单算术收益率

    参数
    ----------
    prices : pd.Series or pd.DataFrame
        价格数据
    rf : float, default 0.0
        无风险利率

    返回
    -------
    pd.Series or pd.DataFrame
        简单算术收益率
    """
    return _prepare_returns(prices, rf)


def to_prices(returns: Returns, base: float = 1e5) -> Returns:
    """
    将收益率序列转换为价格数据

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    base : float, default 1e5
        价格序列的起始基准值

    返回
    -------
    pd.Series or pd.DataFrame
        从收益率计算的价格数据
    """
    from . import stats as _stats  # 延迟导入以避免循环依赖

    # 通过填充 NaN 和替换无穷值来清理收益率数据
    returns = returns.copy().fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # 使用复合总和将收益率转换为价格
    return base + base * _stats.compsum(returns)


def log_returns(returns: Returns, rf: float = 0.0, nperiods: int | None = None) -> Returns:
    """
    to_log_returns 函数的简写

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    rf : float, default 0.0
        无风险利率
    nperiods : int, optional
        用于无风险利率转换的期间数量

    返回
    -------
    pd.Series or pd.DataFrame
        对数收益率
    """
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns: Returns, rf: float = 0.0, nperiods: int | None = None) -> Returns:
    """
    将收益率序列转换为对数收益率

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    rf : float, default 0.0
        无风险利率
    nperiods : int, optional
        用于无风险利率转换的期间数量

    返回
    -------
    pd.Series or pd.DataFrame
        计算为 ln(1 + 收益率) 的对数收益率
    """
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        # 计算对数收益率：ln(1 + 收益率)
        return _np.log(returns + 1).replace([_np.inf, -_np.inf], float("NaN"))  # type: ignore
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        from warnings import warn
        warn(f"转换为对数收益率时出错: {type(e).__name__}: {e}，返回 0.0")
        return 0.0


def exponential_stdev(returns, window=30, is_halflife=False):
    """
    计算收益率的指数加权标准差（波动率）

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    window : int, default 30
        指数加权的窗口大小
    is_halflife : bool, default False
        window 参数是否表示半衰期

    返回
    -------
    pd.Series or pd.DataFrame
        指数加权标准差
    """
    returns = _prepare_returns(returns)
    # 根据 is_halflife 标志设置半衰期参数
    halflife = window if is_halflife else None
    return returns.ewm(
        com=None, span=window, halflife=halflife, min_periods=window
    ).std()


def rebase(prices: Returns, base: float = 100.0) -> Returns:
    """
    将所有序列重定基数为给定的初始基准。
    这样可以更方便地比较/绘制不同序列。
    参数:
        * prices: 价格序列/数据框
        * base (number): 所有序列的起始值。
    """
    # 将价格标准化为从基准值开始
    return prices.dropna() / prices.dropna().iloc[0] * base


def group_returns(returns: Returns, groupby, compounded: bool = False) -> Returns:
    """
    按分组条件汇总收益率

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    groupby : grouper object
        Pandas groupby 对象或条件
    compounded : bool, default False
        是否复合收益率或使用简单求和

    返回
    -------
    pd.Series or pd.DataFrame
        分组后的收益率

    示例
    --------
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        from . import stats as _stats  # 延迟导入以避免循环依赖

        # 使用复合收益率计算
        return returns.groupby(groupby).apply(_stats.comp)
    # 对于非复合收益率使用简单求和
    return returns.groupby(groupby).sum()


def aggregate_returns(returns: Returns, period: str | None = None, compounded: bool = True) -> Returns:
    """
    根据指定时间段聚合收益率

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    period : str, optional
        聚合的时间周期（'month'、'quarter'、'year' 等）
    compounded : bool, default True
        是否复合收益率

    返回
    -------
    pd.Series or pd.DataFrame
        指定期间聚合后的收益率
    """
    # 聚合前规范化时区以保持一致性
    # 如果有时区感知，则转换为 UTC，然后变为 naive
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns = returns.tz_convert('UTC').tz_localize(None)
    
    # 如果未指定周期或为日周期，返回原始数据
    if period is None or "day" in period:
        return returns

    index = returns.index

    # 按月分组
    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    # 按季度分组
    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    # 按年分组（多种可能的周期字符串）
    if period == "YE" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    # 按周分组
    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    # 周末分组
    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    # 月末分组
    if "eom" in period or period == "ME":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    # 季末分组
    if "eoq" in period or period == "QE":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

    # 自定义周期分组（非字符串）
    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    # 默认：返回原始数据
    return returns


def to_excess_returns(returns: Returns, rf: float, nperiods: int | None = None) -> Returns:
    """
    通过从总收益率中减去无风险收益率来计算超额收益率

    参数:
        * returns (Series, DataFrame): 收益率
        * rf (float, Series, DataFrame): 无风险利率
        * nperiods (int): 可选。如果提供，将使用 deannualize 转换 rf 为不同频率
    返回:
        * excess_returns (Series, DataFrame): 收益率 - 无风险利率
    """
    # 将整数 rf 转换为浮点数以保持一致
    if isinstance(rf, int):
        rf = float(rf)

    # 如果 rf 是 series/dataframe，则将其与收益率索引对齐
    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]  # type: ignore

    # 如果提供了 nperiods，则去年度化 rf
    if nperiods is not None:
        # 去年度化
        rf = _np.power(1 + rf, 1.0 / nperiods) - 1.0

    # 计算超额收益率
    df = returns - rf
    df = df.tz_localize(None)
    return df


def _prepare_prices(data, base=1.0):
    """
    将收益率数据转换为价格并执行清理

    参数
    ----------
    data : pd.Series or pd.DataFrame
        输入数据（收益率或价格）
    base : float, default 1.0
        价格转换的基准值

    返回
    -------
    pd.Series or pd.DataFrame
        清理后的价格数据
    """
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # 缓存 dropna 操作以避免重复计算
            col_clean = data[col].dropna()
            # 检查数据是否像收益率（负值或值 < 1）
            if col_clean.min() <= 0 or col_clean.max() < 1:
                data[col] = to_prices(data[col], base)

    # 检查序列是否像收益率数据
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    # 通过填充 NaN 和替换无穷值来清理数据
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # 规范化时区信息以保持一致性
    # 如果有时区感知，则转换为 UTC，然后变为 naive
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data = data.tz_convert('UTC').tz_localize(None)
    return data


def _prepare_returns(data, rf=0.0, nperiods=None):
    """
    将价格数据转换为收益率并执行清理

    参数
    ----------
    data : pd.Series or pd.DataFrame
        输入数据（价格或收益率）
    rf : float, default 0.0
        无风险利率
    nperiods : int, optional
        用于无风险利率转换的期间数量

    返回
    -------
    pd.Series or pd.DataFrame
        清理后的收益率数据
    """
    # 首先尝试从缓存获取
    cache_key = _generate_cache_key(data, rf, nperiods)
    if cache_key:
        with _cache_lock:
            if cache_key in _PREPARE_RETURNS_CACHE:
                return _PREPARE_RETURNS_CACHE[cache_key].copy()

    data = data.copy()
    # 获取调用函数名以便条件处理
    function = inspect.stack()[1][3]

    # 处理 DataFrame 列
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # 缓存 dropna 操作以避免重复计算
            col_clean = data[col].dropna()
            # 检查数据是否像价格（正值 > 1）
            if col_clean.min() >= 0 and col_clean.max() > 1:
                data[col] = data[col].pct_change(fill_method=None)
    # 处理 Series 数据
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change(fill_method=None)

    # 清理数据 - 用 NaN 替换无穷值
    data = data.replace([_np.inf, -_np.inf], float("NaN"))

    # 填充 NaN 值为 0 并替换无穷值
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # 不需要超额收益率计算的函数
    unnecessary_function_calls = [
        "_prepare_benchmark",
        "cagr",
        "gain_to_pain_ratio",
        "rolling_volatility",
    ]

    # 如果 rf > 0 且函数需要，计算超额收益率
    if function not in unnecessary_function_calls:
        if rf > 0:
            result = to_excess_returns(data, rf, nperiods)
            # 缓存结果
            if cache_key:
                _clear_cache_if_full()
                with _cache_lock:
                    _PREPARE_RETURNS_CACHE[cache_key] = result.copy()
            return result

    # 规范化时区信息以保持一致性
    # 如果有时区感知，则转换为 UTC，然后变为 naive
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data = data.tz_convert('UTC').tz_localize(None)

    # 缓存结果
    if cache_key:
        _clear_cache_if_full()
        with _cache_lock:
            _PREPARE_RETURNS_CACHE[cache_key] = data.copy()

    return data


def download_returns(ticker, period="max", proxy=None):
    """
    本地化版本已禁用在线下载行情。

    原版 QuantStats 会通过 yfinance 下载 ticker 的价格并转为收益率；
    当前项目只支持用户显式传入本地 pandas Series/DataFrame 收益率。
    """
    _local_data_only(
        f"QuantStats 本地化版本已删除在线下载功能，不能 download_returns({ticker!r})。"
        "请先在项目回测中计算好收益率，再传入 qs.reports.html(returns, benchmark=...)。"
    )


def _prepare_benchmark(benchmark=None, period="max", rf=0.0, prepare_returns=True):
    """
    准备本地 benchmark 收益率，并传给 _prepare_returns()。

    当前本地化版本不支持 benchmark 使用字符串 ticker 自动下载。
    """
    if benchmark is None:
        return None

    # 本地化改造：benchmark 必须是本地 Series/DataFrame，不能再传入 ticker 字符串。
    if isinstance(benchmark, str):
        _local_data_only(
            f"QuantStats 本地化版本不支持 benchmark={benchmark!r} 自动下载。"
            "请传入本地 benchmark 收益率 Series/DataFrame。"
        )

    # 如果提供了 DataFrame，则提取第一列
    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

    # 如有需要，按策略周期对齐基准
    if isinstance(period, _pd.DatetimeIndex) and set(period) != set(benchmark.index):

        # 将基准调整为策略频率
        benchmark_prices = to_prices(benchmark, base=1)
        new_index = _pd.date_range(start=period[0], end=period[-1], freq="D")
        benchmark = (
            benchmark_prices.reindex(new_index, method="bfill")
            .reindex(period)
            .pct_change(fill_method=None)
            .fillna(0)
        )
        benchmark = benchmark[benchmark.index.isin(period)]

    # 规范化时区信息以便一致比较
    # 如果有时区感知，则转换为 UTC，然后变为 naive
    if hasattr(benchmark.index, 'tz') and benchmark.index.tz is not None:
        benchmark = benchmark.tz_convert('UTC').tz_localize(None)
    # 如果已经是时区 naive，则无需操作

    # 准备收益率或返回原始数据
    if prepare_returns:
        return _prepare_returns(benchmark.dropna(), rf=rf)
    return benchmark.dropna()


def _round_to_closest(val, res, decimals=None):
    """
    将值四舍五入到最接近的分辨率

    参数
    ----------
    val : float
        要四舍五入的值
    res : float
        要四舍五入到的分辨率
    decimals : int, optional
        小数位数

    返回
    -------
    float
        四舍五入后的值
    """
    # 如果未提供，从分辨率自动检测小数位数
    if decimals is None and "." in str(res):
        decimals = len(str(res).split(".")[1])
    return round(round(val / res) * res, decimals)


def _file_stream():
    """
    创建并返回文件流对象

    返回
    -------
    io.BytesIO
        用于处理字节的文件流对象
    """
    return _io.BytesIO()


def _in_notebook(matplotlib_inline=False):
    """
    识别当前环境（notebook、terminal 等）

    参数
    ----------
    matplotlib_inline : bool, default False
        是否启用 matplotlib inline 模式

    返回
    -------
    bool
        如果在 Jupyter notebook 中运行则返回 True，否则返回 False
    """
    try:
        # 获取 IPython shell 类名
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook 或 qtconsole
            if matplotlib_inline:
                get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore
            return True
        if shell == "TerminalInteractiveShell":
            # 运行 IPython 的终端
            return False
        # 其他类型 (?)
        return False
    except NameError:
        # 可能是标准 Python 解释器
        return False


def _count_consecutive(data):
    """
    统计数据中连续出现的次数（类似于 cumsum()，遇到零时重置）

    参数
    ----------
    data : pd.Series or pd.DataFrame
        要统计连续出现次数的输入数据

    返回
    -------
    pd.Series or pd.DataFrame
        带有连续计数的数据
    """

    def _count(data):
        # 按连续值分组并统计出现次数
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    # 通过处理每列来处理 DataFrame
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """
    用适当的符号格式化值字符串（用于绘图）

    参数
    ----------
    val : str or numeric
        要格式化的值

    返回
    -------
    str
        带 + 或 - 符号的格式化字符串
    """
    # 为正数值添加 + 符号，负数已包含 -
    return ("" if "-" in val else "+") + str(val)


def make_index(
    ticker_weights, rebalance="1ME", period="max", returns=None, match_dates=False
):
    """
    根据给定的 ticker 和权重创建指数。
    本地化版本必须传入包含全部 ticker 的 returns DataFrame，不再自动下载行情。

    参数:
        * ticker_weights (Dict): Python 字典，ticker 作为键，权重作为值
        * rebalance: Pandas 重采样间隔，None 表示从不重平衡
        * period: 要下载的收益率时间段
        * returns (Series, DataFrame): 可选。收益率。如果提供，
            首先检查给定 ticker 的收益率是否在此数据框中，
            如果不在，尝试用 yfinance 下载
    返回:
        * index_returns (Series, DataFrame): 指数的收益率
    """
    # 声明一个收益率变量
    index = None
    portfolio = {}

    # 遍历权重，获取每个 ticker 的收益率
    for ticker in ticker_weights.keys():
        if (returns is None) or (ticker not in returns.columns):
            # 本地化版本不允许缺失 ticker 后再联网补齐，缺什么就提示用户传什么。
            _local_data_only(
                f"make_index 缺少 {ticker!r} 的本地收益率列。"
                "请传入包含所有 ticker 的 returns DataFrame。"
            )
        else:
            ticker_returns = returns[ticker]

        portfolio[ticker] = ticker_returns

    # 创建指数成分时间序列
    index = _pd.DataFrame(portfolio).dropna()

    # 匹配日期从第一个非零日期开始
    if match_dates:
        index = index[max(index.ne(0).idxmax()):]

    # 处理无需重平衡的情况
    if rebalance is None:
        # 直接应用权重到收益率
        for ticker, weight in ticker_weights.items():
            index[ticker] = weight * index[ticker]
        return index.sum(axis=1)

    last_day = index.index[-1]

    # 将权重应用到每个 ticker 的收益率
    # 对于加权组合，每天的组合收益率是 (权重 * 资产收益率) 的总和
    for ticker, weight in ticker_weights.items():
        index[ticker] = weight * index[ticker]

    # 将加权资产收益率相加计算每日组合收益率
    portfolio_returns = index.sum(axis=1)

    # 移除所有值都为 NaN 的行
    portfolio_returns = portfolio_returns.dropna()
    return portfolio_returns[portfolio_returns.index <= last_day]


def make_portfolio(returns, start_balance=1e5, mode="comp", round_to=None):
    """
    从收益率计算组合的复合值

    参数
    ----------
    returns : pd.Series or pd.DataFrame
        收益率数据
    start_balance : float, default 1e5
        起始组合余额
    mode : str, default "comp"
        计算模式（"comp"、"cumsum"、"sum" 或其他）
    round_to : int, optional
        四舍五入的小数位数

    返回
    -------
    pd.Series or pd.DataFrame
        随时间变化的组合价值
    """
    returns = _prepare_returns(returns)

    # 根据模式计算组合价值
    if mode.lower() in ["cumsum", "sum"]:
        # 简单累计求和方法
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        # 复合收益率方法
        p1 = to_prices(returns, start_balance)
    else:
        # 每天固定金额方法
        comp_rev = (start_balance + start_balance * returns.shift(1)).fillna(
            start_balance
        ) * returns
        p1 = start_balance + comp_rev.cumsum()

    # 在前一天添加起始余额
    p0 = _pd.Series(data=start_balance, index=p1.index + _pd.Timedelta(days=-1))[:1]

    # 将起始余额与组合价值组合
    portfolio = safe_concat([p0, p1])

    # 处理 DataFrame 情况
    if isinstance(returns, _pd.DataFrame):
        portfolio.iloc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    # 如果请求则四舍五入
    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """
    使用 CSV 转换方法展平多索引数据框

    参数
    ----------
    df : pd.DataFrame
        要展平的多索引数据框
    set_index : str, optional
        展平后用作索引的列

    返回
    -------
    pd.DataFrame
        展平后的数据框
    """
    # 使用字符串缓冲区转换为 CSV 再转回来以展平结构
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    # 从 CSV 读回以获取展平后的结构
    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df
