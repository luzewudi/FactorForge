#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pandas/numpy 版本兼容层
处理不同版本之间的差异和已弃用功能

本模块提供了统一接口，用于在不同版本的 pandas 和 numpy 之间工作，
确保 quantstats 函数在各种依赖版本中一致运行。它处理已弃用的功能和特定版本的更改。
"""

import pandas as pd
import numpy as np
import warnings
from packaging import version
from typing import Union, Optional, List, Callable

# 本地化改造：
# 已删除 yfinance 依赖和在线下载功能。QuantStats 只负责分析本地传入的
# pandas Series/DataFrame 收益率；如果需要行情，请在项目自己的数据层处理。

# 版本检测 - 解析版本字符串以便进行版本比较
PANDAS_VERSION = version.parse(pd.__version__)
NUMPY_VERSION = version.parse(np.__version__)

# 频率别名映射，用于 pandas 兼容性
# 从 pandas 2.2.0 开始，频率别名变得更明确
# M -> ME（月终），Q -> QE（季终），A/Y -> YE（年终）
FREQUENCY_ALIASES = {
    "M": "ME" if PANDAS_VERSION >= version.parse("2.2.0") else "M",
    "Q": "QE" if PANDAS_VERSION >= version.parse("2.2.0") else "Q",
    "A": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "A",
    "Y": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "Y",
}


def get_frequency_alias(freq: str) -> str:
    """
    获取当前 pandas 版本的正确频率别名。

    此函数将旧的频率字符串映射到 pandas 2.2.0+ 中的新等效字符串，
    确保跨 pandas 版本的向后兼容性。

    参数
    ----------
    freq : str
        频率字符串（例如 'M'、'Q'、'A'、'Y'）

    返回
    -------
    str
        当前 pandas 版本的适当频率别名

    示例
    --------
    >>> get_frequency_alias('M')  # 在 pandas 2.2.0+ 返回 'ME'，在旧版本返回 'M'
    >>> get_frequency_alias('D')  # 返回 'D'（不变）
    """
    # 在映射中查找频率，如果未找到则返回原始值
    return FREQUENCY_ALIASES.get(freq, freq)


def normalize_timezone(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    规范化时区信息以便进行一致比较。

    如果数据有时区信息，则转换为 UTC 然后移除时区信息。
    这确保所有数据无论原始时区如何都可以进行比较。

    参数
    ----------
    data : pd.Series 或 pd.DataFrame
        带有 DatetimeIndex 的时间序列数据

    返回
    -------
    pd.Series 或 pd.DataFrame
        带有无时区 DatetimeIndex 的数据
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data

    # 如果有时区感知，先转换为 UTC 然后移除时区
    if data.index.tz is not None:
        result = data.copy()
        result.index = result.index.tz_convert('UTC').tz_localize(None)
        return result

    # 已经是无时区的，直接返回
    return data


def safe_resample(data: Union[pd.Series, pd.DataFrame],
                  freq: str,
                  func_name: Optional[Union[str, Callable]] = None,
                  **kwargs):
    """
    在所有 pandas 版本中安全执行的重新采样操作。

    此函数使用正确的频率别名和聚合方法处理时间序列数据的重新采样，
    这些方法在不同 pandas 版本间兼容。它还规范化时区以确保一致比较。

    参数
    ----------
    data : pd.Series 或 pd.DataFrame
        要重新采样的时间序列数据
    freq : str
        重新采样的频率（例如 'M'、'Q'、'A'、'D'）
    func_name : str 或可调用对象，可选
        要应用的聚合函数。可以是字符串名称如 'sum'、'mean'、'std' 等，
        也可以是可直接调用的函数
    **kwargs
        传递给聚合函数的其他参数

    返回
    -------
    pd.Series 或 pd.DataFrame
        具有指定频率和聚合的重新采样数据，
        如果存在则将时区规范化为 UTC，如果不存在则规范化

    示例
    --------
    >>> safe_resample(data, 'M', 'sum')  # 月度求和聚合
    >>> safe_resample(data, 'Q', 'mean')  # 季度均值聚合
    """
    # 将频率转换为当前 pandas 版本的适当别名
    freq_alias = get_frequency_alias(freq)

    # 使用正确的频率创建重新采样对象
    resampler = data.resample(freq_alias)

    # 如果没有指定聚合函数，返回重新采样对象
    if func_name is None:
        return resampler

    # 使用显式方法调用处理字符串函数名
    # 这种方法可避免弃用警告并确保兼容性
    result = None
    if isinstance(func_name, str):
        # 将常见聚合函数映射到其 pandas 方法
        if func_name == "sum":
            result = resampler.sum(**kwargs)
        elif func_name == "mean":
            result = resampler.mean(**kwargs)
        elif func_name == "std":
            result = resampler.std(**kwargs)
        elif func_name == "count":
            result = resampler.count(**kwargs)
        elif func_name == "min":
            result = resampler.min(**kwargs)
        elif func_name == "max":
            result = resampler.max(**kwargs)
        elif func_name == "first":
            result = resampler.first(**kwargs)
        elif func_name == "last":
            result = resampler.last(**kwargs)
        else:
            # 尝试在重新采样器对象上查找方法
            if hasattr(resampler, func_name):
                result = getattr(resampler, func_name)(**kwargs)
            else:
                # 回退到 apply 处理自定义字符串函数
                result = resampler.apply(func_name, **kwargs)
    else:
        # 对于可调用函数，使用 apply 方法
        # 抑制关于可调用用法的 FutureWarning - 我们的用法是有意的
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message=".*callable.*")
            result = resampler.apply(func_name, **kwargs)

    # 规范化时区以确保一致比较
    return normalize_timezone(result)


def safe_concat(objs: List[Union[pd.Series, pd.DataFrame]],
                axis: int = 0,
                ignore_index: bool = False,
                sort: bool = False,
                **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """
    处理 pandas 版本差异的安全连接操作。

    此函数提供 pd.concat 的包装器，使用一致的参数。

    参数
    ----------
    objs : pd.Series 或 pd.DataFrame 列表
        沿指定轴连接的对象
    axis : int，默认为 0
        连接的轴。0 表示行，1 表示列
    ignore_index : bool，默认为 False
        是否忽略索引并创建新的默认整数索引
    sort : bool，默认为 False
        是否对结果排序
    **kwargs
        传递给 pd.concat 的其他参数

    返回
    -------
    pd.Series 或 pd.DataFrame
        连接后的结果

    示例
    --------
    >>> safe_concat([df1, df2])  # 沿行连接
    >>> safe_concat([df1, df2], axis=1)  # 沿列连接
    """
    # 使用 sort 参数执行连接（pandas 2.0+ 可用）
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, sort=sort, **kwargs)  # type: ignore[arg-type]


def safe_append(df: pd.DataFrame,
                other: Union[pd.DataFrame, pd.Series],
                ignore_index: bool = False,
                sort: bool = False) -> pd.DataFrame:
    """
    使用 pd.concat 的安全追加操作。

    DataFrame.append() 在 pandas 2.0.0 中被移除。此函数提供使用 pd.concat 的统一接口。

    参数
    ----------
    df : pd.DataFrame
        要追加到的 DataFrame（基础 DataFrame）
    other : pd.DataFrame 或 pd.Series
        要追加到基础 DataFrame 的数据
    ignore_index : bool，默认为 False
        是否忽略索引并创建新的默认整数索引
    sort : bool，默认为 False
        是否按列排序结果

    返回
    -------
    pd.DataFrame
        追加操作的结果

    示例
    --------
    >>> safe_append(df, new_row)  # 追加新行
    >>> safe_append(df, other_df, ignore_index=True)  # 追加并重置索引
    """
    # 使用 concat（append 在 pandas 2.0 中被移除）
    result = safe_concat([df, other], ignore_index=ignore_index, sort=sort)
    # 确保返回 DataFrame
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, pd.Series):
        return pd.DataFrame([result])
    else:
        return pd.DataFrame(result)


def safe_frequency_conversion(data: Union[pd.Series, pd.DataFrame],
                              freq: str) -> Union[pd.Series, pd.DataFrame]:
    """
    时间序列数据的安全频率转换。

    此函数使用当前 pandas 版本中最合适的方法将时间序列数据转换为指定频率。

    参数
    ----------
    data : pd.Series 或 pd.DataFrame
        带有日期时间索引的时间序列数据
    freq : str
        目标频率（例如 'D'、'M'、'Q'、'A'）

    返回
    -------
    pd.Series 或 pd.DataFrame
        转换频率后的数据

    示例
    --------
    >>> safe_frequency_conversion(data, 'M')  # 转换为月度频率
    >>> safe_frequency_conversion(data, 'D')  # 转换为日度频率
    """
    # 获取当前 pandas 版本的适当频率别名
    freq_alias = get_frequency_alias(freq)

    # 处理不同的频率转换方法
    if hasattr(data, "asfreq"):
        # 如果可用，使用 asfreq（最直接的方法）
        return data.asfreq(freq_alias)
    else:
        # 回退到使用 'last' 聚合的重新采样
        # 这保留每个周期的最后一个值
        return safe_resample(data, freq_alias, "last")


def handle_pandas_warnings():
    """
    上下文管理器，用于适当处理 pandas 警告。

    此函数返回一个上下文管理器，可用于以受控方式抑制或处理 pandas 警告。
    在处理多个 pandas 版本时用于管理弃用警告非常有用。

    返回
    -------
    warnings.catch_warnings
        用于处理警告的上下文管理器

    示例
    --------
    >>> with handle_pandas_warnings():
    ...     # 可能产生 pandas 警告的代码
    ...     pass
    """
    # 返回用于灵活警告处理的上下文管理器
    return warnings.catch_warnings()


# Pandas 访问器兼容函数
def get_datetime_accessor(series: pd.Series):
    """
    获取 pandas Series 的日期时间访问器。

    此函数提供跨不同版本访问 pandas Series 日期时间属性的统一接口。

    参数
    ----------
    series : pd.Series
        要获取访问器的日期时间数据序列

    返回
    -------
    pd.Series.dt
        序列的日期时间访问器

    示例
    --------
    >>> dt_accessor = get_datetime_accessor(date_series)
    >>> dt_accessor.year  # 访问年份分量
    >>> dt_accessor.month  # 访问月份分量
    """
    # 返回日期时间访问器 - 在 pandas 版本间一致
    return series.dt


def get_string_accessor(series: pd.Series):
    """
    获取 pandas Series 的字符串访问器。

    此函数提供跨不同版本访问 pandas Series 字符串方法的统一接口。

    参数
    ----------
    series : pd.Series
        要获取访问器的字符串数据序列

    返回
    -------
    pd.Series.str
        序列的字符串访问器

    示例
    --------
    >>> str_accessor = get_string_accessor(string_series)
    >>> str_accessor.lower()  # 转换为小写
    >>> str_accessor.contains('pattern')  # 检查模式
    """
    # 返回字符串访问器 - 在 pandas 版本间一致
    return series.str


def safe_yfinance_download(tickers: Union[str, List[str]],
                           proxy: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
    """
    本地化版本已删除 yfinance 下载功能。

    原版函数用于联网下载 Yahoo Finance 行情；当前项目只支持用户传入
    本地收益率或净值序列，因此保留这个函数仅用于给出清晰错误提示。
    """
    message = (
        f"QuantStats 本地化版本不支持 safe_yfinance_download({tickers!r})。"
        "请在外部准备好 pandas Series/DataFrame 收益率后再传入报告函数。"
    )
    try:
        from utils.log_kit import logger

        logger.debug(message)
    except Exception:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    raise RuntimeError(message)
