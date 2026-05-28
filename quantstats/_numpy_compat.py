#!/usr/bin/env python
"""
Numpy 兼容层
处理不同 numpy 版本之间的差异和已弃用功能

本模块提供了统一接口，用于在不同版本的 numpy 之间工作，
确保 quantstats 函数在各种 numpy 版本中一致运行。
它处理已弃用的功能和特定版本的更改，并提供回退机制，
以便在旧版本中可能不存在的函数也能正常工作。
"""

import numpy as np
import warnings
from packaging import version
from typing import Union, Optional, Any

# 版本检测 - 解析 numpy 版本字符串以便进行版本比较
NUMPY_VERSION = version.parse(np.__version__)

# 处理已弃用的 numpy 函数
# 在 numpy 1.25.0+ 中，np.product 已弃用，改用 np.prod
if NUMPY_VERSION >= version.parse("1.25.0"):
    # 在新版本中使用 np.prod 替代已弃用的 np.product
    product = np.prod
else:
    # 在旧版本中使用 np.product，如果不可用则回退到 np.prod
    product = getattr(np, "product", np.prod)


def safe_numpy_operation(data, operation: str):
    """
    带弃用处理的 numpy 安全操作。

    本函数为可能在不同 numpy 版本中已被弃用或行为发生变化的 numpy 操作提供统一接口。
    它处理从已弃用函数到其替代品的过渡。

    参数
    ----------
    data : array_like
        要执行操作的输入数据。可以是 numpy 能处理的任何类数组结构
        （列表、元组、numpy 数组等）
    operation : str
        要执行的 numpy 操作。支持的操包括 'product'、
        'prod' 以及当前版本中存在的任何其他 numpy 函数名

    返回
    -------
    ndarray
        对输入数据应用 numpy 操作后的结果

    示例
    --------
    >>> safe_numpy_operation([1, 2, 3, 4], 'product')  # 返回 24
    >>> safe_numpy_operation([1, 2, 3, 4], 'sum')      # 返回 10
    """
    # 特别处理已弃用的 'product' 操作
    if operation == "product":
        # 使用我们版本感知的 product 函数
        return product(data)
    elif operation == "prod":
        # 对 'prod' 操作直接使用 np.prod
        return np.prod(data)
    else:
        # 对于所有其他操作，动态从 numpy 获取函数
        # 这允许在不同 numpy 版本之间灵活支持操作
        return getattr(np, operation)(data)


def safe_array_function(func_name: str, *args, **kwargs) -> Any:
    """
    numpy 数组函数的安全包装器。

    本函数提供了一种安全调用可能在所有 numpy 版本中都不存在或已被弃用的 numpy 函数的方法。
    它处理版本特定的函数可用性，并提供有意义的错误消息。

    参数
    ----------
    func_name : str
        要调用的 numpy 函数名（例如 'mean'、'std'、'product'）
    *args
        传递给 numpy 函数的位置参数
    **kwargs
        传递给 numpy 函数的关键字参数

    返回
    -------
    Any
        numpy 函数调用的结果。返回类型取决于被调用的特定函数
        和输入数据。

    引发
    ------
    AttributeError
        如果请求的函数在当前 numpy 版本中不存在

    示例
    --------
    >>> safe_array_function('mean', [1, 2, 3, 4])      # 返回 2.5
    >>> safe_array_function('product', [1, 2, 3, 4])   # 返回 24
    """
    # 使用特殊情况处理已弃用的函数
    if func_name == "product":
        # 使用我们版本感知的 product 函数
        return product(*args, **kwargs)

    # 处理可能在旧版 numpy 中不存在的函数
    if hasattr(np, func_name):
        # 函数在当前 numpy 版本中存在，正常调用
        return getattr(np, func_name)(*args, **kwargs)
    else:
        # 函数在当前 numpy 版本中不存在，抛出信息性错误
        raise AttributeError(
            f"numpy 在版本 {NUMPY_VERSION} 中没有属性 '{func_name}'"
        )


def handle_numpy_warnings():
    """
    上下文管理器，用于适当处理 numpy 警告。

    本函数返回一个上下文管理器，可用于以受控方式抑制
    或处理 numpy 警告。在使用多个 numpy 版本时，
    这对管理弃用警告和其他 numpy 特定警告非常有用。

    返回
    -------
    warnings.catch_warnings
        用于处理 numpy 警告的上下文管理器

    示例
    --------
    >>> with handle_numpy_warnings():
    ...     # 可能产生 numpy 警告的代码
    ...     result = np.some_deprecated_function()
    """
    # 返回警告上下文管理器以便灵活处理警告
    return warnings.catch_warnings()


def safe_percentile(data, percentile: Union[float, list], **kwargs):
    """
    安全的百分位数计算。

    本函数提供围绕 np.percentile 的统一参数包装器。

    参数
    ----------
    data : array_like
        要计算百分位数的输入数据。可以是 numpy 能处理的任何类数组
        结构
    percentile : float or array_like
        要计算的百分位数。值应在 0 到 100 之间。
        可以是单个值或值数组
    **kwargs
        传递给 np.percentile 的其他参数

    返回
    -------
    float or ndarray
        计算后的百分位数。单百分位数返回 float，
        多百分位数返回 ndarray

    示例
    --------
    >>> safe_percentile([1, 2, 3, 4, 5], 50)    # 返回 3.0（中位数）
    >>> safe_percentile([1, 2, 3, 4, 5], [25, 75])  # 返回 [2.0, 4.0]
    """
    # Numpy 1.24+ 支持百分位数计算的 'method' 参数
    return np.percentile(data, percentile, **kwargs)


def safe_nanpercentile(data, percentile: Union[float, list], **kwargs):
    """
    忽略 NaN 值的安全 nanpercentile 计算。

    本函数提供围绕 np.nanpercentile 的统一参数包装器。

    参数
    ----------
    data : array_like
        要计算百分位数的输入数据。NaN 值会被忽略。
        可以是 numpy 能处理的任何类数组结构
    percentile : float or array_like
        要计算的百分位数。值应在 0 到 100 之间。
        可以是单个值或值数组
    **kwargs
        传递给 np.nanpercentile 的其他参数

    返回
    -------
    float or ndarray
        忽略 NaN 值计算后的百分位数。单百分位数返回 float，
        多百分位数返回 ndarray

    示例
    --------
    >>> safe_nanpercentile([1, 2, np.nan, 4, 5], 50)    # 返回 3.0
    >>> safe_nanpercentile([1, np.nan, 3, 4, 5], [25, 75])  # 返回 [2.0, 4.5]
    """
    # Numpy 1.24+ 支持 nanpercentile 计算的 'method' 参数
    return np.nanpercentile(data, percentile, **kwargs)


def safe_quantile(data, quantile: Union[float, list], **kwargs):
    """
    安全的分位数计算。

    本函数提供围绕 np.quantile 的统一参数包装器。
    分位数类似于百分位数，但使用 0 到 1 之间的值。

    参数
    ----------
    data : array_like
        要计算分位数的输入数据。可以是 numpy 能处理的任何类数组
        结构
    quantile : float or array_like
        要计算的分位数。值应在 0 到 1 之间。
        可以是单个值或值数组
    **kwargs
        传递给 np.quantile 的其他参数

    返回
    -------
    float or ndarray
        计算后的分位数。单分位数返回 float，
        多分位数返回 ndarray

    示例
    --------
    >>> safe_quantile([1, 2, 3, 4, 5], 0.5)    # 返回 3.0（中位数）
    >>> safe_quantile([1, 2, 3, 4, 5], [0.25, 0.75])  # 返回 [2.0, 4.0]
    """
    # Numpy 1.24+ 支持分位数计算的 'method' 参数
    return np.quantile(data, quantile, **kwargs)


def safe_random_seed(seed: Optional[int]):
    """
    numpy 的安全随机种子设置。

    本函数提供设置随机种子的统一接口。

    参数
    ----------
    seed : int or None
        要设置的随机种子值。如果为 None，则不修改随机状态。
        设置相同的种子确保可重现的随机数序列

    示例
    --------
    >>> safe_random_seed(42)  # 设置种子以获得可重现结果
    >>> safe_random_seed(None)  # 不设置种子，随机行为继续
    """
    if seed is not None:
        # 使用现代随机数生成器（numpy 1.17.0+）
        np.random.default_rng(seed)


def safe_datetime64_unit(dt, unit: str):
    """
    安全的 datetime64 单位转换。

    本函数提供将 numpy datetime64 对象安全转换为不同时间单位的方法。

    参数
    ----------
    dt : np.datetime64
        要转换的输入 datetime 对象
    unit : str
        目标时间单位（例如 'D' 表示天，'H' 表示小时，'M' 表示分钟，
        'S' 表示秒，'ms' 表示毫秒）

    返回
    -------
    np.datetime64
        具有指定单位的转换后 datetime 对象

    示例
    --------
    >>> dt = np.datetime64('2023-01-01T12:00:00')
    >>> safe_datetime64_unit(dt, 'D')  # 转换为天精度
    >>> safe_datetime64_unit(dt, 'H')  # 转换为小时精度
    """
    return dt.astype(f"datetime64[{unit}]")
