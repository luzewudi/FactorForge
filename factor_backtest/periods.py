# -*- coding: utf-8 -*-
"""周期预运算工具。

本模块负责把 EOD 交易日历转换为统一的换仓日布尔矩阵：
- ``period.npy`` 保存周期名称，例如 ``5_0``、``20_19``、``W_4``。
- ``period_dates.npy`` 保存 ``period x dates`` 的 True/False 矩阵。
- ``dates.npy`` 保存和 EOD 对齐后的日期标签。

实现口径参考 ``周期预运算（沈博文增强版）.py``，但不直接修改原脚本。
核心思想是先为每个 ``周期_offset`` 生成“持仓周期编号”，再把编号变化的日期标记为换仓日。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import normalize_offset_label, normalize_period
from .data_loader import decode_array, normalize_date_label


def default_period_dict() -> dict[Any, list[Any]]:
    """返回默认预计算的周期与 offset 集合。"""
    return {
        # 整数周期表示按交易日数量切分；N 日周期天然有 offset0 到 offsetN-1。
        1: [0],
        2: list(range(2)),
        3: list(range(3)),
        4: list(range(4)),
        5: list(range(5)),
        10: list(range(10)),
        20: list(range(20)),
        21: list(range(21)),
        # 周频 offset0-4 对应自然周内不同换仓偏移，和原周期脚本保持一致。
        "W": list(range(5)),
        "2W": [0, 1],
        "3W": [0, 1, 2],
        "4W": [0, 1, 2, 3],
        "5W": [0, 1, 2, 3, 4],
        "6W": [0, 1, 2, 3, 4, 5],
        # M_-5 和 W53 是原脚本里的特殊约定，下面分别有单独处理逻辑。
        "M": [0, -5],
        "W53": [0],
    }


def load_period_files(period_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """读取预计算周期文件，并校验 ``period_dates`` 的二维形状。"""
    period_path = Path(period_path)
    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    missing = [path for path in [names_path, mask_path, dates_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"period files not found: {missing_text}")

    # period.npy 允许 object/bytes/string 混合保存，这里统一成普通字符串列表。
    period_names = decode_array(np.load(names_path, allow_pickle=True))
    period_dates = np.load(mask_path, allow_pickle=False)
    dates = [normalize_date_label(x) for x in decode_array(np.load(dates_path, allow_pickle=True))]
    if period_dates.shape != (len(period_names), len(dates)):
        raise ValueError(
            f"period_dates shape {period_dates.shape} does not match "
            f"period count {len(period_names)} and date count {len(dates)}"
        )
    return period_names, period_dates.astype(bool, copy=False), dates


def resolve_period_keys(period_names: list[str], period: Any, offsets: str | list[str]) -> list[str]:
    """把 YAML 中的 period/offsets 配置解析成 ``20_7`` 这样的具体列名。"""
    base = normalize_period(period)
    prefix = f"{base}_"
    available = [name for name in period_names if name.startswith(prefix)]
    if offsets == "all":
        keys = available
    else:
        keys = [format_period_key(base, offset) for offset in offsets]

    missing = [key for key in keys if key not in period_names]
    if missing:
        sample = ", ".join(sort_period_keys(available)[:12])
        raise ValueError(f"period keys not found: {missing}. Available for {base}: {sample}")
    return sort_period_keys(keys)


def format_period_key(period: Any, offset: Any) -> str:
    """按 ``period.npy`` 的命名规则拼出周期键。"""
    return f"{normalize_period(period)}_{normalize_offset_label(offset)}"


def sort_period_keys(keys: list[str]) -> list[str]:
    """按 offset 后缀排序；数字 offset 按数值排，字符串 offset 保持字典序。"""

    def _key(value: str) -> tuple[int, int | str]:
        suffix = value.split("_", 1)[1] if "_" in value else value
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(keys, key=_key)


def build_period_id_frame(dates: list[str], period_dict: dict[Any, list[Any]] | None = None, min_day: int = 2) -> pd.DataFrame:
    """根据 EOD 交易日生成各周期 offset 的“持仓周期编号”宽表。"""
    period_dict = period_dict or default_period_dict()
    # base 只使用真实交易日，不使用离线节假日推断，确保和 EOD 面板完全对齐。
    base = pd.DataFrame({"trade_date": pd.to_datetime(dates, format="%Y%m%d")})
    base["is_trade"] = 1
    base["period_end"] = base["trade_date"]
    out = base[["trade_date"]].copy()

    for period_type, offsets in period_dict.items():
        if period_type == "W53":
            out = out.merge(_w53_period_ids(base), on="trade_date", how="left")
            continue
        for offset in offsets:
            col = format_period_key(period_type, offset)
            ids = _single_period_ids(base, period_type, offset, min_day=min_day)
            out = out.merge(ids[["trade_date", col]], on="trade_date", how="left")

    # 原脚本头部未进入第一个完整周期的日期会填 0，这里沿用同一口径。
    out.fillna(0, inplace=True)
    return out


def build_period_rebalance_matrix(dates: list[str], period_dict: dict[Any, list[Any]] | None = None) -> tuple[list[str], np.ndarray]:
    """把周期编号宽表转换成换仓日布尔矩阵。"""
    ids = build_period_id_frame(dates, period_dict=period_dict)
    period_names = [col for col in ids.columns if col != "trade_date"]
    values = ids[period_names].to_numpy(dtype=float)
    # 周期编号从上一交易日变到当前交易日，说明当前交易日是该 offset 的调仓日。
    previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    rebalance = (values != previous) & (values != 0)
    if rebalance.shape[0] > 0:
        # 首日没有上一交易日因子，强制不换仓，避免未来函数。
        rebalance[0, :] = False
    return period_names, rebalance.T.astype(bool, copy=False)


def save_period_files(eod_path: Path, period_path: Path) -> tuple[Path, Path, Path]:
    """从 ``eod/dates.npy`` 生成三个周期预运算文件。"""
    eod_path = Path(eod_path)
    period_path = Path(period_path)
    period_path.mkdir(parents=True, exist_ok=True)
    dates = [normalize_date_label(x) for x in decode_array(np.load(eod_path / "dates.npy", allow_pickle=True))]
    period_names, rebalance = build_period_rebalance_matrix(dates)

    # 文件放在仓库外的数据目录，代码只负责生成和读取，不把大数据纳入 git。
    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    np.save(names_path, np.asarray(period_names, dtype=object))
    np.save(mask_path, rebalance)
    np.save(dates_path, np.asarray(dates, dtype="S8"))
    return names_path, mask_path, dates_path


def _single_period_ids(base: pd.DataFrame, period_type: Any, offset: Any, min_day: int) -> pd.DataFrame:
    """生成单个 ``周期_offset`` 的持仓周期编号。"""
    data = base.copy()
    col = format_period_key(period_type, offset)
    agg_dict = {"period_end": "last", "is_trade": "sum"}

    if isinstance(period_type, int):
        # 整数周期直接按交易日序号切组；offset 决定从第几个交易日开始错位。
        index = np.arange(len(data), dtype=float)
        data["group"] = np.trunc((index - int(offset)) / period_type).astype(int)
        period_df = data.groupby("group").agg(agg_dict)
        period_df.rename(columns={"is_trade": "trading_days"}, inplace=True)
        period_df["trading_days"] = max(int(period_type), min_day)
    else:
        period_text = str(period_type).upper()
        if period_text == "M" and str(offset) == "-5":
            # M_-5 是原脚本特殊月频口径：用自然日补齐后处理月末附近非交易日。
            period_df = _month_minus_five_periods(data)
        else:
            # 周频/月频通过移动日期后 resample，使 pandas 的周期末落到目标换仓日。
            data = _shift_trade_dates(data, period_text, offset)
            data.loc[data.index.max() + 1, "trade_date"] = pd.to_datetime("1990-01-01")
            data.set_index("trade_date", inplace=True)
            period_df = data.resample(rule=_pandas_resample_rule(period_text)).agg(agg_dict)
            period_df.rename(columns={"is_trade": "trading_days"}, inplace=True)

    period_df = period_df[period_df["trading_days"] > 0]
    period_df = _merge_short_periods(period_df, min_day=min_day)
    period_df.reset_index(drop=True, inplace=True)
    period_df.rename(columns={"period_end": "trade_date"}, inplace=True)
    # 某些特殊月末可能映射到同一个交易日，保留最后一条，保证输出和 EOD 日期一一对应。
    period_df = period_df.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    period_df[col] = 1

    out = base[["trade_date"]].merge(period_df[["trade_date", col]], on="trade_date", how="left")
    # expanding().sum().shift() 是原脚本的关键口径：
    # 当天的编号代表“今天开盘后持有的周期”，编号变化发生在下一交易日可见。
    out[col] = out[col].expanding().sum().shift()
    return out


def _shift_trade_dates(data: pd.DataFrame, period_text: str, offset: Any) -> pd.DataFrame:
    """按原脚本规则移动日期，让 resample 的周期边界落到指定 offset。"""
    if any(char.isdigit() for char in period_text):
        if isinstance(offset, str) and "D" in offset.upper():
            # 兼容 2W 的 0D/1D 等自然日 offset 写法。
            data["trade_date"] -= pd.to_timedelta(offset.upper())
        else:
            # nW 的整数 offset 按“周”错位，和原脚本保持一致。
            data["trade_date"] -= pd.to_timedelta(f"{int(offset) * 7}D")
    else:
        # W 的 offset0-4 直接按自然日移动，约定对应周内不同交易日。
        data["trade_date"] -= pd.to_timedelta(f"{int(offset)}D")
    return data


def _pandas_resample_rule(period_text: str) -> str:
    """把公开周期名转换成当前 pandas 推荐的 resample 频率名。"""
    return "ME" if period_text == "M" else period_text


def _month_minus_five_periods(data: pd.DataFrame) -> pd.DataFrame:
    """处理原脚本中的 ``M_-5`` 月频特殊口径。"""
    start_date = data["trade_date"].min()
    end_date = data["trade_date"].max()
    # 先补齐自然日，再用 period_end 前向填充，才能判断自然月末和非交易日关系。
    natural = data.set_index("trade_date").reindex(pd.date_range(start=start_date, end=end_date, freq="D")).reset_index()
    natural.rename(columns={"index": "trade_date"}, inplace=True)
    natural["period_end"] = natural["period_end"].ffill()
    natural["is_trade"] = natural["is_trade"].fillna(value=0)

    month_end = pd.DataFrame({"trade_date": pd.date_range(start=start_date, end=end_date, freq="ME")})
    month_end["month_end"] = 1
    natural = natural.merge(month_end, on="trade_date", how="left")
    # 如果自然月末是交易日且次日非交易，则把该自然月末当作周期边界处理。
    natural.loc[(natural["month_end"] == 1) & (natural["is_trade"].shift(-1) == 0), "is_trade"] = 0
    natural = natural[natural["is_trade"] == 0].set_index("trade_date")
    period_df = natural.resample(rule="ME").agg({"period_end": "last", "is_trade": "sum"})
    period_df["trading_days"] = 20
    return period_df


def _merge_short_periods(period_df: pd.DataFrame, min_day: int) -> pd.DataFrame:
    """把交易天数过短的周期并入下一周期，避免 T+1 下无法完成买卖。"""
    to_remove = []
    add_num = 0
    for index, row in period_df.iterrows():
        period_df.at[index, "trading_days"] += add_num
        add_num = 0
        if row["trading_days"] < min_day:
            to_remove.append(index)
            add_num = row["trading_days"]
    return period_df.drop(to_remove)


def _w53_period_ids(base: pd.DataFrame) -> pd.DataFrame:
    """生成 ``W53_0`` 特殊周期编号。

    ``W53`` 来自原脚本里的“周五买、周三卖”口径：周三作为实际卖出边界，
    周四属于名义周期但收益在下游会按负编号特殊处理。
    """
    col = "W53_0"
    data = base.copy()
    start_date = data["trade_date"].min()
    end_date = data["trade_date"].max()

    shifted = data.copy()
    # 第一步：先用周三作为周期末，找出实际卖出日。
    shifted["trade_date"] -= pd.to_timedelta("3D")
    shifted.loc[shifted.index.max() + 1, "trade_date"] = pd.to_datetime("1990-01-01")
    shifted.set_index("trade_date", inplace=True)
    period_df = shifted.resample(rule="W").agg({"period_end": "last", "is_trade": "sum"})
    period_df.rename(columns={"is_trade": "trading_days"}, inplace=True)
    period_df = period_df[period_df["trading_days"] > 0].reset_index(drop=True)
    period_df.rename(columns={"period_end": "trade_date"}, inplace=True)
    period_df["W53"] = 1

    out = base[["trade_date"]].merge(period_df[["trade_date", "W53"]], on="trade_date", how="left")
    out[col] = out["W53"].expanding().sum().shift()

    # 第二步：找出自然周四，并把这些日期标记为空，后面转成负周期编号。
    natural = pd.DataFrame({"trade_date": pd.date_range(start=start_date, end=end_date, freq="D")})
    natural["period_end"] = natural["trade_date"]
    natural["trade_date"] -= pd.to_timedelta("4D")
    natural.loc[natural.index.max() + 1, "trade_date"] = pd.to_datetime("1990-01-01")
    natural.set_index("trade_date", inplace=True)
    thursday_df = natural.resample(rule="W").agg({"period_end": "last"})
    out.loc[out["trade_date"].isin(thursday_df["period_end"]), col] = None

    # 第三步：单交易日周期无法满足 T+1，剔除后交给下一周期承接。
    counts = out[col].value_counts()
    single_values = counts[counts == 1].index
    out.loc[out[col].isin(single_values), col] = None
    marker = out[col].copy()
    # 第四步：空值日期继承上一个周期编号并取负，保留“名义持有但实际不计收益”的标记。
    out[col] = out[col].ffill()
    out.loc[pd.isnull(marker), col] = -out[col]
    return out[["trade_date", col]]
