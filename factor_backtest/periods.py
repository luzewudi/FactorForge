# -*- coding: utf-8 -*-
"""周期预运算工具。

本模块已经把 ``周期预运算（沈博文增强版）.py`` 中 FactorForge 需要的周期
生成逻辑搬进来，后续不再依赖那个独立脚本。唯一的输入变化是：
原脚本从指数 CSV 获取交易日，这里从 ``eod/dates.npy`` 获取交易日。

输出文件：
- ``period.npy``：周期名称，例如 ``5_0``、``20_19``、``W_4``。
- ``period_dates.npy``：``period x dates`` 的 True/False 换仓日矩阵。
- ``dates.npy``：与 EOD 对齐的日期标签。
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
        1: [0],
        2: list(range(2)),
        3: list(range(3)),
        4: list(range(4)),
        5: list(range(5)),
        10: list(range(10)),
        20: list(range(20)),
        21: list(range(21)),
        "W": [0, 1, 2, 3, 4],
        "2W": [0, 1, "0D", "1D", "2D", "3D", "4D", "7D", "8D", "9D", "10D", "11D"],
        "3W": [0, 1, 2],
        "4W": [0, 1, 2, 3],
        "5W": [0, 1, 2, 3, 4],
        "6W": [0, 1, 2, 3, 4, 5],
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
        sample = ", ".join(available[:12])
        raise ValueError(f"period keys not found: {missing}. Available for {base}: {sample}")
    return keys


def format_period_key(period: Any, offset: Any) -> str:
    """按 ``period.npy`` 的命名规则拼出周期键。"""
    return f"{normalize_period(period)}_{normalize_offset_label(offset)}"


def sort_period_keys(keys: list[str]) -> list[str]:
    """保留兼容接口；周期文件本身已经按原脚本生成顺序排列。"""
    return list(keys)


def build_period_id_frame(dates: list[str], period_dict: dict[Any, list[Any]] | None = None, min_day: int = 2) -> pd.DataFrame:
    """根据 EOD 交易日生成各周期 offset 的“持仓周期编号”宽表。"""
    period_dict = period_dict or default_period_dict()
    index_data = pd.DataFrame({"交易日期": pd.to_datetime(dates, format="%Y%m%d")})
    index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)
    raw = calc_period_and_offset(period_dict, index_data, min_day=min_day)
    return _align_period_frame(raw, index_data["交易日期"])


def calc_period_and_offset(period_dict: dict[Any, list[Any]], _index_data: pd.DataFrame, min_day: int = 2) -> pd.DataFrame:
    """按原脚本 ``calc_period_and_offset`` 的周期逻辑生成周期编号宽表。

    与原脚本相比这里只删除了 ``TradeCalendar`` 补未来交易日的部分，因为本项目要求
    严格使用 ``eod/dates.npy`` 作为交易日历。除此之外，W、M、M_-5、W53、短周期合并、
    ``expanding().sum().shift()`` 等处理均保留原脚本口径。
    """
    _index_data = _index_data.copy()
    _index_data["交易日期"] = pd.to_datetime(_index_data["交易日期"])
    _index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)
    _index_data["是否交易"] = 1
    _index_data["周期最后交易日"] = _index_data["交易日期"]
    all_period_offst_df = _index_data[["交易日期"]].copy()

    agg_dict = {"周期最后交易日": "last", "是否交易": "sum"}

    for period_type in period_dict:
        if period_type in ["W53"]:
            index_data = _index_data.copy()
            start_date = index_data["交易日期"].min()
            end_date = index_data["交易日期"].max()

            # 第一步：先把 W_3 算一遍，目的和原脚本一致，是把周三周期末定义出来。
            index_data["交易日期"] -= pd.to_timedelta("3D")
            index_data.loc[index_data.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            index_data.set_index("交易日期", inplace=True)
            period_df = index_data.resample(rule="W").agg(agg_dict)
            period_df.rename(columns={"是否交易": "交易天数"}, inplace=True)
            period_df = period_df[period_df["交易天数"] > 0]
            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={"周期最后交易日": "交易日期"}, inplace=True)
            period_df["W53"] = 1

            df = pd.merge(_index_data[["交易日期"]].copy(), right=period_df[["交易日期", "W53"]], on="交易日期", how="left")
            df["W53_0"] = df["W53"].expanding().sum().shift()

            # 第二步：把所有周四标记为 None。
            date_range_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="D"), columns=["交易日期"])
            date_range_df["周期最后交易日"] = date_range_df["交易日期"].copy()
            date_range_df["交易日期"] -= pd.to_timedelta("4D")
            date_range_df.loc[date_range_df.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            date_range_df.set_index("交易日期", inplace=True)
            period_df = date_range_df.resample(rule="W").agg({"周期最后交易日": "last"})
            df.loc[df["交易日期"].isin(period_df["周期最后交易日"]), "W53_0"] = None

            # 第三步：单交易日周期不满足 T+1，设置为 None。
            counts = df["W53_0"].value_counts()
            single_occurrence_values = counts[counts == 1].index
            df.loc[df["W53_0"].isin(single_occurrence_values), "W53_0"] = None

            # 第四步：None 做 ffill 后取负，保留原脚本的名义周期标记。
            df["_W53"] = df["W53_0"].copy()
            df["W53_0"] = df["W53_0"].ffill()
            df.loc[pd.isnull(df["_W53"]), "W53_0"] = -df["W53_0"]
            all_period_offst_df = pd.merge(
                left=all_period_offst_df,
                right=df[["交易日期", "W53_0"]],
                on="交易日期",
                how="left",
            )
            continue

        for offset in period_dict[period_type]:
            index_data = _index_data.copy()
            if type(period_type) == int:
                index_data["group"] = pd.Series((index_data.index - offset) / period_type).apply(int)
                period_df = index_data.groupby("group").agg(agg_dict)
                period_df["交易天数"] = period_type
            else:
                if (period_type == "M") and (offset == -5):
                    start_date = index_data["交易日期"].min()
                    end_date = index_data["交易日期"].max()
                    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                    index_data = index_data.set_index("交易日期").reindex(date_range).reset_index()
                    index_data.rename(columns={"index": "交易日期"}, inplace=True)
                    index_data["周期最后交易日"] = index_data["周期最后交易日"].ffill()
                    index_data["是否交易"] = index_data["是否交易"].fillna(value=0)

                    date_range_m = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="M"), columns=["交易日期"])
                    date_range_m["月末"] = 1
                    index_data = pd.merge(left=index_data, right=date_range_m, on="交易日期", how="left")
                    index_data.loc[(index_data["月末"] == 1) & (index_data["是否交易"].shift(-1) == 0), "是否交易"] = 0
                    index_data = index_data[index_data["是否交易"] == 0]
                    index_data.set_index("交易日期", inplace=True)
                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df["交易天数"] = 20
                else:
                    if (lambda s: any(char.isdigit() for char in s))(period_type):
                        if isinstance(offset, str) and "D" in offset.upper():
                            offset = offset.upper()
                            index_data["交易日期"] -= pd.to_timedelta(offset)
                        else:
                            index_data["交易日期"] -= pd.to_timedelta(f"{offset * 7}D")
                    else:
                        index_data["交易日期"] -= pd.to_timedelta(f"{offset}D")

                    index_data.loc[index_data.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
                    index_data.set_index("交易日期", inplace=True)
                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df.rename(columns={"是否交易": "交易天数"}, inplace=True)

            period_df = period_df[period_df["交易天数"] > 0]

            index_to_remove = []
            add_num = 0
            for index, row in period_df.iterrows():
                period_df.at[index, "交易天数"] += add_num
                add_num = 0
                if row["交易天数"] < min_day:
                    index_to_remove.append(index)
                    add_num = row["交易天数"]
            period_df = period_df.drop(index_to_remove)

            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={"周期最后交易日": "交易日期"}, inplace=True)
            period_df[f"{period_type}_{offset}"] = 1
            all_period_offst_df = pd.merge(
                left=all_period_offst_df,
                right=period_df[["交易日期", f"{period_type}_{offset}"]],
                on="交易日期",
                how="left",
            )
            all_period_offst_df[f"{period_type}_{offset}"] = all_period_offst_df[f"{period_type}_{offset}"].expanding().sum().shift()

    all_period_offst_df.fillna(value=0, inplace=True)
    return all_period_offst_df


def _align_period_frame(raw: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """把周期编号宽表整理成与 ``eod/dates.npy`` 一日一行的输出。"""
    out = raw.copy()
    out["交易日期"] = pd.to_datetime(out["交易日期"])
    # M_-5 在极少数月末场景可能让同一交易日出现两行；npy 矩阵必须和 EOD 日期一一对应。
    out = out.drop_duplicates(subset=["交易日期"], keep="last")
    aligned = pd.DataFrame({"交易日期": pd.to_datetime(target_dates)})
    aligned = aligned.merge(out, on="交易日期", how="left")
    aligned.fillna(value=0, inplace=True)
    return aligned.rename(columns={"交易日期": "trade_date"})


def build_period_rebalance_matrix(dates: list[str], period_dict: dict[Any, list[Any]] | None = None) -> tuple[list[str], np.ndarray]:
    """把周期编号宽表转换成换仓日布尔矩阵。"""
    ids = build_period_id_frame(dates, period_dict=period_dict)
    period_names = [col for col in ids.columns if col != "trade_date"]
    values = ids[period_names].to_numpy(dtype=float)
    previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    rebalance = (values != previous) & (values != 0)
    if rebalance.shape[0] > 0:
        rebalance[0, :] = False
    return period_names, rebalance.T.astype(bool, copy=False)


def save_period_files(eod_path: Path, period_path: Path) -> tuple[Path, Path, Path]:
    """从 ``eod/dates.npy`` 生成三个周期预运算文件。"""
    eod_path = Path(eod_path)
    period_path = Path(period_path)
    period_path.mkdir(parents=True, exist_ok=True)
    dates = [normalize_date_label(x) for x in decode_array(np.load(eod_path / "dates.npy", allow_pickle=True))]
    period_names, rebalance = build_period_rebalance_matrix(dates)

    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    np.save(names_path, np.asarray(period_names, dtype=object))
    np.save(mask_path, rebalance)
    np.save(dates_path, np.asarray(dates, dtype="S8"))
    return names_path, mask_path, dates_path
