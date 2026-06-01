# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import normalize_offset_label, normalize_period
from .data_loader import decode_array, normalize_date_label


def default_period_dict() -> dict[Any, list[Any]]:
    """Return the supported period/offset universe used by the precompute script."""
    return {
        1: [0],
        2: list(range(2)),
        3: list(range(3)),
        4: list(range(4)),
        5: list(range(5)),
        10: list(range(10)),
        20: list(range(20)),
        21: list(range(21)),
        "W": list(range(5)),
        "2W": [0, 1],
        "3W": [0, 1, 2],
        "4W": [0, 1, 2, 3],
        "5W": [0, 1, 2, 3, 4],
        "6W": [0, 1, 2, 3, 4, 5],
        "M": [0, -5],
        "W53": [0],
    }


def load_period_files(period_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Load precomputed period names, rebalance masks and dates."""
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
    """Resolve a period + offsets config into concrete keys such as 20_7."""
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
    """Build a period key with the same naming convention as period.npy."""
    return f"{normalize_period(period)}_{normalize_offset_label(offset)}"


def sort_period_keys(keys: list[str]) -> list[str]:
    """Sort period keys by their offset suffix while keeping string offsets stable."""

    def _key(value: str) -> tuple[int, int | str]:
        suffix = value.split("_", 1)[1] if "_" in value else value
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(keys, key=_key)


def build_period_id_frame(dates: list[str], period_dict: dict[Any, list[Any]] | None = None, min_day: int = 2) -> pd.DataFrame:
    """Build period-id columns from EOD trading dates using the legacy period logic."""
    period_dict = period_dict or default_period_dict()
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

    out.fillna(0, inplace=True)
    return out


def build_period_rebalance_matrix(dates: list[str], period_dict: dict[Any, list[Any]] | None = None) -> tuple[list[str], np.ndarray]:
    """Return period names and a bool matrix where True means that date is a rebalance date."""
    ids = build_period_id_frame(dates, period_dict=period_dict)
    period_names = [col for col in ids.columns if col != "trade_date"]
    values = ids[period_names].to_numpy(dtype=float)
    previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    rebalance = (values != previous) & (values != 0)
    if rebalance.shape[0] > 0:
        rebalance[0, :] = False
    return period_names, rebalance.T.astype(bool, copy=False)


def save_period_files(eod_path: Path, period_path: Path) -> tuple[Path, Path, Path]:
    """Generate period.npy, period_dates.npy and dates.npy from an EOD dates.npy file."""
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


def _single_period_ids(base: pd.DataFrame, period_type: Any, offset: Any, min_day: int) -> pd.DataFrame:
    data = base.copy()
    col = format_period_key(period_type, offset)
    agg_dict = {"period_end": "last", "is_trade": "sum"}

    if isinstance(period_type, int):
        index = np.arange(len(data), dtype=float)
        data["group"] = np.trunc((index - int(offset)) / period_type).astype(int)
        period_df = data.groupby("group").agg(agg_dict)
        period_df.rename(columns={"is_trade": "trading_days"}, inplace=True)
        period_df["trading_days"] = max(int(period_type), min_day)
    else:
        period_text = str(period_type).upper()
        if period_text == "M" and str(offset) == "-5":
            period_df = _month_minus_five_periods(data)
        else:
            data = _shift_trade_dates(data, period_text, offset)
            data.loc[data.index.max() + 1, "trade_date"] = pd.to_datetime("1990-01-01")
            data.set_index("trade_date", inplace=True)
            period_df = data.resample(rule=_pandas_resample_rule(period_text)).agg(agg_dict)
            period_df.rename(columns={"is_trade": "trading_days"}, inplace=True)

    period_df = period_df[period_df["trading_days"] > 0]
    period_df = _merge_short_periods(period_df, min_day=min_day)
    period_df.reset_index(drop=True, inplace=True)
    period_df.rename(columns={"period_end": "trade_date"}, inplace=True)
    period_df = period_df.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    period_df[col] = 1

    out = base[["trade_date"]].merge(period_df[["trade_date", col]], on="trade_date", how="left")
    out[col] = out[col].expanding().sum().shift()
    return out


def _shift_trade_dates(data: pd.DataFrame, period_text: str, offset: Any) -> pd.DataFrame:
    if any(char.isdigit() for char in period_text):
        if isinstance(offset, str) and "D" in offset.upper():
            data["trade_date"] -= pd.to_timedelta(offset.upper())
        else:
            data["trade_date"] -= pd.to_timedelta(f"{int(offset) * 7}D")
    else:
        data["trade_date"] -= pd.to_timedelta(f"{int(offset)}D")
    return data


def _pandas_resample_rule(period_text: str) -> str:
    """Use current pandas aliases while keeping public period names unchanged."""
    return "ME" if period_text == "M" else period_text


def _month_minus_five_periods(data: pd.DataFrame) -> pd.DataFrame:
    start_date = data["trade_date"].min()
    end_date = data["trade_date"].max()
    natural = data.set_index("trade_date").reindex(pd.date_range(start=start_date, end=end_date, freq="D")).reset_index()
    natural.rename(columns={"index": "trade_date"}, inplace=True)
    natural["period_end"] = natural["period_end"].ffill()
    natural["is_trade"] = natural["is_trade"].fillna(value=0)

    month_end = pd.DataFrame({"trade_date": pd.date_range(start=start_date, end=end_date, freq="ME")})
    month_end["month_end"] = 1
    natural = natural.merge(month_end, on="trade_date", how="left")
    natural.loc[(natural["month_end"] == 1) & (natural["is_trade"].shift(-1) == 0), "is_trade"] = 0
    natural = natural[natural["is_trade"] == 0].set_index("trade_date")
    period_df = natural.resample(rule="ME").agg({"period_end": "last", "is_trade": "sum"})
    period_df["trading_days"] = 20
    return period_df


def _merge_short_periods(period_df: pd.DataFrame, min_day: int) -> pd.DataFrame:
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
    col = "W53_0"
    data = base.copy()
    start_date = data["trade_date"].min()
    end_date = data["trade_date"].max()

    shifted = data.copy()
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

    natural = pd.DataFrame({"trade_date": pd.date_range(start=start_date, end=end_date, freq="D")})
    natural["period_end"] = natural["trade_date"]
    natural["trade_date"] -= pd.to_timedelta("4D")
    natural.loc[natural.index.max() + 1, "trade_date"] = pd.to_datetime("1990-01-01")
    natural.set_index("trade_date", inplace=True)
    thursday_df = natural.resample(rule="W").agg({"period_end": "last"})
    out.loc[out["trade_date"].isin(thursday_df["period_end"]), col] = None

    counts = out[col].value_counts()
    single_values = counts[counts == 1].index
    out.loc[out[col].isin(single_values), col] = None
    marker = out[col].copy()
    out[col] = out[col].ffill()
    out.loc[pd.isnull(marker), col] = -out[col]
    return out[["trade_date", col]]
