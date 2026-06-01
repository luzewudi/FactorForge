import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

# 默认路径；也可以通过命令行参数或环境变量覆盖。
IS_LINUX = sys.platform.startswith("linux")
DEFAULT_EOD_PATH = Path("/mnt/ssd/eod") if IS_LINUX else Path("D:/凯纳/原始数据/eod")
DEFAULT_LABEL_PATH = Path("/home/luze/label") if IS_LINUX else Path("D:/凯纳/原始数据/label")
DEFAULT_PERIOD_PATH = Path("/home/luze/period") if IS_LINUX else Path("D:/凯纳/原始数据/period")


def calc_future_return(price, n):
    """
    Compute future n-day return.
    label[t] = price[t + n + 1] / price[t + 1] - 1
    Last n+1 columns filled with np.nan.
    """
    numerator = price[:, n + 1 :]
    if n == 0:
        denominator = price[:, :1]
        denominator = denominator.reshape(-1, 1)
    else:
        denominator = price[:, 1 : -n]
    label = numerator / denominator - 1.0
    label = _clean_label(label)
    nan_pad = np.full((price.shape[0], n + 1), np.nan, dtype=np.float64)
    label = np.concatenate([label, nan_pad], axis=1)
    return label


def calc_intraday(close, openp):
    """
    Intraday return: close[t+1] / open[t+1] - 1
    First column and last 1 column filled with np.nan.
    """
    label = close[:, 1:] / openp[:, 1:] - 1.0
    label = _clean_label(label)
    nan_pad = np.full((close.shape[0], 1), np.nan, dtype=np.float64)
    label = np.concatenate([nan_pad, label], axis=1)
    label[:, -1] = np.nan
    return label


def calc_period_future_return(price, period_names, period_dates, dates, period_prefix="W"):
    """
    根据 period 换仓日矩阵计算未来周期收益率。

    对 W 周期：
    - 如果 t 是 W_4，则用后面连续两个 W_0 换仓日价格计算：
      label[t] = price[next_next_W_0] / price[next_W_0] - 1
    - W_0/W_1/W_2/W_3 按同样逻辑使用下一个 offset 的连续两次换仓日。
    """
    if period_dates.shape[1] != price.shape[1]:
        raise ValueError(
            f"period_dates 列数 ({period_dates.shape[1]}) 与 price 列数 ({price.shape[1]}) 不一致"
        )

    period_indexes = _get_period_indexes(period_names, period_prefix)
    offsets = sorted(period_indexes)
    selected_offsets = _resolve_period_offsets(period_dates, dates, period_indexes, offsets)

    label = np.full(price.shape, np.nan, dtype=np.float64)
    for i, offset in enumerate(offsets):
        next_offset = offsets[(i + 1) % len(offsets)]
        event_cols = np.flatnonzero(period_dates[period_indexes[next_offset]])
        select_cols = np.flatnonzero(selected_offsets == offset)
        if event_cols.size < 2 or select_cols.size == 0:
            continue

        entry_pos = np.searchsorted(event_cols, select_cols, side="right")
        valid = entry_pos + 1 < event_cols.size
        if not np.any(valid):
            continue

        select_cols = select_cols[valid]
        entry_cols = event_cols[entry_pos[valid]]
        exit_cols = event_cols[entry_pos[valid] + 1]
        label[:, select_cols] = _clean_label(price[:, exit_cols] / price[:, entry_cols] - 1.0)

    return label


def _clean_label(label):
    """Replace inf, -1, and extreme values (|x| > 5) with nan."""
    valid_mask = np.isfinite(label) & (label != -1.0) & (np.abs(label) <= 5)
    return np.where(valid_mask, label, np.nan)


def _decode_text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _get_period_indexes(period_names, period_prefix):
    prefix = f"{period_prefix}_"
    indexes = {}
    for index, name in enumerate(period_names):
        text = _decode_text(name)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            indexes[int(suffix)] = index
    if not indexes:
        raise ValueError(f"period.npy 中没有找到 {period_prefix} 周期")
    return indexes


def _resolve_period_offsets(period_dates, dates, period_indexes, offsets):
    """正常交易日使用 period 标记；节假日挤压导致多标记/无标记时用自然星期兜底。"""
    ordered_indexes = [period_indexes[offset] for offset in offsets]
    membership = period_dates[ordered_indexes].T
    counts = membership.sum(axis=1)

    selected = np.full(period_dates.shape[1], -1, dtype=np.int16)
    single_rows = counts == 1
    if np.any(single_rows):
        selected[single_rows] = np.asarray(offsets, dtype=np.int16)[np.argmax(membership[single_rows], axis=1)]

    fallback_rows = ~single_rows
    if np.any(fallback_rows):
        weekday_offsets = np.asarray([_weekday_offset(value) for value in dates], dtype=np.int16)
        valid_weekday = np.isin(weekday_offsets, np.asarray(offsets, dtype=np.int16))
        fallback_valid = fallback_rows & valid_weekday
        selected[fallback_valid] = weekday_offsets[fallback_valid]

    return selected


def _weekday_offset(value):
    text = _decode_text(value).replace("-", "").replace("/", "")
    if len(text) != 8 or not text.isdigit():
        return -1
    weekday = datetime.strptime(text, "%Y%m%d").weekday()
    return weekday if 0 <= weekday <= 4 else -1


def load_period_data(period_path):
    """读取 period 文件；文件缺失时返回 None，让主流程跳过 W label。"""
    period_dir = Path(period_path).expanduser().resolve()
    period_dir.mkdir(parents=True, exist_ok=True)
    required_files = {
        "period.npy": period_dir / "period.npy",
        "period_dates.npy": period_dir / "period_dates.npy",
        "dates.npy": period_dir / "dates.npy",
    }
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        print("未找到完整 period 文件，跳过 W label 生成。")
        print(f"period 路径：{period_dir}")
        print(f"缺失文件：{', '.join(missing_files)}")
        print("请先运行 generate_period_files.py 生成 period 文件，或通过 --period-path 指定目录。")
        return None

    period_names = np.load(required_files["period.npy"], allow_pickle=True)
    period_dates = np.load(required_files["period_dates.npy"])
    dates = np.load(required_files["dates.npy"], allow_pickle=True)
    return period_names, period_dates, dates


def _resolve_path(value, default):
    path = Path(value) if value else Path(default)
    return path.expanduser().resolve()


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate label npy files from EOD data.")
    parser.add_argument(
        "--eod-path",
        default=os.environ.get("EOD_PATH"),
        help=f"EOD 数据目录，需包含 ClosePrice.npy/VWAP.npy/OpenPrice.npy。默认：{DEFAULT_EOD_PATH}",
    )
    parser.add_argument(
        "--output-path",
        "--label-path",
        dest="output_path",
        default=os.environ.get("LABEL_PATH"),
        help=f"label 输出目录。默认：{DEFAULT_LABEL_PATH}",
    )
    parser.add_argument(
        "--period-path",
        default=os.environ.get("PERIOD_PATH"),
        help=f"period 文件目录。默认：{DEFAULT_PERIOD_PATH}",
    )
    return parser.parse_args()


def _load_eod_array(eod_path, filename):
    path = eod_path / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到 EOD 文件：{path}")
    return np.load(path)


def main():
    args = _parse_args()
    eod_dir = _resolve_path(args.eod_path, DEFAULT_EOD_PATH)
    base_dir = _resolve_path(args.output_path, DEFAULT_LABEL_PATH)
    period_path = _resolve_path(args.period_path, DEFAULT_PERIOD_PATH)
    base_dir.mkdir(parents=True, exist_ok=True)

    close = _load_eod_array(eod_dir, "ClosePrice.npy")
    vwap  = _load_eod_array(eod_dir, "VWAP.npy")
    openp = _load_eod_array(eod_dir, "OpenPrice.npy")

    print(f"Loaded: shape={close.shape}, dtype={close.dtype}")
    print(f"EOD path: {eod_dir}")
    print(f"Output path: {base_dir}")

    items = [
        ("close_1d",    close,  1),
        ("close_3d",    close,  3),
        ("close_5d",    close,  5),
        ("close_10d",   close,  10),
        ("close_20d",   close,  20),
        ("close_21d",   close,  21),
        ("vwap_1d",     vwap,   1),
        ("vwap_3d",     vwap,   3),
        ("vwap_5d",     vwap,   5),
        ("vwap_10d",    vwap,   10),
        ("vwap_20d",    vwap,   20),
        ("vwap_21d",    vwap,   21),
        ("open_1d",     openp,  1),
        ("open_3d",     openp,  3),
        ("open_5d",     openp,  5),
        ("open_10d",    openp,  10),
        ("open_20d",    openp,  20),
        ("open_21d",    openp,  21),
    ]

    for name, price, n in items:
        label = calc_future_return(price, n)
        out_path = base_dir / f"{name}.npy"
        np.save(out_path, label)
        assert label.shape == close.shape, f"{name}: shape mismatch"
        assert label.dtype == np.float64,  f"{name}: dtype mismatch"
        nan_ratio = np.isnan(label).mean()
        print(f"Saved {name}.npy  shape={label.shape}  nan_ratio={nan_ratio:.4%}")

    # intraday_0d uses a separate formula: close[t+1] / open[t+1] - 1
    intraday_label = calc_intraday(close, openp)
    intraday_path = base_dir / "intraday_0d.npy"
    np.save(intraday_path, intraday_label)
    assert intraday_label.shape == close.shape, "intraday_0d: shape mismatch"
    assert intraday_label.dtype == np.float64,  "intraday_0d: dtype mismatch"
    print(f"Saved intraday_0d.npy  shape={intraday_label.shape}  nan_ratio={np.isnan(intraday_label).mean():.4%}")

    period_data = load_period_data(period_path)
    if period_data is None:
        return

    period_names, period_dates, period_calendar_dates = period_data
    period_items = [
        ("close_W", close),
        ("vwap_W",  vwap),
        ("open_W",  openp),
    ]

    for name, price in period_items:
        label = calc_period_future_return(price, period_names, period_dates, period_calendar_dates, "W")
        out_path = base_dir / f"{name}.npy"
        np.save(out_path, label)
        assert label.shape == close.shape, f"{name}: shape mismatch"
        assert label.dtype == np.float64,  f"{name}: dtype mismatch"
        nan_ratio = np.isnan(label).mean()
        print(f"Saved {name}.npy  shape={label.shape}  nan_ratio={nan_ratio:.4%}")


if __name__ == "__main__":
    main()
