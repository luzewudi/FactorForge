# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config_loader import BacktestConfig, strip_npy_suffix


def decode_array(values: Iterable) -> list[str]:
    """把 npy 中可能存在的 bytes/string 混合元数据统一解码成字符串列表。"""
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.append(str(value).strip())
    return out


def normalize_date_label(value: str) -> str:
    """把日期标签统一成 YYYYMMDD，兼容 YYYY-MM-DD 和 YYYY/MM/DD。"""
    text = str(value).strip().replace("-", "").replace("/", "")
    return text if len(text) == 8 and text.isdigit() else str(value).strip()


def normalize_ticker_label(value: str) -> str:
    """把股票代码统一成 6 位代码，兼容 SH/SZ/BJ 前缀格式。"""
    text = str(value).strip().upper()
    if len(text) == 8 and text[:2] in {"SH", "SZ", "BJ"}:
        return text[2:]
    return text


def normalize_trade_price_name(name: str) -> str:
    """统一交易价格字段名称，避免 period_vwap 和 periodvwap 两种写法分裂。"""
    text = str(name).strip().lower()
    if text == "period_vwap":
        return "periodvwap"
    return text


def trade_price_file(name: str) -> str:
    """把配置中的交易价格名称映射到 EOD 目录里的具体 npy 文件名。"""
    text = normalize_trade_price_name(name)
    if text == "open":
        return "OpenPrice.npy"
    if text == "close":
        return "ClosePrice.npy"
    if text == "vwap":
        return "VWAP.npy"
    if text == "periodvwap":
        return "PeriodVWAP.npy"
    raise ValueError(f"unsupported trade_price: {name}")


def label_price_prefix(name: str) -> str:
    """把交易价格名称映射到 label 文件前缀，例如 vwap_1d.npy。"""
    text = normalize_trade_price_name(name)
    if text == "periodvwap":
        text = "vwap"
    return text


def clean_numeric(arr: np.ndarray) -> np.ndarray:
    """把行情矩阵转成 float，并将 inf 和 0 统一视为缺失值。"""
    out = np.asarray(arr, dtype=float)
    out = np.where(np.isfinite(out), out, np.nan)
    out = np.where(out == 0.0, np.nan, out)
    return out


@dataclass
class DateWindow:
    start: int
    end: int
    dates: list[str]

    @property
    def slice(self) -> slice:
        """返回可直接用于 numpy 日期轴切片的 slice。"""
        return slice(self.start, self.end)

    @property
    def pandas_index(self) -> pd.DatetimeIndex:
        """返回报告和 CSV 使用的 pandas 日期索引。"""
        return pd.to_datetime(self.dates, format="%Y%m%d")


class BacktestData:
    def __init__(self, config: BacktestConfig):
        """初始化数据访问层，并加载 EOD 日期、股票代码等公共元数据。"""
        self.config = config
        self.eod_path = config.paths.eod_path
        self.fund_path = config.paths.fund_path
        self.data_fund_path = config.paths.data_fund_path
        self.dates = [normalize_date_label(x) for x in decode_array(np.load(self.eod_path / "dates.npy", allow_pickle=True))]
        self.tickers = [
            normalize_ticker_label(x) for x in decode_array(np.load(self.eod_path / "ticker_names.npy", allow_pickle=True))
        ]
        self._ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        self._array_cache: dict[Path, np.ndarray] = {}

    def date_window(self, start_date: str, end_date: str) -> DateWindow:
        """根据配置日期在 EOD 交易日中定位闭区间窗口。"""
        start = int(np.searchsorted(self.dates, start_date, side="left"))
        end = int(np.searchsorted(self.dates, end_date, side="right"))
        if start >= end:
            raise ValueError(f"empty date window: {start_date} to {end_date}")
        return DateWindow(start=start, end=end, dates=self.dates[start:end])

    def npy(self, path: Path, mmap: bool = True) -> np.ndarray:
        """带缓存读取 npy；默认使用 mmap，避免大矩阵反复完整载入内存。"""
        path = Path(path)
        if path not in self._array_cache:
            if mmap:
                self._array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=True)
            else:
                self._array_cache[path] = np.load(path, allow_pickle=True)
        return self._array_cache[path]

    def load_eod_panel(self, file_name: str, window: DateWindow | None = None) -> np.ndarray:
        """读取 EOD 矩阵并统一成 stock x date 形状，这是全项目的标准矩阵方向。"""
        arr = self.npy(self.eod_path / file_name)
        if arr.ndim != 2:
            raise ValueError(f"{file_name} must be 2D")
        if arr.shape[0] == len(self.tickers):
            panel = arr[:, window.slice] if window else arr
        elif arr.shape[1] == len(self.tickers):
            panel = arr[window.slice, :].T if window else arr.T
        else:
            raise ValueError(f"{file_name} shape {arr.shape} does not match EOD metadata")
        return clean_numeric(panel)

    def load_panel_from_dir(
        self,
        folder: Path,
        file_name: str,
        window: DateWindow | None = None,
        zero_as_nan: bool = False,
    ) -> np.ndarray:
        """读取 fund/data_fund 等目录下的矩阵，并按 EOD 元数据重新对齐。"""
        path = Path(folder) / file_name
        arr = self.npy(path)
        panel = self._align_panel(arr, Path(folder), window)
        panel = np.asarray(panel, dtype=float)
        panel = np.where(np.isfinite(panel), panel, np.nan)
        if zero_as_nan:
            panel = np.where(panel == 0.0, np.nan, panel)
        return panel

    def _align_panel(self, arr: np.ndarray, metadata_dir: Path, window: DateWindow | None) -> np.ndarray:
        """根据源目录的 dates/ticker_names 判断矩阵方向，并必要时重排到 EOD 口径。"""
        if arr.ndim != 2:
            raise ValueError(f"array at {metadata_dir} must be 2D, got {arr.shape}")

        meta_dates = self._load_metadata(metadata_dir, "dates.npy")
        meta_tickers = self._load_metadata(metadata_dir, "ticker_names.npy")
        if meta_dates and meta_tickers:
            if arr.shape == (len(meta_tickers), len(meta_dates)):
                src = arr
            elif arr.shape == (len(meta_dates), len(meta_tickers)):
                src = arr.T
            else:
                src = arr
            return self._reindex_panel(src, meta_tickers, meta_dates, window)

        if arr.shape[0] == len(self.tickers):
            return arr[:, window.slice] if window else arr
        if arr.shape[1] == len(self.tickers):
            return arr[window.slice, :].T if window else arr.T
        raise ValueError(f"cannot infer panel orientation for shape {arr.shape}")

    @staticmethod
    def _load_metadata(folder: Path, name: str) -> list[str]:
        """读取某个目录下的日期或股票元数据，并做统一格式清洗。"""
        path = folder / name
        if not path.exists():
            return []
        values = decode_array(np.load(path, allow_pickle=True))
        if "date" in name.lower():
            return [normalize_date_label(value) for value in values]
        if "ticker" in name.lower():
            return [normalize_ticker_label(value) for value in values]
        return values

    def _reindex_panel(
        self,
        panel: np.ndarray,
        src_tickers: list[str],
        src_dates: list[str],
        window: DateWindow | None,
    ) -> np.ndarray:
        """把源矩阵按股票和日期双维度重排到目标 EOD 股票池和日期窗口。"""
        target_dates = window.dates if window else self.dates
        out = np.full((len(self.tickers), len(target_dates)), np.nan, dtype=float)
        ticker_map = {ticker: i for i, ticker in enumerate(src_tickers)}
        date_map = {date: i for i, date in enumerate(src_dates)}

        row_pairs = [(target_i, ticker_map[ticker]) for target_i, ticker in enumerate(self.tickers) if ticker in ticker_map]
        col_pairs = [(target_i, date_map[date]) for target_i, date in enumerate(target_dates) if date in date_map]
        if not row_pairs or not col_pairs:
            return out

        target_rows = np.array([p[0] for p in row_pairs], dtype=int)
        src_rows = np.array([p[1] for p in row_pairs], dtype=int)
        target_cols = np.array([p[0] for p in col_pairs], dtype=int)
        src_cols = np.array([p[1] for p in col_pairs], dtype=int)
        out[np.ix_(target_rows, target_cols)] = np.asarray(panel[np.ix_(src_rows, src_cols)], dtype=float)
        return out

    def load_factor(self, factor_path: Path, window: DateWindow) -> np.ndarray:
        """读取单个因子文件，并兼容因子自带 dates/tickers 或直接使用 EOD 元数据两种情况。"""
        arr = self.npy(factor_path)
        stem = strip_npy_suffix(Path(factor_path).name)
        factor_dates = self._load_metadata(factor_path.parent, f"{stem}_dates.npy") or self._load_metadata(
            factor_path.parent, "dates.npy"
        )
        factor_tickers = self._load_metadata(factor_path.parent, f"{stem}_tickers.npy") or self._load_metadata(
            factor_path.parent, "ticker_names.npy"
        )

        if factor_dates and factor_tickers:
            if arr.shape == (len(factor_tickers), len(factor_dates)):
                src = arr
            elif arr.shape == (len(factor_dates), len(factor_tickers)):
                src = arr.T
            else:
                src = arr
            panel = self._reindex_panel(src, factor_tickers, factor_dates, window)
        else:
            panel = self._align_panel(arr, self.config.paths.factor_folder, window)
        return np.where(np.isfinite(panel), panel, np.nan)

    def load_label(self, trade_price: str, turnover: int, window: DateWindow) -> np.ndarray:
        """按 generate_labels.py 约定读取 label，例如 vwap_1d.npy，并裁剪到研究窗口。"""
        prefix = label_price_prefix(trade_price)
        path = self.config.paths.label_path / f"{prefix}_{int(turnover)}d.npy"
        if not path.exists():
            raise FileNotFoundError(f"label file not found: {path}")
        arr = self.npy(path)
        if arr.shape[0] == len(self.tickers):
            panel = arr[:, window.slice]
        elif arr.shape[1] == len(self.tickers):
            panel = arr[window.slice, :].T
        else:
            raise ValueError(f"label shape {arr.shape} does not match EOD metadata")
        return np.where(np.isfinite(panel), panel, np.nan)

    def load_universe(self, universe: str, window: DateWindow) -> np.ndarray:
        """读取股票池掩码；支持全市场、宽基指数成分、Top 市值和申万行业股票池。"""
        text = str(universe).strip().lower()
        shape = (len(self.tickers), len(window.dates))
        if text in {"0", "all", "a", "market"}:
            return np.ones(shape, dtype=bool)
        if text in {"300", "hs300"}:
            return self.load_panel_from_dir(self.data_fund_path, "HS300_NO_WGT.npy", window) > 0
        if text in {"500", "zz500"}:
            return self.load_panel_from_dir(self.data_fund_path, "ZZ500_NO_WGT.npy", window) > 0
        if text in {"800", "zz800"}:
            return self.load_panel_from_dir(self.data_fund_path, "ZZ800_NO_WGT.npy", window) > 0
        if text in {"1000", "zz1000"}:
            return self.load_panel_from_dir(self.data_fund_path, "ZZ1000_NO_WGT.npy", window) > 0
        if text.startswith("mkt"):
            n = int(text.replace("mkt_", "").replace("mkt", ""))
            cap = self.load_market_cap(window)
            return top_n_mask(cap, n)
        if text.startswith("sw"):
            code = int(text.replace("sw_", "").replace("sw", ""))
            sw = self.load_industry(window)
            return (sw // 10000).astype(float) == float(code)
        raise ValueError(f"unsupported universe: {universe}")

    def load_market_cap(self, window: DateWindow) -> np.ndarray:
        """读取总市值矩阵，主要用于市值加权和市值中性化。"""
        return self.load_panel_from_dir(self.fund_path, "CAPQ0_MKTCAP.npy", window, zero_as_nan=True)

    def load_industry(self, window: DateWindow) -> np.ndarray:
        """读取申万行业矩阵，优先使用新口径 SWIND_NEW1。"""
        path = self.data_fund_path / "SWIND_NEW1.npy"
        if path.exists():
            return self.load_panel_from_dir(self.data_fund_path, "SWIND_NEW1.npy", window)
        return self.load_panel_from_dir(self.data_fund_path, "SWIND.npy", window)

    def load_benchmark_nav(self, benchmark: str, window: DateWindow, compound: bool) -> pd.Series:
        """读取指数收盘价并生成基准净值；Step1 可单利，Step2 使用复利。"""
        code = str(benchmark).strip()
        if not code or code.lower() in {"none", "nan"}:
            return pd.Series(dtype=float)
        index_dir = self.eod_path / "index"
        names_path = index_dir / "ticker_names.npy"
        close_path = index_dir / "ClosePrice.npy"
        if not names_path.exists() or not close_path.exists():
            return pd.Series(dtype=float)
        names = decode_array(np.load(names_path, allow_pickle=True))
        try:
            row = names.index(code)
        except ValueError:
            hits = [i for i, name in enumerate(names) if name == code or name.endswith(code)]
            if not hits:
                return pd.Series(dtype=float)
            row = hits[0]
        close = self.npy(close_path)[row, window.slice].astype(float)
        close = pd.Series(np.where(np.isfinite(close) & (close != 0), close, np.nan), index=window.pandas_index)
        close = close.ffill()
        if close.dropna().empty:
            return pd.Series(dtype=float)
        if compound:
            base = close.dropna().iloc[0]
            return close / base
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return 1.0 + returns.cumsum()

    def load_simulation_price_panels(self, trade_price: str, window: DateWindow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取 Step2 所需的成交价、估值收盘价和涨跌停状态。"""
        trade = self.load_eod_panel(trade_price_file(trade_price), window)
        close = self.load_eod_panel("ClosePrice.npy", window)
        limit_status = self.load_eod_panel("UpDownLimitStatus.npy", window)
        limit_status = np.nan_to_num(limit_status, nan=0.0)
        return trade, close, limit_status

    def load_analysis_price_panels(self, trade_price: str, window: DateWindow) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """读取 Step1 分层 PnL 所需的 PRC、PrePRC、成交价和复权因子。"""
        prc = self.load_eod_panel("PRC.npy", window)
        pre_prc = self.load_eod_panel("PrePRC.npy", window)
        execution_price = self.load_eod_panel(trade_price_file(trade_price), window)
        adj_factor = self.load_eod_panel("AdjFactor.npy", window)
        return prc, pre_prc, execution_price, adj_factor


def top_n_mask(values: np.ndarray, n: int) -> np.ndarray:
    """逐日取数值最大的前 n 只股票，用于 mkt_n 股票池。"""
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=bool)
    if n <= 0:
        return out
    for col in range(arr.shape[1]):
        data = arr[:, col]
        valid = np.isfinite(data)
        if not valid.any():
            continue
        valid_idx = np.where(valid)[0]
        take = min(n, valid_idx.size)
        order = np.argsort(data[valid_idx], kind="mergesort")
        out[valid_idx[order[-take:]], col] = True
    return out
