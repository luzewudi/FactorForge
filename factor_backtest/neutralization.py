# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import BacktestData, DateWindow
from utils.path_kit import get_folder_by_root


def winsorize_cross_section(matrix: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """逐日横截面分位去极值，降低极端因子值对中性化回归的影响。"""
    arr = np.asarray(matrix, dtype=float).copy()
    for col in range(arr.shape[1]):
        values = arr[:, col]
        valid = np.isfinite(values)
        if valid.sum() < 3:
            continue
        lo = np.nanpercentile(values[valid], lower)
        hi = np.nanpercentile(values[valid], upper)
        arr[valid, col] = np.clip(values[valid], lo, hi)
    return arr


def standardize_cross_section(matrix: np.ndarray) -> np.ndarray:
    """逐日横截面标准化，让因子和暴露变量处于可比较的量纲。"""
    arr = np.asarray(matrix, dtype=float).copy()
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (arr - mean) / std
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def preprocess_exposure(matrix: np.ndarray) -> np.ndarray:
    """中性化前的统一预处理：先去极值，再做横截面标准化。"""
    return standardize_cross_section(winsorize_cross_section(matrix))


def neutralize_factor(
    factor: np.ndarray,
    mode: str,
    data: BacktestData,
    window: DateWindow,
    cache_path: Path | None = None,
) -> np.ndarray:
    """按配置执行市值或行业市值中性化，输出逐日横截面回归残差。"""
    # 中性化流程：先对原始因子做横截面去极值和标准化，再逐日回归取残差。
    mode = str(mode).strip().lower()
    if mode == "none":
        return factor
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == factor.shape:
            return np.asarray(cached, dtype=float)

    y_matrix = preprocess_exposure(factor)
    size = build_size_exposure(data, window)
    industry = build_industry_exposure(data, window) if mode == "industry_market" else None

    result = np.full(factor.shape, np.nan, dtype=float)
    for date_idx in range(factor.shape[1]):
        y = y_matrix[:, date_idx]
        exposures = [size[:, date_idx]]
        if industry is not None:
            exposures.extend(ind[:, date_idx] for ind in industry)
        x = np.column_stack(exposures)
        valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        if valid.sum() <= x.shape[1]:
            continue
        try:
            beta = np.linalg.lstsq(x[valid], y[valid], rcond=None)[0]
            resid = y[valid] - x[valid].dot(beta)
            result[valid, date_idx] = resid
        except np.linalg.LinAlgError:
            continue

    if cache_path:
        get_folder_by_root(cache_path.parent)
        np.save(cache_path, result)
        np.save(cache_path.with_name(cache_path.stem + "_dates.npy"), np.asarray(window.dates, dtype="S8"))
        np.save(cache_path.with_name(cache_path.stem + "_tickers.npy"), np.asarray(data.tickers, dtype="S16"))
    return result


def build_size_exposure(data: BacktestData, window: DateWindow) -> np.ndarray:
    """构造市值暴露变量：使用 log(CAPQ0_MKTCAP) 并做横截面预处理。"""
    cap = data.load_market_cap(window)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_cap = np.log(cap)
    return preprocess_exposure(log_cap)


def build_industry_exposure(data: BacktestData, window: DateWindow) -> list[np.ndarray]:
    """构造行业哑变量暴露；行业市值中性化时与 SIZE 一起作为回归自变量。"""
    raw = data.load_industry(window)
    with np.errstate(invalid="ignore"):
        industry = np.floor(raw / 10000.0)
    industry[~np.isfinite(industry)] = np.nan
    codes = sorted(int(x) for x in np.unique(industry[np.isfinite(industry)]) if int(x) != 0)
    comprehensive = load_comprehensive_code(data.data_fund_path)
    if comprehensive in codes:
        codes.remove(comprehensive)
    if len(codes) > 1:
        codes = codes[:-1]
    exposures = []
    for code in codes:
        dummy = np.where(industry == code, 1.0, 0.0)
        dummy[~np.isfinite(industry)] = np.nan
        exposures.append(dummy)
    return exposures


def load_comprehensive_code(data_fund_path: Path) -> int | None:
    """从 sw1.csv 中读取“综合”行业代码，用于行业哑变量中剔除该行业。"""
    path = Path(data_fund_path) / "sw1.csv"
    if not path.exists():
        return None
    for encoding in ("gbk", "utf-8", "utf-8-sig"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None
    if "行业名称" not in df.columns or "行业代码" not in df.columns:
        return None
    hit = df[df["行业名称"].astype(str) == "综合"]
    if hit.empty:
        return None
    try:
        return int(hit["行业代码"].iloc[0]) // 10000
    except Exception:
        return None
