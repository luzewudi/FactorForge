# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """逐行计算 Pearson 相关系数；每一行代表一个交易日的横截面样本。"""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    n = valid.sum(axis=1)

    left_sum = np.where(valid, left, 0.0).sum(axis=1)
    right_sum = np.where(valid, right, 0.0).sum(axis=1)
    left_mean = np.divide(left_sum, n, out=np.full(n.shape, np.nan, dtype=float), where=n > 0)
    right_mean = np.divide(right_sum, n, out=np.full(n.shape, np.nan, dtype=float), where=n > 0)

    left_centered = np.where(valid, left - left_mean[:, None], 0.0)
    right_centered = np.where(valid, right - right_mean[:, None], 0.0)
    cov = (left_centered * right_centered).sum(axis=1)
    var_left = (left_centered**2).sum(axis=1)
    var_right = (right_centered**2).sum(axis=1)
    denom = np.sqrt(var_left * var_right)

    corr = np.full(n.shape, np.nan, dtype=float)
    mask = (n > 1) & (denom > 0)
    corr[mask] = cov[mask] / denom[mask]
    return corr


def rank_rows(matrix: np.ndarray) -> np.ndarray:
    """逐行做平均秩排名，用于把原始值转换成 RankIC 需要的秩变量。"""
    return pd.DataFrame(matrix).rank(axis=1, method="average", na_option="keep").to_numpy(dtype=float)


def pearson_ic(factor: np.ndarray, label: np.ndarray) -> np.ndarray:
    """计算每日 IC：因子值与未来收益 label 的横截面 Pearson 相关。"""
    return rowwise_corr(factor.T, label.T)


def rank_ic(factor: np.ndarray, label: np.ndarray) -> np.ndarray:
    """计算每日 RankIC：因子秩与未来收益秩的横截面 Spearman 相关。"""
    return rowwise_corr(rank_rows(factor.T), rank_rows(label.T))


def series_stats(series: pd.Series, annualization_factor: float = 1.0) -> dict[str, float]:
    """计算序列统计；均值保留原始方向，IR/胜率/t 值按有效方向展示因子强度。"""
    s = pd.Series(series, dtype=float).dropna()
    if s.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "annualized_ir": np.nan,
            "win_rate": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "count": 0,
        }
    direction = -1.0 if s.mean() < 0 else 1.0
    directional = s * direction
    std = s.std(ddof=0)
    if len(s) >= 2:
        test = stats.ttest_1samp(directional.to_numpy(dtype=float), 0.0, nan_policy="omit")
        t_value = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        t_value = np.nan
        p_value = np.nan
    ir = float(directional.mean() / std) if std and np.isfinite(std) else np.nan
    return {
        "mean": float(s.mean()),
        "std": float(std),
        "ir": ir,
        "annualized_ir": float(ir * annualization_factor) if np.isfinite(ir) else np.nan,
        "win_rate": float((directional > 0).mean()),
        "t_value": t_value,
        "p_value": p_value,
        "count": int(len(s)),
    }


def annual_ic_stats(ic: pd.Series, rankic: pd.Series, annualization_factor: float = 1.0) -> pd.DataFrame:
    """把 IC 和 RankIC 分别按年度和 TOTAL 输出同一套统计指标。"""
    rows = []
    for label, series in [("IC", ic), ("RankIC", rankic)]:
        by_year = series.groupby(series.index.year)
        for year, part in by_year:
            row = {"period": str(year), "series": label}
            stats_row = series_stats(part, annualization_factor=annualization_factor)
            row.update(stats_row)
            rows.append(row)
        row = {"period": "TOTAL", "series": label}
        stats_row = series_stats(series, annualization_factor=annualization_factor)
        row.update(stats_row)
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_t_value(series: pd.Series, window: int = 252, min_periods: int = 21) -> pd.Series:
    """计算滚动 t 值曲线，用于观察 IC 均值显著性的时间变化。"""
    s = pd.Series(series, dtype=float)

    def calc(values: np.ndarray) -> float:
        """单个滚动窗口内的 t 值计算，缺失值会被剔除。"""
        vals = values[np.isfinite(values)]
        if len(vals) < min_periods:
            return np.nan
        std = vals.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            return np.nan
        return float(vals.mean() / (std / np.sqrt(len(vals))))

    return s.rolling(window=window, min_periods=min_periods).apply(calc, raw=True)


def drawdown(nav: pd.Series) -> pd.Series:
    """根据净值序列计算回撤曲线，口径为当前净值/历史峰值-1。"""
    s = pd.Series(nav, dtype=float)
    peak = s.cummax()
    return s / peak - 1.0


def nav_metrics(nav: pd.Series, benchmark_nav: pd.Series | None = None, annual_days: int = 252) -> dict[str, float]:
    """计算策略净值的常用绩效指标，并在有基准时补充超额收益。"""
    nav = pd.Series(nav, dtype=float).dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann_return = (nav.iloc[-1] / nav.iloc[0]) ** (annual_days / max(len(nav) - 1, 1)) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(annual_days) if len(ret) > 1 else np.nan
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(annual_days) if len(ret) > 1 and ret.std(ddof=1) != 0 else np.nan
    dd = drawdown(nav)
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    result = {
        "start": str(nav.index[0].date() if hasattr(nav.index[0], "date") else nav.index[0]),
        "end": str(nav.index[-1].date() if hasattr(nav.index[-1], "date") else nav.index[-1]),
        "total_return": float(total_return),
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol) if np.isfinite(ann_vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "daily_mean": float(ret.mean()) if len(ret) else np.nan,
        "daily_std": float(ret.std(ddof=1)) if len(ret) > 1 else np.nan,
    }
    if benchmark_nav is not None and not benchmark_nav.empty:
        aligned = pd.concat([nav, benchmark_nav], axis=1, join="inner").dropna()
        if len(aligned) >= 2:
            excess_nav = aligned.iloc[:, 0] / aligned.iloc[:, 1]
            result["excess_return"] = float(excess_nav.iloc[-1] / excess_nav.iloc[0] - 1.0)
    return result


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    """把指标字典转换成两列表格，方便 CSV 和 HTML 报告统一展示。"""
    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
