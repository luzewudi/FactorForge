# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config_loader import BacktestConfig, resolve_factor_files
from .data_loader import BacktestData, DateWindow
from .factor_analysis import apply_neutralization, stable_softmax
from .metrics import metrics_to_frame, nav_metrics
from .plot_results import write_simulation_html, write_summary_html
from .simulator import Simulator, vector_to_limit_dict, vector_to_price_dict
from utils.log_kit import logger
from utils.path_kit import PROJECT_ROOT, get_folder_by_root


@dataclass
class SimulationResult:
    factor_name: str
    nav: pd.DataFrame
    trades: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    html_path: Path


def run_simulation(config: BacktestConfig) -> list[SimulationResult]:
    """执行 Step2 真实账户模拟主流程，逐因子输出净值、交易明细和 HTML 报告。"""
    # Step1：加载公共行情数据，并把所有矩阵裁剪到配置中的回测日期区间。
    data = BacktestData(config)
    window = data.date_window(config.analysis.start_date, config.analysis.end_date)
    out_root = Path(get_folder_by_root(config.paths.output_folder, "step2_simulation"))

    # Step2：读取成交价、估值价、交易状态、涨跌停和复利基准净值。
    # simulation.universe 是真实账户模拟的独立股票池，允许和 Step1 因子分析股票池不同。
    universe = data.load_universe(config.simulation.universe, window)
    period_masks = data.load_period_masks(config.simulation.period, config.simulation.offsets, window)
    trade_status = data.load_eod_panel("TradeStatus.npy", window)
    trade_prices, close_prices, limit_status = data.load_simulation_price_panels(config.simulation.trade_price, window)
    benchmark_nav = data.load_benchmark_nav(config.simulation.benchmark, window, compound=True)
    if benchmark_nav.empty:
        logger.warning(
            f"[step2] 指数 benchmark={config.simulation.benchmark} 未找到，"
            "本次报告不会绘制 benchmark / excess_nav。"
        )
    market_cap = data.load_market_cap(window)
    industry = data.load_industry(window)
    industry_names = load_industry_name_map(config)

    results: list[SimulationResult] = []
    summary_rows: list[dict] = []
    factor_files = resolve_factor_files(config, stage="simulation")
    total_factors = len(factor_files)
    logger.info(f"[step2] 共发现 {total_factors} 个因子，开始逐个模拟回测。")
    logger.info(
        f"[step2] 股票池={config.simulation.universe}，选股数量={config.simulation.select_n}，"
        f"周期={config.simulation.period}，offset数={len(period_masks)}，加权方法={config.simulation.weight_method}。"
    )
    for factor_idx, factor_path in enumerate(factor_files, start=1):
        started_at = perf_counter()
        factor_name = factor_path.stem
        logger.info(f"[step2][{factor_idx}/{total_factors}] 开始模拟因子：{factor_name}")

        # Step3：因子对齐、股票池过滤、中性化都复用 Step1 的统一逻辑，保证研究和模拟口径一致。
        logger.debug(f"[step2][{factor_idx}/{total_factors}] {factor_name} - Step3：读取、对齐并处理中性化。")
        factor = data.load_factor(factor_path, window)
        factor = np.where(universe, factor, np.nan)
        factor = apply_neutralization(config, data, window, factor_name, factor)
        try:
            factor_direction, rankic_mean = load_step1_rankic_direction(config, factor_name)
        except ValueError as exc:
            logger.warning(f"[step2][{factor_idx}/{total_factors}] 跳过因子 {factor_name}：{exc}")
            continue
        direction_text = "选因子值小的股票" if factor_direction else "选因子值大的股票"
        logger.info(
            f"[step2][{factor_idx}/{total_factors}] {factor_name} - "
            f"读取 Step1 RankIC 均值 {rankic_mean:.6f}，自动方向：{direction_text}。"
        )

        logger.debug(f"[step2][{factor_idx}/{total_factors}] {factor_name} - Step4：进入真实账户逐日模拟。")
        result = simulate_one_factor(
            config=config,
            data=data,
            window=window,
            factor_name=factor_name,
            factor=factor,
            universe=universe,
            period_masks=period_masks,
            trade_status=trade_status,
            trade_prices=trade_prices,
            close_prices=close_prices,
            limit_status=limit_status,
            market_cap=market_cap,
            industry=industry,
            industry_names=industry_names,
            benchmark_nav=benchmark_nav,
            out_root=out_root,
            factor_direction=factor_direction,
            rankic_mean=rankic_mean,
        )
        results.append(result)
        summary_rows.append(build_summary_row(result))
        elapsed = perf_counter() - started_at
        logger.ok(f"[step2][{factor_idx}/{total_factors}] 完成因子：{factor_name}，耗时 {elapsed:.1f} 秒。")

    summary = pd.DataFrame(summary_rows)
    _write_csv_safely(summary, out_root / "summary.csv", index=False)
    write_summary_html(summary, out_root / "summary.html", "Step2 Simulation Summary")
    logger.ok(f"[step2] 完成 {len(results)}/{total_factors} 个有效因子模拟，汇总报告：{out_root / 'summary.html'}")
    return results


def simulate_one_factor(
    config: BacktestConfig,
    data: BacktestData,
    window: DateWindow,
    factor_name: str,
    factor: np.ndarray,
    universe: np.ndarray,
    period_masks: dict[str, np.ndarray],
    trade_status: np.ndarray,
    trade_prices: np.ndarray,
    close_prices: np.ndarray,
    limit_status: np.ndarray,
    market_cap: np.ndarray | None,
    industry: np.ndarray | None,
    industry_names: dict[int, str],
    benchmark_nav: pd.Series,
    out_root: Path,
    factor_direction: bool,
    rankic_mean: float,
) -> SimulationResult:
    """模拟单个因子的真实账户回测：上一日因子选股，下一交易日成交并收盘估值。"""
    sim_cfg = config.simulation
    date_index = window.pandas_index
    if not period_masks:
        raise ValueError("period_masks is empty")
    sleeve_initial = sim_cfg.initial_capital / len(period_masks)

    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    simulators: list[Simulator] = []

    for period_key, rebalance_mask in period_masks.items():
        simulator, daily, trades, selections = simulate_one_offset(
            config=config,
            data=data,
            window=window,
            factor=factor,
            universe=universe,
            trade_status=trade_status,
            trade_prices=trade_prices,
            close_prices=close_prices,
            limit_status=limit_status,
            market_cap=market_cap,
            factor_direction=factor_direction,
            period_key=period_key,
            rebalance_mask=rebalance_mask,
            initial_capital=sleeve_initial,
        )
        simulators.append(simulator)
        daily_frames.append(daily)
        if not trades.empty:
            trade_frames.append(trades)
        if not selections.empty:
            selection_frames.append(selections)

    # 将多 offset sleeve 记录整理为合成净值、交易流水、选股结果和指标表，分别落盘并写入 HTML。
    nav_df, offset_nav_df = build_ensemble_nav(
        date_index=date_index,
        daily_frames=daily_frames,
        benchmark_nav=benchmark_nav,
        initial_capital=sim_cfg.initial_capital,
    )
    if benchmark_nav is not None and not benchmark_nav.empty:
        nav_df["benchmark_nav"] = benchmark_nav.reindex(nav_df.index).ffill()
        nav_df["excess_nav"] = nav_df["strategy_nav"] / nav_df["benchmark_nav"]

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    selections_df = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    selection_profile_df = build_selection_profile(
        selections_df=selections_df,
        tickers=data.tickers,
        dates=window.dates,
        industry=industry,
        industry_names=industry_names,
        market_cap=market_cap,
    )
    metrics = nav_metrics(nav_df["strategy_nav"], nav_df.get("benchmark_nav"))
    metrics.update(
        {
            "total_commission": sum(simulator.total_commission for simulator in simulators),
            "total_stamp_tax": sum(simulator.total_stamp_tax for simulator in simulators),
            "avg_turnover": float(nav_df["turnover"].mean()) if "turnover" in nav_df else np.nan,
            "trade_count": int(len(trades_df[trades_df["shares"] > 0])) if not trades_df.empty else 0,
            "rankic_mean_from_step1": rankic_mean,
            "auto_factor_direction": "small_is_better" if factor_direction else "large_is_better",
            "benchmark_status": "found" if benchmark_nav is not None and not benchmark_nav.empty else f"missing:{sim_cfg.benchmark}",
            "simulation_universe": sim_cfg.universe,
            "weight_method": sim_cfg.weight_method,
            "period": sim_cfg.period,
            "offset_count": len(period_masks),
        }
    )
    metrics_df = metrics_to_frame(metrics)

    factor_out = Path(get_folder_by_root(out_root, factor_name))
    _write_csv_safely(nav_df, factor_out / "nav.csv")
    _write_csv_safely(offset_nav_df, factor_out / "offset_nav.csv")
    _write_csv_safely(trades_df, factor_out / "trades.csv", index=False)
    _write_csv_safely(selections_df, factor_out / "selections.csv", index=False)
    _write_csv_safely(selection_profile_df, factor_out / "selection_profile.csv", index=False)
    _write_csv_safely(metrics_df, factor_out / "metrics.csv", index=False)
    html_path = factor_out / f"{factor_name}_simulation.html"
    write_simulation_html(
        html_path,
        factor_name,
        nav_df,
        metrics_df,
        trades_df,
        selections_df,
        selection_profile_df,
        offset_nav_df=offset_nav_df,
    )
    maybe_write_quantstats(config, factor_name, nav_df, factor_out)

    return SimulationResult(factor_name, nav_df, trades_df, selections_df, metrics_df, html_path)


def simulate_one_offset(
    config: BacktestConfig,
    data: BacktestData,
    window: DateWindow,
    factor: np.ndarray,
    universe: np.ndarray,
    trade_status: np.ndarray,
    trade_prices: np.ndarray,
    close_prices: np.ndarray,
    limit_status: np.ndarray,
    market_cap: np.ndarray | None,
    factor_direction: bool,
    period_key: str,
    rebalance_mask: np.ndarray,
    initial_capital: float,
) -> tuple[Simulator, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one offset sleeve as an independent trading account."""
    sim_cfg = config.simulation
    simulator = Simulator(
        initial_cash=initial_capital,
        commission_rate=sim_cfg.fee,
        stamp_tax_rate=sim_cfg.stamp_duty,
    )
    selections: list[dict] = []
    rebalance_mask = np.asarray(rebalance_mask, dtype=bool)

    for date_idx, date in enumerate(window.dates):
        turnover = 0.0
        close_dict = vector_to_price_dict(data.tickers, close_prices[:, date_idx])
        if date_idx > 0 and date_idx < len(rebalance_mask) and rebalance_mask[date_idx]:
            factor_idx = date_idx - 1
            selected, scores = select_stocks(
                factor=factor[:, factor_idx],
                universe=universe[:, factor_idx],
                trade_status=trade_status[:, date_idx],
                trade_price=trade_prices[:, date_idx],
                tickers=data.tickers,
                select_n=sim_cfg.select_n,
                factor_direction=factor_direction,
            )
            weights = build_target_weights(
                selected=selected,
                scores=scores,
                tickers=data.tickers,
                date_idx=date_idx,
                market_cap=market_cap,
                weight_method=sim_cfg.weight_method,
                factor_direction=factor_direction,
            )
            for rank, stock in enumerate(selected, start=1):
                selections.append(
                    {
                        "period_key": period_key,
                        "date": date,
                        "factor_date": window.dates[factor_idx],
                        "rank": rank,
                        "stock_code": stock,
                        "score": scores.get(stock, np.nan),
                        "target_weight": weights.get(stock, np.nan),
                    }
                )
            trade_dict = vector_to_price_dict(data.tickers, trade_prices[:, date_idx])
            limit_dict = vector_to_limit_dict(data.tickers, limit_status[:, date_idx])
            turnover = simulator.adjust_to_target_weights(
                target_weights=weights,
                trade_prices=trade_dict,
                close_prices=close_dict,
                limit_status=limit_dict,
                trade_date=date,
                scores=scores,
                factor_direction=factor_direction,
                slippage=sim_cfg.slippage,
            )
        simulator.mark_to_market(date, close_dict, turnover=turnover)

    daily = pd.DataFrame(simulator.daily_records)
    if not daily.empty:
        daily.insert(0, "period_key", period_key)
        daily.index = pd.to_datetime(daily["date"], format="%Y%m%d")
    trades = pd.DataFrame(simulator.trade_records)
    if not trades.empty:
        trades.insert(0, "period_key", period_key)
    selections_df = pd.DataFrame(selections)
    return simulator, daily, trades, selections_df


def build_ensemble_nav(
    date_index: pd.DatetimeIndex,
    daily_frames: list[pd.DataFrame],
    benchmark_nav: pd.Series,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine offset sleeves into one account-level NAV and a per-offset NAV table."""
    nav_df = pd.DataFrame(index=date_index)
    offset_nav_df = pd.DataFrame(index=date_index)
    total_equity = pd.Series(0.0, index=date_index)
    cash = pd.Series(0.0, index=date_index)
    position_value = pd.Series(0.0, index=date_index)
    turnover_value = pd.Series(0.0, index=date_index)

    for daily in daily_frames:
        if daily.empty:
            continue
        period_key = str(daily["period_key"].iloc[0])
        aligned = daily.reindex(date_index)
        equity = pd.to_numeric(aligned["total_equity"], errors="coerce").ffill().fillna(0.0)
        total_equity += equity
        cash += pd.to_numeric(aligned["cash"], errors="coerce").ffill().fillna(0.0)
        position_value += pd.to_numeric(aligned["position_value"], errors="coerce").ffill().fillna(0.0)
        turnover = pd.to_numeric(aligned["turnover"], errors="coerce").fillna(0.0)
        turnover_value += turnover * equity
        offset_nav_df[period_key] = pd.to_numeric(aligned["nav"], errors="coerce").ffill()

    nav_df["strategy_nav"] = total_equity / max(initial_capital, 1.0)
    nav_df["cash"] = cash
    nav_df["position_value"] = position_value
    nav_df["turnover"] = np.divide(
        turnover_value.to_numpy(dtype=float),
        np.maximum(total_equity.to_numpy(dtype=float), 1.0),
    )
    if benchmark_nav is not None and not benchmark_nav.empty:
        offset_nav_df["benchmark_nav"] = benchmark_nav.reindex(offset_nav_df.index).ffill()
    return nav_df, offset_nav_df


def _write_csv_safely(df: pd.DataFrame, path: Path, index: bool = True) -> bool:
    """写 CSV；Windows 文件被占用时不中断整个批处理。"""
    try:
        df.to_csv(path, index=index, encoding="utf-8-sig")
        return True
    except PermissionError:
        logger.warning(f"[step2] 文件被占用，跳过覆盖：{path}")
        return False


def load_industry_name_map(config: BacktestConfig) -> dict[int, str]:
    """读取申万一级行业代码名称表；缺失时用代码兜底。"""
    path = config.paths.data_fund_path / "sw1.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gb18030")
    if df.empty or len(df.columns) < 2:
        return {}
    code_col, name_col = df.columns[:2]
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        code = pd.to_numeric(pd.Series([row[code_col]]), errors="coerce").iloc[0]
        if pd.notna(code):
            out[int(code)] = str(row[name_col])
    return out


def build_selection_profile(
    selections_df: pd.DataFrame,
    tickers: list[str],
    dates: list[str],
    industry: np.ndarray | None,
    industry_names: dict[int, str],
    market_cap: np.ndarray | None,
) -> pd.DataFrame:
    """统计入选股票的行业和市值分布；按目标权重聚合为组合暴露百分比。"""
    if selections_df.empty:
        return pd.DataFrame(columns=["category", "segment", "weight_pct", "sample_count", "sample_pct"])

    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    stocks = normalize_selection_stock_series(selections_df.get("stock_code", pd.Series(dtype=object)))
    selection_dates = normalize_selection_date_series(selections_df.get("date", pd.Series(dtype=object)))
    stock_idx = stocks.map(ticker_to_idx)
    date_idx = selection_dates.map(date_to_idx)
    valid = stock_idx.notna() & date_idx.notna()
    if not valid.any():
        return pd.DataFrame(columns=["category", "segment", "weight_pct", "sample_count", "sample_pct"])

    stock_idx_arr = stock_idx[valid].astype(int).to_numpy()
    date_idx_arr = date_idx[valid].astype(int).to_numpy()
    weight = pd.to_numeric(selections_df.loc[valid, "target_weight"], errors="coerce").astype(float)
    weight = weight.where(weight > 0, 1.0).fillna(1.0).to_numpy()

    industry_segment = np.full(len(stock_idx_arr), "Unknown", dtype=object)
    if industry is not None:
        codes = industry[stock_idx_arr, date_idx_arr]
        finite = np.isfinite(codes)
        code_int = np.zeros(len(codes), dtype=int)
        code_int[finite] = codes[finite].astype(int)
        for code in np.unique(code_int[finite]):
            industry_segment[finite & (code_int == code)] = industry_names.get(int(code), str(int(code)))

    cap_segment = np.full(len(stock_idx_arr), "Unknown", dtype=object)
    if market_cap is not None:
        cap_values = market_cap[stock_idx_arr, date_idx_arr]
        finite_cap = np.isfinite(cap_values) & (cap_values > 0)
        cap_segment[finite_cap & (cap_values < 5e9)] = "<50亿"
        cap_segment[finite_cap & (cap_values >= 5e9) & (cap_values < 1e10)] = "50-100亿"
        cap_segment[finite_cap & (cap_values >= 1e10) & (cap_values < 2e10)] = "100-200亿"
        cap_segment[finite_cap & (cap_values >= 2e10) & (cap_values < 5e10)] = "200-500亿"
        cap_segment[finite_cap & (cap_values >= 5e10) & (cap_values < 1e11)] = "500-1000亿"
        cap_segment[finite_cap & (cap_values >= 1e11)] = ">=1000亿"

    raw = pd.DataFrame({"industry": industry_segment, "market_cap": cap_segment, "weight": weight})
    out_rows: list[dict] = []
    for category, col in [("industry", "industry"), ("market_cap", "market_cap")]:
        total_weight = raw["weight"].sum()
        total_count = len(raw)
        grouped = raw.groupby(col, dropna=False).agg(weight=("weight", "sum"), sample_count=("weight", "size"))
        grouped = grouped.sort_values("weight", ascending=False)
        for segment, part in grouped.iterrows():
            out_rows.append(
                {
                    "category": category,
                    "segment": str(segment),
                    "weight_pct": float(part["weight"] / total_weight) if total_weight > 0 else np.nan,
                    "sample_count": int(part["sample_count"]),
                    "sample_pct": float(part["sample_count"] / total_count) if total_count > 0 else np.nan,
                }
            )
    return pd.DataFrame(out_rows)


def normalize_selection_stock(value) -> str:
    """把 selections.csv 里可能被读成数字的股票代码恢复为 6 位字符串。"""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{int(numeric):06d}"
    text = str(value).strip().upper()
    if len(text) == 8 and text[:2] in {"SH", "SZ", "BJ"}:
        text = text[2:]
    return text.zfill(6) if text.isdigit() else text


def normalize_selection_stock_series(values: pd.Series) -> pd.Series:
    """向量化恢复 6 位股票代码。"""
    numeric = pd.to_numeric(values, errors="coerce")
    out = values.astype(str).str.strip().str.upper()
    numeric_mask = numeric.notna()
    out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(int).map(lambda value: f"{value:06d}")
    prefixed = out.str.len().eq(8) & out.str[:2].isin(["SH", "SZ", "BJ"])
    out.loc[prefixed] = out.loc[prefixed].str[2:]
    digit_mask = out.str.isdigit()
    out.loc[digit_mask] = out.loc[digit_mask].str.zfill(6)
    return out


def normalize_selection_date(value) -> str:
    """把 selections.csv 里可能被读成数字或日期的字段统一成 YYYYMMDD。"""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{int(numeric):08d}"
    return str(value).strip().replace("-", "").replace("/", "")


def normalize_selection_date_series(values: pd.Series) -> pd.Series:
    """向量化恢复 YYYYMMDD 日期字符串。"""
    numeric = pd.to_numeric(values, errors="coerce")
    out = values.astype(str).str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    numeric_mask = numeric.notna()
    out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(int).map(lambda value: f"{value:08d}")
    return out


def market_cap_bucket(value: float) -> str:
    """按常见 A 股市值分桶，输入单位为元。"""
    yi = value / 1e8
    if yi < 50:
        return "<50亿"
    if yi < 100:
        return "50-100亿"
    if yi < 200:
        return "100-200亿"
    if yi < 500:
        return "200-500亿"
    if yi < 1000:
        return "500-1000亿"
    return ">=1000亿"


def should_rebalance(date_idx: int, freq: int) -> bool:
    """判断当前日期是否为换仓日；第 0 天没有前一日因子，因此不换仓。"""
    if date_idx <= 0:
        return False
    return (date_idx - 1) % freq == 0


def select_stocks(
    factor: np.ndarray,
    universe: np.ndarray,
    trade_status: np.ndarray,
    trade_price: np.ndarray,
    tickers: list[str],
    select_n: float,
    factor_direction: bool,
) -> tuple[list[str], dict[str, float]]:
    """根据当日可交易股票和因子方向生成目标股票列表及对应因子分数。"""
    valid = np.isfinite(factor) & universe & (trade_status == 1) & np.isfinite(trade_price) & (trade_price > 0)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return [], {}
    order = np.argsort(factor[valid_idx], kind="mergesort")
    if not factor_direction:
        order = order[::-1]
    ordered_idx = valid_idx[order]
    count = resolve_selection_count(select_n, len(ordered_idx))
    selected_idx = ordered_idx[:count]
    selected = [tickers[i] for i in selected_idx]
    scores = {tickers[i]: float(factor[i]) for i in selected_idx}
    return selected, scores


def resolve_selection_count(select_n: float, valid_count: int) -> int:
    """解析 select_n：-1 表示全选，0~1 表示比例，>=1 表示固定只数。"""
    if valid_count <= 0:
        return 0
    if select_n == -1:
        return valid_count
    if 0 < select_n < 1:
        return max(1, min(valid_count, int(np.floor(valid_count * select_n))))
    if select_n >= 1:
        return max(1, min(valid_count, int(select_n)))
    raise ValueError("simulation.select_n must be -1, a positive ratio, or a positive count")


def build_target_weights(
    selected: list[str],
    scores: dict[str, float],
    tickers: list[str],
    date_idx: int,
    market_cap: np.ndarray | None,
    weight_method: str,
    factor_direction: bool,
) -> dict[str, float]:
    """按 simulation.weight_method 生成目标持仓权重，供真实账户模拟器下单使用。"""
    if not selected:
        return {}

    method = str(weight_method).strip().lower()
    if method == "equal":
        return equal_target_weights(selected)

    if method == "market":
        return market_target_weights(selected, tickers, date_idx, market_cap)

    if method == "factor_softmax":
        return factor_softmax_target_weights(selected, scores, factor_direction)

    raise ValueError(f"unsupported weight_method: {weight_method}")


def equal_target_weights(selected: list[str]) -> dict[str, float]:
    """等权目标组合：所有入选股票的目标权重相同。"""
    if not selected:
        return {}
    weight = 1.0 / len(selected)
    return {stock: weight for stock in selected}


def market_target_weights(
    selected: list[str],
    tickers: list[str],
    date_idx: int,
    market_cap: np.ndarray | None,
) -> dict[str, float]:
    """市值加权目标组合：按调仓日总市值归一化，市值不可用时回退为等权。"""
    if market_cap is None:
        return equal_target_weights(selected)

    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    values = np.array(
        [
            market_cap[ticker_to_idx[stock], date_idx] if stock in ticker_to_idx else np.nan
            for stock in selected
        ],
        dtype=float,
    )
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return equal_target_weights(selected)

    weights = np.zeros(len(selected), dtype=float)
    weights[valid] = values[valid] / values[valid].sum()
    return {stock: float(weight) for stock, weight in zip(selected, weights)}


def factor_softmax_target_weights(
    selected: list[str],
    scores: dict[str, float],
    factor_direction: bool,
) -> dict[str, float]:
    """因子 softmax 目标组合：先按多头方向调整分数，再 softmax 成目标权重。"""
    raw_scores = np.array([scores.get(stock, np.nan) for stock in selected], dtype=float)
    # factor_direction=True 表示 RankIC 为负、因子值越小越好，因此取负数后再 softmax。
    directional_scores = -raw_scores if factor_direction else raw_scores
    weights = stable_softmax(directional_scores)
    if np.isfinite(weights).all() and weights.sum() > 0:
        return {stock: float(weight) for stock, weight in zip(selected, weights)}
    return equal_target_weights(selected)


def load_step1_rankic_direction(config: BacktestConfig, factor_name: str) -> tuple[bool, float]:
    """从 Step1 统计结果读取 TOTAL RankIC 均值，并据此自动决定 Step2 多头方向。"""
    stats_path = config.paths.output_folder / "step1_factor_analysis" / factor_name / "stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"找不到因子 {factor_name} 的 Step1 统计文件：{stats_path}。"
            f"请先运行：python step1_因子分析.py --config {config.config_path}"
        )

    stats_df = pd.read_csv(stats_path)
    required_cols = {"period", "series", "mean"}
    if not required_cols.issubset(stats_df.columns):
        raise ValueError(f"Step1 统计文件缺少必要列 {required_cols}：{stats_path}")

    mask = (stats_df["period"].astype(str) == "TOTAL") & (stats_df["series"].astype(str) == "RankIC")
    if not mask.any():
        raise ValueError(
            f"Step1 统计文件中找不到 TOTAL RankIC 行：{stats_path}。"
            f"请重新运行对应因子的 Step1。"
        )

    rankic_mean = pd.to_numeric(stats_df.loc[mask, "mean"], errors="coerce").dropna()
    if rankic_mean.empty:
        raise ValueError(
            f"Step1 统计文件中的 TOTAL RankIC mean 无法解析：{stats_path}。"
            f"请重新运行对应因子的 Step1。"
        )

    value = float(rankic_mean.iloc[0])
    # RankIC 均值 >= 0 说明因子值越大未来收益越高，Step2 选大；小于 0 则选小。
    return value < 0, value


def maybe_write_quantstats(config: BacktestConfig, factor_name: str, nav_df: pd.DataFrame, factor_out: Path) -> None:
    """调用项目内置 quantstats 生成增强报告，不再使用外部安装的同名库。"""
    if not config.simulation.enable_quantstats:
        return
    try:
        qs = import_local_quantstats()
    except Exception as exc:
        logger.warning(f"[step2] 本地 quantstats 不可用，跳过 {factor_name} 的增强报告：{exc}")
        return
    if "strategy_nav" not in nav_df or "benchmark_nav" not in nav_df:
        return
    returns = nav_df["strategy_nav"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    benchmark = nav_df["benchmark_nav"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_title = benchmark_display_name(config.simulation.benchmark)
    benchmark.name = benchmark_title
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    try:
        qs.reports.html(
            aligned.iloc[:, 0],
            benchmark=aligned.iloc[:, 1],
            output=str(factor_out / "quantstats_report.html"),
            title=f"{factor_name} QuantStats Report",
            benchmark_title=benchmark_title,
        )
    except Exception as exc:
        logger.warning(f"[step2] 本地 quantstats 增强报告生成失败，因子 {factor_name}：{exc}")


def benchmark_display_name(code: str | int | float | None) -> str:
    """把配置中的指数代码显示成更可读的 benchmark 名称；未知代码保留原代码。"""
    text = "" if code is None else str(code).strip()
    normalized = text.split(".")[0].upper()
    names = {
        "000001": "000001 上证指数",
        "000016": "000016 上证50",
        "000300": "000300 沪深300",
        "000905": "000905 中证500",
        "000906": "000906 中证800",
        "000852": "000852 中证1000",
        "000985": "000985 中证全指",
        "000986": "000986 中证全指证券公司",
        "000987": "000987 中证全指能源",
        "399001": "399001 深证成指",
        "399006": "399006 创业板指",
        "399300": "399300 沪深300",
        "399905": "399905 中证500",
        "399906": "399906 中证800",
        "399852": "399852 中证1000",
        "881001": "881001 万得全A",
    }
    return names.get(normalized, text or "基准")


def import_local_quantstats():
    """只允许导入项目目录下的 quantstats，防止误用 pip/conda 环境中的外部库。"""
    import importlib
    import sys

    project_root = Path(PROJECT_ROOT).resolve()
    local_pkg_dir = project_root / "quantstats"
    if not local_pkg_dir.exists():
        raise FileNotFoundError(f"项目内 quantstats 目录不存在：{local_pkg_dir}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 如果当前进程已经误加载外部 quantstats，先移除缓存，确保后续导入命中本项目目录。
    loaded_qs = sys.modules.get("quantstats")
    if loaded_qs is not None:
        loaded_file = Path(getattr(loaded_qs, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(local_pkg_dir):
            for module_name in list(sys.modules):
                if module_name == "quantstats" or module_name.startswith("quantstats."):
                    del sys.modules[module_name]

    qs = importlib.import_module("quantstats")
    qs_file = Path(qs.__file__).resolve()
    if not qs_file.is_relative_to(local_pkg_dir):
        raise ImportError(f"当前导入的 quantstats 不是项目内版本：{qs_file}")
    logger.debug(f"[step2] 使用项目内本地 quantstats：{qs_file}")
    return qs


def build_summary_row(result: SimulationResult) -> dict:
    """整理单因子的模拟回测摘要行，供 step2 summary.csv/html 使用。"""
    metrics = dict(zip(result.metrics["metric"], result.metrics["value"]))
    row = {"factor": result.factor_name, "html": str(result.html_path)}
    for key in ["total_return", "annual_return", "sharpe", "max_drawdown", "excess_return", "avg_turnover", "trade_count"]:
        if key in metrics:
            row[key] = metrics[key]
    return row
