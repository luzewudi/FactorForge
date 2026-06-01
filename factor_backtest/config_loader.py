# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

import yaml


def _as_bool(value: Any, default: bool = False) -> bool:
    """把 YAML 中常见的布尔写法统一转成 bool，避免字符串 true/false 被误判。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_date(value: Any, label: str) -> str:
    """把日期配置统一成 YYYYMMDD 字符串，作为后续交易日切片的标准口径。"""
    if value is None:
        raise ValueError(f"{label} is required")
    text = str(value).strip().replace("-", "").replace("/", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{label} must use YYYYMMDD or YYYY-MM-DD format")
    return text


def normalize_weight_method(raw: Any, default: str = "equal") -> str:
    """统一加权方式配置，兼容旧 mkt_weight，并把常见别名归一到三个正式取值。"""
    if raw is None:
        return default
    text = str(raw).strip().lower()
    aliases = {
        "equal": "equal",
        "equal_weight": "equal",
        "eq": "equal",
        "market": "market",
        "mkt": "market",
        "market_cap": "market",
        "mkt_weight": "market",
        "factor": "factor_softmax",
        "factor_weight": "factor_softmax",
        "factor_softmax": "factor_softmax",
        "softmax": "factor_softmax",
    }
    return aliases.get(text, text)


@dataclass
class PathConfig:
    eod_path: Path
    fund_path: Path
    data_fund_path: Path
    label_path: Path
    factor_folder: Path
    output_folder: Path
    period_path: Path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PathConfig":
        """读取 paths 模块，并把所有路径转换为 Path 对象，便于后续统一拼接。"""
        required = [
            "eod_path",
            "fund_path",
            "data_fund_path",
            "label_path",
            "factor_folder",
            "output_folder",
        ]
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ValueError(f"paths missing required keys: {', '.join(missing)}")
        values = {key: Path(raw[key]) for key in required}
        values["period_path"] = Path(raw.get("period_path") or (values["eod_path"].parent / "period"))
        return cls(**values)


@dataclass
class TStatConfig:
    window: int = 252
    min_periods: int = 21

    @classmethod
    def from_value(cls, raw: Any) -> "TStatConfig":
        """读取 IC 滚动 t 值配置；允许直接写窗口数字，也允许写 window/min_periods 字典。"""
        if raw is None:
            return cls()
        if isinstance(raw, dict):
            return cls(
                window=int(raw.get("window", 252)),
                min_periods=int(raw.get("min_periods", 21)),
            )
        if isinstance(raw, (int, float)):
            return cls(window=int(raw), min_periods=21)
        return cls()


@dataclass
class AnalysisConfig:
    start_date: str
    end_date: str
    factors: str | List[str] = "all"
    turnover: int = 1
    period: str = "1"
    offsets: str | List[str] = "all"
    offset_mode: str = "ensemble"
    label_days: str | int = 1
    per_divide_num: int = 5
    universe: str = "0"
    benchmark: str = "000852"
    trade_price: str = "vwap"
    weight_method: str = "equal"
    neutralization: str = "none"
    ic_tstat: TStatConfig = field(default_factory=TStatConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], default_factors: str | List[str] = "all") -> "AnalysisConfig":
        """读取 analysis 模块，定义 Step1 因子研究需要的日期、分组、基准和中性化口径。"""
        period = normalize_period(raw.get("period", raw.get("turnover", 1)))
        label_days = normalize_label_period(raw.get("label_days", raw.get("turnover", period_to_label_days(period))))
        return cls(
            start_date=normalize_date(raw.get("start_date"), "analysis.start_date"),
            end_date=normalize_date(raw.get("end_date"), "analysis.end_date"),
            factors=normalize_factors(raw.get("factors", default_factors)),
            turnover=int(raw.get("turnover", label_period_to_days(label_days))),
            period=period,
            offsets=normalize_offsets(raw.get("offsets", "all")),
            offset_mode=str(raw.get("offset_mode", "ensemble")).strip().lower(),
            label_days=label_days,
            per_divide_num=int(raw.get("per_divide_num", 5)),
            universe=str(raw.get("universe", "0")),
            benchmark=str(raw.get("benchmark", "000852")),
            trade_price=str(raw.get("trade_price", "vwap")).lower(),
            weight_method=read_weight_method(raw, "analysis"),
            neutralization=str(raw.get("neutralization", "none")).lower(),
            ic_tstat=TStatConfig.from_value(raw.get("ic_tstat")),
        )


@dataclass
class SimulationConfig:
    factors: str | List[str] = "all"
    universe: str = "0"
    weight_method: str = "equal"
    select_n: float = 100
    fee: float = 0.0003
    stamp_duty: float = 0.001
    initial_capital: float = 10_000_000.0
    rebalance_freq_days: int = 1
    period: str = "1"
    offsets: str | List[str] = "all"
    offset_mode: str = "ensemble"
    trade_price: str = "vwap"
    benchmark: str = "000852"
    slippage: float = 0.0
    enable_quantstats: bool = True

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        analysis: AnalysisConfig,
        default_factors: str | List[str] = "all",
    ) -> "SimulationConfig":
        """读取 simulation 模块，定义 Step2 真实账户模拟的选股、费用、滑点和调仓口径。"""
        period = normalize_period(raw.get("period", raw.get("rebalance_freq_days", 1)))
        rebalance_freq_days = int(raw.get("rebalance_freq_days", period_to_label_days(period)))
        return cls(
            factors=normalize_factors(raw.get("factors", default_factors)),
            universe=str(raw.get("universe", analysis.universe)),
            weight_method=read_weight_method(raw, "simulation"),
            select_n=float(raw.get("select_n", 100)),
            fee=float(raw.get("fee", 0.0003)),
            stamp_duty=float(raw.get("stamp_duty", 0.001)),
            initial_capital=float(raw.get("initial_capital", 10_000_000.0)),
            rebalance_freq_days=rebalance_freq_days,
            period=period,
            offsets=normalize_offsets(raw.get("offsets", "all")),
            offset_mode=str(raw.get("offset_mode", "ensemble")).strip().lower(),
            trade_price=str(raw.get("trade_price", analysis.trade_price)).lower(),
            benchmark=str(raw.get("benchmark", analysis.benchmark)),
            slippage=float(raw.get("slippage", 0.0)),
            enable_quantstats=_as_bool(raw.get("enable_quantstats"), True),
        )


@dataclass
class BacktestConfig:
    paths: PathConfig
    factors: str | List[str]
    analysis: AnalysisConfig
    simulation: SimulationConfig
    config_path: Path

    @classmethod
    def load(cls, config_path: str | Path) -> "BacktestConfig":
        """加载唯一 YAML 配置，并拆分成 paths/factors/analysis/simulation 四个模块。"""
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("config yaml must contain a mapping")

        paths = PathConfig.from_dict(raw.get("paths", {}) or {})
        # 兼容旧版顶层 factors；新版推荐分别写 analysis.factors 和 simulation.factors。
        factors = normalize_factors(raw.get("factors", "all"))
        analysis = AnalysisConfig.from_dict(raw.get("analysis", {}) or {}, factors)
        simulation = SimulationConfig.from_dict(raw.get("simulation", {}) or {}, analysis, analysis.factors)

        validate_config(paths, analysis, simulation)
        return cls(paths=paths, factors=factors, analysis=analysis, simulation=simulation, config_path=path)


def normalize_factors(raw: Any) -> str | List[str]:
    """解析 factors 配置；支持 all，也支持单个因子名或因子名列表。"""
    if raw is None:
        return "all"
    if isinstance(raw, str):
        text = raw.strip()
        return "all" if text.lower() == "all" else [strip_npy_suffix(text)]
    if isinstance(raw, Sequence):
        values = [strip_npy_suffix(str(item).strip()) for item in raw if str(item).strip()]
        return values or "all"
    raise ValueError("factors must be 'all' or a list of factor names")


def normalize_period(raw: Any) -> str:
    """把 period 配置标准化成周期名前缀，例如 5、20、W、2W、M。"""
    if raw is None:
        return "1"
    if isinstance(raw, (int, float)) and float(raw).is_integer():
        return str(int(raw))
    text = str(raw).strip()
    if not text:
        return "1"
    if text.replace(".", "", 1).isdigit() and float(text).is_integer():
        return str(int(float(text)))
    return text.upper()


def normalize_offsets(raw: Any) -> str | List[str]:
    """解析 offsets 配置；all 表示使用该 period 下全部 offset。"""
    if raw is None:
        return "all"
    if isinstance(raw, str):
        text = raw.strip()
        return "all" if text.lower() == "all" else [normalize_offset_label(text)]
    if isinstance(raw, Sequence):
        values = [normalize_offset_label(item) for item in raw if str(item).strip()]
        return values or "all"
    return [normalize_offset_label(raw)]


def normalize_offset_label(raw: Any) -> str:
    """把 offset 配置标准化成 period 文件里的后缀。"""
    if isinstance(raw, (int, float)) and float(raw).is_integer():
        return str(int(raw))
    text = str(raw).strip()
    if text.replace("-", "", 1).replace(".", "", 1).isdigit() and float(text).is_integer():
        return str(int(float(text)))
    return text.upper()


def period_to_label_days(period: str) -> int:
    """非日频周期没有天然 label 文件名时，提供常用交易日近似。"""
    text = normalize_period(period)
    if text.isdigit():
        return int(text)
    aliases = {
        "W": 5,
        "2W": 10,
        "3W": 15,
        "4W": 20,
        "5W": 25,
        "6W": 30,
        "M": 20,
        "W53": 5,
    }
    return aliases.get(text, 1)


def normalize_label_period(raw: Any) -> str | int:
    """解析 label_days，兼容 5/vwap_5d.npy 和 W/vwap_W.npy 两类标签文件。"""
    if raw is None:
        return 1
    if isinstance(raw, (int, float)) and float(raw).is_integer():
        return int(raw)
    text = str(raw).strip()
    if not text:
        return 1
    if text.replace(".", "", 1).isdigit() and float(text).is_integer():
        return int(float(text))
    return normalize_period(text)


def label_period_to_days(label_period: str | int) -> int:
    """把 label 周期转换为年化近似交易日数；文件名解析仍保留原 label_period。"""
    if isinstance(label_period, int):
        return label_period
    text = normalize_period(label_period)
    if text.isdigit():
        return int(text)
    return period_to_label_days(text)


def strip_npy_suffix(name: str) -> str:
    """去掉因子名末尾的 .npy，保证 YAML 中写不写后缀都能找到同一个文件。"""
    return name[:-4] if name.lower().endswith(".npy") else name


def read_weight_method(raw: dict[str, Any], section: str) -> str:
    """读取新的 weight_method；若旧配置仍写 mkt_weight，则自动兼容成 equal/market。"""
    if "weight_method" in raw:
        return normalize_weight_method(raw.get("weight_method"))
    if "mkt_weight" in raw:
        return "market" if _as_bool(raw.get("mkt_weight"), False) else "equal"
    return "equal"


def validate_config(paths: PathConfig, analysis: AnalysisConfig, simulation: SimulationConfig) -> None:
    """在正式运行前检查路径、日期、交易价格和核心参数，尽早暴露配置错误。"""
    for label, path in [
        ("eod_path", paths.eod_path),
        ("fund_path", paths.fund_path),
        ("data_fund_path", paths.data_fund_path),
        ("label_path", paths.label_path),
        ("factor_folder", paths.factor_folder),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"paths.{label} does not exist: {path}")

    if analysis.start_date > analysis.end_date:
        raise ValueError("analysis.start_date must be <= analysis.end_date")
    if analysis.turnover <= 0:
        raise ValueError("analysis.turnover must be positive")
    if label_period_to_days(analysis.label_days) <= 0:
        raise ValueError("analysis.label_days must be positive")
    if analysis.offset_mode != "ensemble":
        raise ValueError("analysis.offset_mode currently supports only ensemble")
    if analysis.per_divide_num <= 1:
        raise ValueError("analysis.per_divide_num must be > 1")
    if analysis.trade_price not in {"open", "close", "vwap", "periodvwap", "period_vwap"}:
        raise ValueError("analysis.trade_price must be open, close, vwap, or periodvwap")
    if analysis.weight_method not in {"equal", "market", "factor_softmax"}:
        raise ValueError("analysis.weight_method must be equal, market, or factor_softmax")
    if analysis.neutralization not in {"none", "market", "industry_market"}:
        raise ValueError("analysis.neutralization must be none, market, or industry_market")
    if simulation.rebalance_freq_days <= 0:
        raise ValueError("simulation.rebalance_freq_days must be positive")
    if simulation.offset_mode != "ensemble":
        raise ValueError("simulation.offset_mode currently supports only ensemble")
    if simulation.trade_price not in {"open", "close", "vwap", "periodvwap", "period_vwap"}:
        raise ValueError("simulation.trade_price must be open, close, vwap, or periodvwap")
    if simulation.weight_method not in {"equal", "market", "factor_softmax"}:
        raise ValueError("simulation.weight_method must be equal, market, or factor_softmax")


def resolve_factor_files(config: BacktestConfig, stage: str = "analysis") -> list[Path]:
    """根据 analysis/simulation 各自的 factors 配置展开因子文件列表。"""
    folder = config.paths.factor_folder
    factor_spec = config.analysis.factors if stage == "analysis" else config.simulation.factors
    if factor_spec == "all":
        files = []
        for path in sorted(folder.glob("*.npy")):
            stem = path.stem
            if stem in {"dates", "ticker_names"}:
                continue
            if stem.endswith("_dates") or stem.endswith("_tickers"):
                continue
            files.append(path)
        if not files:
            raise FileNotFoundError(f"no factor .npy files found in {folder}")
        return files

    files = []
    for name in factor_spec:
        path = folder / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"factor file not found: {path}")
        files.append(path)
    return files
