#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从服务器 datahub 读取日频数据，并生成 GRU 回测可直接使用的 pickle 文件。

目标输出格式：
1) *_close.pkl: pandas.DataFrame(index=交易日, columns=股票代码)
2) *_factors.pkl: dict，至少包含键:
   - factors: Dict[str, DataFrame]
   - metadata: Dict[str, Any]

默认输出到 config.json 中当前回测路径：
- /home/intership/guojh/GA和GRU/stock_factors_backtest_latest_close.pkl
- /home/intership/guojh/GA和GRU/stock_factors_backtest_latest_factors.pkl
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_CLOSE = "/home/intership/guojh/GA和GRU/stock_factors_backtest_latest_close.pkl"
DEFAULT_OUTPUT_FACTORS = "/home/intership/guojh/GA和GRU/stock_factors_backtest_latest_factors.pkl"
MIN_OPTIONAL_FACTOR_NON_NULL_RATIO = 0.60
MIN_OPTIONAL_FACTOR_EFFECTIVE_STD_DAYS = 5
DEFAULT_UNIVERSE_FALLBACK_DAYS = 10

UNIVERSE_FIELD_CANDIDATES = {
    "hs300": ["HS300_data"],
    "zz500": ["ZZ500_data"],
    "zz1000": ["ZZ1000_data", "CSI1000_data"],
}

BENCHMARK_FIELD_CANDIDATES = {
    "hs300": ["close_000300SH"],
    "zz500": ["close_000905SH"],
}

# AShareMoneyFlow.xlsx 中维护的资金流字段代码（按 DataHub 字段名直连）
MONEYFLOW_DATAHUB_FIELDS = [
    "BUY_VALUE_EXLARGE_ORDER",
    "SELL_VALUE_EXLARGE_ORDER",
    "BUY_VALUE_LARGE_ORDER",
    "SELL_VALUE_LARGE_ORDER",
    "BUY_VALUE_MED_ORDER",
    "SELL_VALUE_MED_ORDER",
    "BUY_VALUE_SMALL_ORDER",
    "SELL_VALUE_SMALL_ORDER",
    "BUY_VOLUME_EXLARGE_ORDER",
    "SELL_VOLUME_EXLARGE_ORDER",
    "BUY_VOLUME_LARGE_ORDER",
    "SELL_VOLUME_LARGE_ORDER",
    "BUY_VOLUME_MED_ORDER",
    "SELL_VOLUME_MED_ORDER",
    "BUY_VOLUME_SMALL_ORDER",
    "SELL_VOLUME_SMALL_ORDER",
    "BUY_TRADES_EXLARGE_ORDER",
    "SELL_TRADES_EXLARGE_ORDER",
    "BUY_TRADES_LARGE_ORDER",
    "SELL_TRADES_LARGE_ORDER",
    "BUY_TRADES_MED_ORDER",
    "SELL_TRADES_MED_ORDER",
    "BUY_TRADES_SMALL_ORDER",
    "SELL_TRADES_SMALL_ORDER",
    "VOLUME_DIFF_SMALL_TRADER",
    "VOLUME_DIFF_SMALL_TRADER_ACT",
    "VOLUME_DIFF_MED_TRADER",
    "VOLUME_DIFF_MED_TRADER_ACT",
    "VOLUME_DIFF_LARGE_TRADER",
    "VOLUME_DIFF_LARGE_TRADER_ACT",
    "VOLUME_DIFF_INSTITUTE",
    "VOLUME_DIFF_INSTITUTE_ACT",
    "VALUE_DIFF_SMALL_TRADER",
    "VALUE_DIFF_SMALL_TRADER_ACT",
    "VALUE_DIFF_MED_TRADER",
    "VALUE_DIFF_MED_TRADER_ACT",
    "VALUE_DIFF_LARGE_TRADER",
    "VALUE_DIFF_LARGE_TRADER_ACT",
    "VALUE_DIFF_INSTITUTE",
    "VALUE_DIFF_INSTITUTE_ACT",
    "S_MFD_INFLOWVOLUME",
    "NET_INFLOW_RATE_VOLUME",
    "S_MFD_INFLOW_OPENVOLUME",
    "S_MFD_INFLOW_CLOSEVOLUME",
    "S_MFD_INFLOW",
    "NET_INFLOW_RATE_VALUE",
    "S_MFD_INFLOW_OPEN",
    "S_MFD_INFLOW_CLOSE",
    "MONEYFLOW_PCT_VOLUME",
    "OPEN_MONEYFLOW_PCT_VOLUME",
    "CLOSE_MONEYFLOW_PCT_VOLUME",
    "S_MFD_INFLOWVOLUME_LARGE_ORDER",
    "NET_INFLOW_RATE_VOLUME_L",
    "S_MFD_INFLOW_LARGE_ORDER",
    "MONEYFLOW_PCT_VOLUME_L",
    "S_MFD_INFLOW_OPENVOLUME_L",
    "S_MFD_INFLOW_OPEN_LARGE_ORDER",
    "OPEN_MONEYFLOW_PCT_VALUE_L",
    "S_MFD_INFLOW_CLOSEVOLUME_L",
    "S_MFD_INFLOW_CLOSE_LARGE_ORDER",
    "CLOSE_MONEYFLOW_PCT_VOLUME_L",
    "BUY_VALUE_EXLARGE_ORDER_ACT",
    "SELL_VALUE_EXLARGE_ORDER_ACT",
    "BUY_VALUE_LARGE_ORDER_ACT",
    "SELL_VALUE_LARGE_ORDER_ACT",
    "BUY_VALUE_MED_ORDER_ACT",
    "SELL_VALUE_MED_ORDER_ACT",
    "BUY_VALUE_SMALL_ORDER_ACT",
    "SELL_VALUE_SMALL_ORDER_ACT",
    "BUY_VOLUME_EXLARGE_ORDER_ACT",
    "SELL_VOLUME_EXLARGE_ORDER_ACT",
    "BUY_VOLUME_LARGE_ORDER_ACT",
    "SELL_VOLUME_LARGE_ORDER_ACT",
    "BUY_VOLUME_MED_ORDER_ACT",
    "SELL_VOLUME_MED_ORDER_ACT",
    "BUY_VOLUME_SMALL_ORDER_ACT",
    "SELL_VOLUME_SMALL_ORDER_ACT",
]

# 这些是回测最关键字段（与现有管线约定一致）
CORE_FACTORS = [
    "close",
    "open",
    "high",
    "low",
    "pct_chg",
    "turnover_rate",
    "total_mv",
    "circ_mv",
]

# datahub 字段候选映射：target_factor -> [candidate_keys...]
DEFAULT_FIELD_CANDIDATES = {
    "close": ["close_adj", "close"],
    "open": ["open_adj", "open"],
    "high": ["high_adj", "high"],
    "low": ["low_adj", "low"],
    "pct_chg": ["pct_chg"],
    "turnover_rate": ["turn", "free_turn"],
    "total_mv": ["mkt_cap_ard"],
    "circ_mv": ["free_mkt_cap"],
    "vol": ["volume", "volume_by_share"],
    "amount": ["amt"],
    "free_turn": ["free_turn"],
    "turn": ["turn"],
    "S_MFD_INFLOW_LARGE_ORDER": ["S_MFD_INFLOW_LARGE_ORDER"],
    "S_MFD_INFLOW": ["S_MFD_INFLOW"],
    "volatility": ["volatility"],
    "s_val_pb_new": ["s_val_pb_new"],
    "roe_ttm": ["roe_ttm"],
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Build backtest pickle files from datahub_daily")
    parser.add_argument("--start", default="2024-10-25", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", default="2025-10-24", help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--cpu-num", type=int, default=64, help="datahub 并发参数")
    parser.add_argument("--datahub-path", default="/home/intership/public/datahub/", help="datahub 模块路径")
    parser.add_argument("--output-close", default=DEFAULT_OUTPUT_CLOSE, help="输出 close pickle 路径")
    parser.add_argument("--output-factors", default=DEFAULT_OUTPUT_FACTORS, help="输出 factors pickle 路径")
    parser.add_argument(
        "--fields",
        default="",
        help="可选：手工指定 datahub 字段（逗号分隔）。为空则按候选字段自动尝试。",
    )
    parser.add_argument(
        "--allow-missing-core",
        action="store_true",
        help="允许缺失核心字段（缺失时用 0 填充）；默认不允许。",
    )
    parser.add_argument(
        "--metadata-job",
        default="backtest_datahub_replace",
        help="写入 metadata.job 的标识",
    )
    parser.add_argument(
        "--universe",
        default="all_a",
        choices=["all_a", "hs300", "zz500", "zz500_zz1000"],
        help="股票池范围：all_a(默认)、hs300、zz500、zz500_zz1000",
    )
    parser.add_argument(
        "--universe-fallback-days",
        type=int,
        default=DEFAULT_UNIVERSE_FALLBACK_DAYS,
        help="指数成分权重字段缺失时，最多向前回退多少个自然日寻找最近可用日。仅影响股票池过滤，不影响价格/因子 end。",
    )
    parser.add_argument(
        "--minute-fields",
        default="",
        help="可选：分钟字段（逗号分隔，按 datahub_minute 读取并按日聚合后并入因子）。",
    )
    parser.add_argument(
        "--minute-agg",
        default="mean",
        choices=["mean", "last", "sum", "std"],
        help="分钟字段按日聚合方式。",
    )
    parser.add_argument(
        "--minute-start-time",
        default=None,
        help="分钟读取起始时刻，例如 0930。默认读取全天。",
    )
    parser.add_argument(
        "--minute-end-time",
        default=None,
        help="分钟读取结束时刻，例如 1500。默认读取全天。",
    )
    parser.add_argument(
        "--minute-prefix",
        default="minute_",
        help="分钟聚合因子名前缀。",
    )
    return parser.parse_args()


def _ensure_datahub_importable(datahub_path):
    if datahub_path and datahub_path not in sys.path:
        sys.path.append(datahub_path)
    try:
        read_datahub = importlib.import_module("read_datahub")
        datahub_daily = getattr(read_datahub, "datahub_daily")
        datahub_minute = getattr(read_datahub, "datahub_minute", None)
    except Exception as exc:
        raise RuntimeError(
            "无法导入 read_datahub.datahub_daily，请确认 --datahub-path 和运行环境（建议 pd1.0-torch）"
        ) from exc
    return datahub_daily, datahub_minute


def _get_serializer_module():
    """优先使用 dill；若未安装则回退到标准库 pickle。"""
    try:
        return importlib.import_module("dill")
    except Exception:
        import pickle

        print("[WARN] 未检测到 dill，已回退使用 pickle 序列化")
        return pickle


def _safe_to_numeric(df):
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_frame(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("期望 DataFrame")
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.columns = out.columns.astype(str)
    out = _safe_to_numeric(out)
    return out


def _normalize_benchmark_series(frame: pd.DataFrame, source_name: str) -> pd.Series:
    """将指数价格表统一整理为按日期索引的单列 Series。"""
    normalized = _normalize_frame(frame)
    if normalized.empty:
        raise ValueError(f"benchmark source {source_name} is empty")

    if source_name in normalized.columns:
        series = normalized[source_name]
    elif normalized.shape[1] == 1:
        series = normalized.iloc[:, 0]
    else:
        series = normalized.mean(axis=1)

    series = pd.to_numeric(series, errors="coerce")
    series.name = source_name
    return series.sort_index().ffill()


def _collect_universe_filter_fields(universe: str) -> set[str]:
    if universe == "hs300":
        return set(UNIVERSE_FIELD_CANDIDATES["hs300"])
    if universe == "zz500":
        return set(UNIVERSE_FIELD_CANDIDATES["zz500"])
    if universe == "zz500_zz1000":
        return set(UNIVERSE_FIELD_CANDIDATES["zz500"]) | set(UNIVERSE_FIELD_CANDIDATES["zz1000"])
    return set()


def _fallback_end_dates(end: str, fallback_days: int) -> list[str]:
    max_days = max(0, int(fallback_days or 0))
    if max_days == 0:
        return []
    end_ts = pd.Timestamp(end).normalize()
    return [str((end_ts - timedelta(days=offset)).date()) for offset in range(1, max_days + 1)]


def _fetch_fields(
    datahub_daily_fn,
    fields,
    start,
    end,
    cpu_num,
    universe_fields: set[str] | None = None,
    universe_fallback_days: int = 0,
):
    """逐字段拉取，容错更高（避免某个字段不存在导致整批失败）。"""
    fetched = {}
    fallback_info = {}
    universe_fields = set(universe_fields or [])
    requested_start_ts = pd.Timestamp(start).normalize()
    for key in fields:
        attempt_ends = [end]
        if key in universe_fields:
            attempt_ends.extend(_fallback_end_dates(end, universe_fallback_days))

        last_error = None
        for attempt_index, attempt_end in enumerate(attempt_ends):
            attempt_end_ts = pd.Timestamp(attempt_end).normalize()
            attempt_start = str(min(requested_start_ts, attempt_end_ts).date())
            try:
                data = datahub_daily_fn([key], attempt_start, attempt_end, cpu_num=cpu_num)
                if key in data and isinstance(data[key], pd.DataFrame):
                    fetched[key] = _normalize_frame(data[key])
                    if attempt_end != end:
                        fallback_info[key] = {
                            "requested_start": str(start),
                            "requested_end": str(end),
                            "actual_start": attempt_start,
                            "actual_end": str(attempt_end),
                            "calendar_days_back": attempt_index,
                        }
                        print(
                            "[WARN] fetched universe field with fallback:",
                            key,
                            f"requested_end={end}",
                            f"actual_end={attempt_end}",
                            fetched[key].shape,
                        )
                    else:
                        print("[OK] fetched field:", key, fetched[key].shape)
                    break
                last_error = RuntimeError(f"field not returned as DataFrame: {key}")
                print("[WARN] field not returned as DataFrame:", key, f"(end={attempt_end})")
            except Exception as exc:
                last_error = exc
                if attempt_end == end:
                    print("[WARN] fetch failed for field {}: {}".format(key, exc))
                else:
                    print(
                        "[WARN] fallback fetch failed for field {} with end={}: {}".format(
                            key,
                            attempt_end,
                            exc,
                        )
                    )
        else:
            if last_error is not None:
                print("[WARN] giving up field {} after fallback attempts: {}".format(key, last_error))
    return fetched, fallback_info


def _aggregate_minute_to_daily(frame: pd.DataFrame, agg: str) -> pd.DataFrame:
    out = _normalize_frame(frame)
    day_index = out.index.normalize()
    if agg == "mean":
        daily = out.groupby(day_index).mean()
    elif agg == "last":
        daily = out.groupby(day_index).last()
    elif agg == "sum":
        daily = out.groupby(day_index).sum()
    elif agg == "std":
        daily = out.groupby(day_index).std().fillna(0.0)
    else:
        raise ValueError(f"Unsupported minute aggregation: {agg}")
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def _fetch_minute_fields(
    datahub_minute_fn,
    minute_fields,
    start,
    end,
    cpu_num,
    agg,
    minute_start_time,
    minute_end_time,
    prefix,
):
    fetched = {}
    loaded = []

    if not minute_fields:
        return fetched, loaded

    if datahub_minute_fn is None:
        print("[WARN] 当前 read_datahub 不支持 datahub_minute，跳过分钟字段")
        return fetched, loaded

    for key in minute_fields:
        try:
            minute_dict = datahub_minute_fn(
                [key],
                start,
                end,
                start_time=minute_start_time,
                end_time=minute_end_time,
                by_dict=True,
                isst=False,
                date_nm="date",
                cpu_num=max(1, min(int(cpu_num), 32)),
            )
            if key not in minute_dict or not isinstance(minute_dict[key], pd.DataFrame):
                print("[WARN] minute field not returned as DataFrame:", key)
                continue

            daily = _aggregate_minute_to_daily(minute_dict[key], agg=agg)
            factor_name = f"{prefix}{key}"
            fetched[factor_name] = daily
            loaded.append(key)
            print("[OK] fetched minute field:", key, "->", factor_name, daily.shape)
        except Exception as exc:
            print("[WARN] minute fetch failed for field {}: {}".format(key, exc))

    return fetched, loaded


def _pick_first_available(fetched, candidates):
    for key in candidates:
        if key in fetched:
            return fetched[key], key
    return None, None


def _align_all(close_df, factors):
    common_index = close_df.index
    common_columns = close_df.columns

    for _, frame in factors.items():
        common_index = common_index.intersection(frame.index)
        common_columns = common_columns.intersection(frame.columns)

    common_index = common_index.sort_values()
    common_columns = common_columns.sort_values()

    if len(common_index) == 0:
        raise ValueError("对齐后日期为空，请检查 start/end 与字段可用性")
    if len(common_columns) == 0:
        raise ValueError("对齐后股票池为空，请检查字段列名是否一致")

    close_aligned = close_df.reindex(index=common_index, columns=common_columns).ffill().fillna(0.0)

    aligned_factors = {}
    for name, frame in factors.items():
        aligned = frame.reindex(index=common_index, columns=common_columns).ffill().fillna(0.0)
        aligned_factors[name] = aligned

    return close_aligned, aligned_factors


def _evaluate_factor_quality(frame: pd.DataFrame) -> dict:
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    total_count = int(cleaned.size)
    non_null_count = int(cleaned.notna().sum().sum())
    non_null_ratio = float(non_null_count / total_count) if total_count > 0 else 0.0

    cross_sectional_std = cleaned.std(axis=1, skipna=True)
    effective_std_days = int((cross_sectional_std > 1e-12).sum())

    return {
        "non_null_ratio": round(non_null_ratio, 6),
        "effective_std_days": effective_std_days,
    }


def _filter_optional_factor_quality(close_df, factor_frames, used_mapping):
    filtered = {}
    quality_report = {}

    protected_factors = set(CORE_FACTORS) | {
        "vol",
        "amount",
        "free_turn",
        "turn",
        "S_MFD_INFLOW_LARGE_ORDER",
        "S_MFD_INFLOW",
        "volatility",
        "s_val_pb_new",
        "roe_ttm",
    }

    for name, frame in factor_frames.items():
        aligned = frame.reindex(index=close_df.index, columns=close_df.columns)
        quality = _evaluate_factor_quality(aligned)
        source_name = used_mapping.get(name, name)

        keep = True
        reason = "kept"
        if name not in CORE_FACTORS:
            if quality["non_null_ratio"] < MIN_OPTIONAL_FACTOR_NON_NULL_RATIO:
                keep = False
                reason = "dropped_low_coverage"
            elif quality["effective_std_days"] < MIN_OPTIONAL_FACTOR_EFFECTIVE_STD_DAYS:
                keep = False
                reason = "dropped_low_cross_sectional_variation"

        quality_report[name] = {
            "source": source_name,
            "status": reason,
            **quality,
        }

        if keep:
            filtered[name] = frame

    return filtered, quality_report


def _build_factor_dict(fetched, allow_missing_core):
    factor_frames = {}
    used_mapping = {}
    non_stock_auxiliary_sources = {
        field
        for groups in (UNIVERSE_FIELD_CANDIDATES, BENCHMARK_FIELD_CANDIDATES)
        for candidates in groups.values()
        for field in candidates
    }

    # 先拿 close（必须）
    close_df, close_source = _pick_first_available(fetched, DEFAULT_FIELD_CANDIDATES["close"])
    if close_df is None:
        raise ValueError("缺少 close/close_adj，无法构建数据集")
    used_mapping["close"] = close_source
    factor_frames["close"] = close_df

    # 核心字段
    for factor in CORE_FACTORS:
        if factor == "close":
            continue
        frame, source = _pick_first_available(fetched, DEFAULT_FIELD_CANDIDATES.get(factor, []))
        if frame is not None:
            factor_frames[factor] = frame
            used_mapping[factor] = source
        else:
            if factor in ("open", "high", "low"):
                factor_frames[factor] = close_df.copy()
                used_mapping[factor] = "proxy_from_close"
                print("[WARN] {} 缺失，已使用 close 代理".format(factor))
            elif factor == "pct_chg":
                pct = close_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
                factor_frames[factor] = pct
                used_mapping[factor] = "computed_from_close"
                print("[WARN] pct_chg 缺失，已由 close 计算 (x100)")
            elif factor == "total_mv" and "circ_mv" in factor_frames:
                factor_frames[factor] = factor_frames["circ_mv"].copy()
                used_mapping[factor] = "proxy_from_circ_mv"
                print("[WARN] total_mv 缺失，已使用 circ_mv 代理")
            elif allow_missing_core:
                factor_frames[factor] = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
                used_mapping[factor] = "filled_zero"
                print("[WARN] 核心字段 {} 缺失，已用 0 填充".format(factor))
            else:
                raise ValueError(
                    "核心字段 {} 缺失；请补充 datahub 字段，或加 --allow-missing-core".format(factor)
                )

    # 额外可用字段（增强量能与风险维度）
    extra_targets = [
        "vol",
        "amount",
        "free_turn",
        "turn",
        "S_MFD_INFLOW_LARGE_ORDER",
        "S_MFD_INFLOW",
        "volatility",
        "s_val_pb_new",
        "roe_ttm",
    ]
    for factor in extra_targets:
        frame, source = _pick_first_available(fetched, DEFAULT_FIELD_CANDIDATES.get(factor, []))
        if frame is not None:
            factor_frames[factor] = frame
            used_mapping[factor] = source

    # 将用户显式拉取到的其他字段作为附加因子保留（避免“抓到了但没进模型”）
    already_used_sources = {
        v for v in used_mapping.values()
        if isinstance(v, str) and not v.startswith("derived_from_") and v not in {"filled_zero", "proxy_from_close", "proxy_from_circ_mv", "computed_from_close"}
    }
    for source_name, frame in fetched.items():
        if source_name in already_used_sources:
            continue
        if source_name in non_stock_auxiliary_sources:
            continue
        if source_name in factor_frames:
            continue
        factor_frames[source_name] = frame
        used_mapping[source_name] = source_name

    # 生成与原 TuShare 管线一致的量能衍生因子，避免表达式缺失（如 VOLUME_RATIO_10）
    # 优先使用 vol；若无则退化使用 amount；再无则使用 turnover_rate 近似
    volume_base_name = None
    if "vol" in factor_frames:
        volume_base_name = "vol"
    elif "amount" in factor_frames:
        volume_base_name = "amount"
    elif "turnover_rate" in factor_frames:
        volume_base_name = "turnover_rate"

    if volume_base_name is not None:
        volume = factor_frames[volume_base_name].copy()
        rolling_windows = {5: 1.5, 10: 1.8, 20: 2.0}
        for window, break_threshold in rolling_windows.items():
            mean_volume = volume.rolling(window=window).mean()
            ratio = (volume / mean_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            brk = (volume > mean_volume * break_threshold).astype(float).fillna(0.0)
            factor_frames[f"VOLUME_RATIO_{window}"] = ratio
            factor_frames[f"VOLUME_BREAK_{window}"] = brk
            used_mapping[f"VOLUME_RATIO_{window}"] = f"derived_from_{volume_base_name}"
            used_mapping[f"VOLUME_BREAK_{window}"] = f"derived_from_{volume_base_name}"

        momentum_5 = (volume / volume.shift(5)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        momentum_10 = (volume / volume.shift(10)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        factor_frames["VOLUME_MOMENTUM_5"] = momentum_5
        factor_frames["VOLUME_MOMENTUM_10"] = momentum_10
        used_mapping["VOLUME_MOMENTUM_5"] = f"derived_from_{volume_base_name}"
        used_mapping["VOLUME_MOMENTUM_10"] = f"derived_from_{volume_base_name}"

    return factor_frames, used_mapping


def _load_custom_fields(args):
    if args.fields.strip():
        return [f.strip() for f in args.fields.split(",") if f.strip()]

    # 自动尝试所有候选字段
    fields = []
    for _, candidates in DEFAULT_FIELD_CANDIDATES.items():
        for field in candidates:
            if field not in fields:
                fields.append(field)

    for _, candidates in BENCHMARK_FIELD_CANDIDATES.items():
        for field in candidates:
            if field not in fields:
                fields.append(field)

    for field in MONEYFLOW_DATAHUB_FIELDS:
        if field not in fields:
            fields.append(field)

    # 非全市场时，自动附加指数成分权重字段（用于股票池过滤）
    if args.universe == "hs300":
        for field in UNIVERSE_FIELD_CANDIDATES["hs300"]:
            if field not in fields:
                fields.append(field)
    elif args.universe == "zz500":
        for field in UNIVERSE_FIELD_CANDIDATES["zz500"]:
            if field not in fields:
                fields.append(field)
    elif args.universe == "zz500_zz1000":
        for key in ("zz500", "zz1000"):
            for field in UNIVERSE_FIELD_CANDIDATES[key]:
                if field not in fields:
                    fields.append(field)

    return fields


def _first_existing_frame(fetched, candidates):
    for name in candidates:
        frame = fetched.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame, name
    return None, None


def _build_benchmark_series(fetched, date_index: pd.Index) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """提取并对齐指数基线，当前默认保留 HS300 与 ZZ500。"""
    benchmarks: dict[str, pd.Series] = {}
    benchmark_sources: dict[str, str] = {}

    for benchmark_name, candidates in BENCHMARK_FIELD_CANDIDATES.items():
        frame, source_name = _first_existing_frame(fetched, candidates)
        if frame is None:
            continue
        try:
            series = _normalize_benchmark_series(frame, source_name)
            aligned = series.reindex(date_index).ffill()
            if aligned.notna().any():
                benchmarks[benchmark_name] = aligned.astype(float)
                benchmark_sources[benchmark_name] = source_name
        except Exception as exc:
            print(f"[WARN] benchmark {benchmark_name} build failed: {exc}")

    return benchmarks, benchmark_sources


def _build_universe_columns(universe, fetched, date_index, columns):
    """根据指数成分权重构建静态股票池列集合（样本期间任一日成分即纳入）。"""
    if universe == "all_a":
        return columns, "all_a", []

    used_sources = []

    def _cols_from_weight(weight_df: pd.DataFrame):
        aligned = weight_df.reindex(index=date_index, columns=columns).fillna(0.0)
        return set(aligned.columns[(aligned > 0).any(axis=0)])

    if universe == "hs300":
        weight, src = _first_existing_frame(fetched, UNIVERSE_FIELD_CANDIDATES["hs300"])
        if weight is None:
            raise ValueError("未获取到 HS300 成分权重字段（例如 HS300_data），无法按 hs300 过滤")
        used_sources.append(src)
        selected = _cols_from_weight(weight)
        return pd.Index(sorted(selected)), "hs300", used_sources

    if universe == "zz500":
        weight, src = _first_existing_frame(fetched, UNIVERSE_FIELD_CANDIDATES["zz500"])
        if weight is None:
            raise ValueError("未获取到 ZZ500 成分权重字段（例如 ZZ500_data），无法按 zz500 过滤")
        used_sources.append(src)
        selected = _cols_from_weight(weight)
        return pd.Index(sorted(selected)), "zz500", used_sources

    if universe == "zz500_zz1000":
        weight_500, src_500 = _first_existing_frame(fetched, UNIVERSE_FIELD_CANDIDATES["zz500"])
        weight_1000, src_1000 = _first_existing_frame(fetched, UNIVERSE_FIELD_CANDIDATES["zz1000"])

        if weight_500 is None:
            raise ValueError("未获取到 ZZ500 成分权重字段（例如 ZZ500_data），无法按 zz500_zz1000 过滤")
        if weight_1000 is None:
            raise ValueError("未获取到 ZZ1000 成分权重字段（例如 ZZ1000_data），无法按 zz500_zz1000 过滤")

        used_sources.extend([src_500, src_1000])
        selected = _cols_from_weight(weight_500) | _cols_from_weight(weight_1000)
        return pd.Index(sorted(selected)), "zz500_zz1000", used_sources

    raise ValueError(f"Unsupported universe: {universe}")


def _save_pickle(path, obj):
    serializer = _get_serializer_module()
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        serializer.dump(obj, f)


def main():
    args = _parse_args()
    datahub_daily_fn, datahub_minute_fn = _ensure_datahub_importable(args.datahub_path)

    fetch_fields = _load_custom_fields(args)
    print("=" * 80)
    print("Build backtest pickles from datahub")
    print("date range:", args.start, "~", args.end)
    print("requested fields:", ", ".join(fetch_fields))
    print("=" * 80)

    universe_fields = _collect_universe_filter_fields(args.universe)
    fetched_daily, field_fallback_info = _fetch_fields(
        datahub_daily_fn,
        fetch_fields,
        args.start,
        args.end,
        args.cpu_num,
        universe_fields=universe_fields,
        universe_fallback_days=args.universe_fallback_days,
    )

    minute_fields = [f.strip() for f in args.minute_fields.split(",") if f.strip()]
    fetched_minute, minute_loaded = _fetch_minute_fields(
        datahub_minute_fn=datahub_minute_fn,
        minute_fields=minute_fields,
        start=args.start,
        end=args.end,
        cpu_num=args.cpu_num,
        agg=args.minute_agg,
        minute_start_time=args.minute_start_time,
        minute_end_time=args.minute_end_time,
        prefix=args.minute_prefix,
    )

    fetched = dict(fetched_daily)
    fetched.update(fetched_minute)

    if not fetched:
        raise RuntimeError("没有成功获取任何字段")

    factor_frames, used_mapping = _build_factor_dict(fetched, allow_missing_core=args.allow_missing_core)

    close_df = factor_frames["close"]
    factor_only = {k: v for k, v in factor_frames.items()}

    close_df, factor_only = _align_all(close_df, factor_only)

    selected_columns, universe_name, universe_sources = _build_universe_columns(
        args.universe,
        fetched,
        close_df.index,
        close_df.columns,
    )
    if len(selected_columns) == 0:
        raise ValueError(f"universe={universe_name} 过滤后股票池为空，请检查指数成分数据和日期范围")

    close_df = close_df.reindex(columns=selected_columns)
    for name in list(factor_only.keys()):
        factor_only[name] = factor_only[name].reindex(columns=selected_columns)

    factor_only, factor_quality = _filter_optional_factor_quality(close_df, factor_only, used_mapping)

    benchmarks, benchmark_sources = _build_benchmark_series(fetched, close_df.index)

    print(f"[INFO] universe={universe_name}, selected_stocks={len(selected_columns)}")
    if universe_sources:
        print(f"[INFO] universe_sources={', '.join(universe_sources)}")
    if field_fallback_info:
        print(f"[INFO] universe field fallback applied: {json.dumps(field_fallback_info, ensure_ascii=False)}")
    if benchmark_sources:
        print(f"[INFO] benchmarks={benchmark_sources}")

    # 再次清洗，确保无 inf
    close_df = close_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    for name in list(factor_only.keys()):
        factor_only[name] = factor_only[name].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    metadata = {
        "job": args.metadata_job,
        "stock_count": int(close_df.shape[1]),
        "date_range": [str(close_df.index.min().date()), str(close_df.index.max().date())],
        "factor_count": int(len(factor_only)),
        "years": round((close_df.index.max() - close_df.index.min()).days / 365.0, 4),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "datahub_daily",
        "source_minute_agg": args.minute_agg,
        "minute_fields_requested": minute_fields,
        "minute_fields_loaded": minute_loaded,
        "universe": universe_name,
        "universe_sources": universe_sources,
        "universe_fallback_days": int(args.universe_fallback_days),
        "field_fetch_fallback": field_fallback_info,
        "benchmark_sources": benchmark_sources,
        "field_mapping": used_mapping,
        "factor_quality": factor_quality,
        "shape": {
            "dates": int(close_df.shape[0]),
            "stocks": int(close_df.shape[1]),
        },
    }

    # close 文件：直接 DataFrame（与现有管线兼容）
    _save_pickle(args.output_close, close_df)

    # factors 文件：dict + metadata（与现有管线一致）
    _save_pickle(
        args.output_factors,
        {
            "factors": factor_only,
            "metadata": metadata,
            "benchmarks": benchmarks,
        },
    )

    print("\n✅ 输出完成")
    print("close ->", args.output_close)
    print("factors ->", args.output_factors)
    print("samples:", close_df.shape[0], "stocks:", close_df.shape[1], "factors:", len(factor_only))
    print("core factors check:", {k: (k in factor_only) for k in CORE_FACTORS})
    print("\nmetadata preview:\n", json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
