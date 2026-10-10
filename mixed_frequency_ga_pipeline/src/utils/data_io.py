"""Shared helpers for loading dill-serialised market datasets."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import dill
import numpy as np
import pandas as pd


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return dill.load(handle)


def _normalise_benchmark_series(series_like: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(series_like, pd.DataFrame):
        if series_like.shape[1] == 0:
            raise ValueError("Benchmark dataframe has no columns")
        if series_like.shape[1] == 1:
            series = series_like.iloc[:, 0]
        else:
            series = series_like.mean(axis=1)
    else:
        series = series_like.copy()

    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce")
    return series.sort_index()


def load_price_factor_bundle(
    price_file: str | Path,
    factor_file: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, pd.Series]]:
    """Load price/factor data together with optional metadata and benchmarks."""

    price_path = Path(price_file)
    factor_path = Path(factor_file)

    price_data = _load_pickle(price_path)

    if isinstance(price_data, pd.DataFrame):
        price_df = price_data
    elif isinstance(price_data, dict) and "factors" in price_data:
        price_df = price_data["factors"].get("close")
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported price data format in {price_path}")

    factor_data = _load_pickle(factor_path)

    metadata: Dict[str, Any] = {}
    benchmarks: Dict[str, pd.Series] = {}
    if isinstance(factor_data, dict) and "factors" in factor_data:
        factor_dict = factor_data["factors"]
        metadata = factor_data.get("metadata", {}) or {}
        raw_benchmarks = factor_data.get("benchmarks", {}) or {}
        if isinstance(raw_benchmarks, dict):
            for name, item in raw_benchmarks.items():
                try:
                    benchmarks[str(name)] = _normalise_benchmark_series(item)
                except Exception:
                    continue
    elif isinstance(factor_data, dict) and all(isinstance(item, pd.DataFrame) for item in factor_data.values()):
        factor_dict = factor_data
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported factor data format in {factor_path}")

    if "close" in factor_dict and isinstance(factor_dict["close"], pd.DataFrame):
        # Prefer the factor bundle's close matrix so downstream training/backtests
        # do not depend on a separately pickled price matrix that may have been
        # backfilled historically.
        price_df = factor_dict["close"].copy()

    price_df.index = pd.to_datetime(price_df.index)
    for name, frame in factor_dict.items():
        aligned = frame.reindex(price_df.index).ffill()
        factor_dict[name] = aligned.fillna(0)

    aligned_benchmarks: Dict[str, pd.Series] = {}
    for name, series in benchmarks.items():
        aligned = series.reindex(price_df.index).ffill()
        if aligned.notna().any():
            aligned_benchmarks[name] = aligned.astype(float)

    return price_df, factor_dict, metadata, aligned_benchmarks


def load_price_and_factors(
    price_file: str | Path,
    factor_file: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load price and factor dataframes from dill files.

    Both files may contain either raw :class:`pandas.DataFrame` objects or a
    dictionary with a ``factors`` key that maps factor names to individual
    dataframes. The helper normalises both representations so callers receive a
    price dataframe indexed by ``DatetimeIndex`` and a dictionary of factor
    dataframes aligned to the same index.
    """

    price_df, factor_dict, _, _ = load_price_factor_bundle(price_file, factor_file)
    return price_df, factor_dict


@lru_cache(maxsize=8)
def load_minute_supported_stock_columns(minute_cache_root: str | Path) -> pd.Index:
    """Return the stock universe that is actually covered by minute_cache.

    Prefer the explicit whitelist saved next to the cache. Fall back to the
    union of all cached `stocks.npy` files when the whitelist is absent.
    """

    cache_root = Path(minute_cache_root)
    whitelist_path = cache_root.parent / "stock_whitelist.txt"
    if whitelist_path.exists():
        stocks = [
            line.strip()
            for line in whitelist_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.Index(stocks, dtype=object)

    stock_union: set[str] = set()
    for stocks_file in cache_root.rglob("stocks.npy"):
        values = np.load(stocks_file, allow_pickle=True)
        stock_union.update(str(item) for item in values.tolist())

    return pd.Index(sorted(stock_union), dtype=object)


def restrict_price_factors_to_columns(
    price_df: pd.DataFrame,
    factor_dict: Dict[str, pd.DataFrame],
    supported_columns: pd.Index | list[str] | tuple[str, ...],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Restrict price/factor bundle to a shared stock-column intersection."""

    supported = {str(item) for item in supported_columns}
    selected_columns = pd.Index([col for col in price_df.columns if str(col) in supported])
    if selected_columns.empty:
        raise ValueError("No overlapping stock columns remain after minute-cache intersection")

    restricted_price = price_df.reindex(columns=selected_columns)
    restricted_factors = {
        name: frame.reindex(columns=selected_columns)
        for name, frame in factor_dict.items()
    }
    return restricted_price, restricted_factors
