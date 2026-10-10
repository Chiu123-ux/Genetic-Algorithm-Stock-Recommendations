"""Helpers for loading 5D VWAP targets from parquet and aligning signal/entry dates."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def list_parquet_files(path_like: str | Path) -> list[Path]:
    """Resolve parquet files from a single file or a folder (flat/recursive)."""
    root = Path(path_like)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Parquet path not found: {root}")

    files = sorted(root.glob("*.parquet"))
    if files:
        return files

    files = sorted(root.glob("**/*.parquet"))
    if files:
        return files

    raise FileNotFoundError(f"No parquet files found under: {root}")


def load_vwap_target_long(
    path_like: str | Path,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """Load long-format target rows from parquet files.

    Expected columns include: date, stock, and one or more target fields.
    """
    files = list_parquet_files(path_like)
    frames: list[pd.DataFrame] = []

    must_have = list(dict.fromkeys(required_columns))
    if "date" not in must_have:
        must_have.insert(0, "date")
    if "stock" not in must_have:
        must_have.insert(1, "stock")

    for parquet_file in files:
        frame = pd.read_parquet(parquet_file)
        missing = [name for name in must_have if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing} in parquet file: {parquet_file}"
            )
        frames.append(frame[must_have].copy())

    if not frames:
        return pd.DataFrame(columns=must_have)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["stock"] = combined["stock"].astype(str)
    combined = combined.dropna(subset=["date", "stock"])

    # If duplicate (date, stock) rows exist across shards, keep the last one.
    combined = combined.drop_duplicates(subset=["date", "stock"], keep="last")
    return combined


def build_target_matrix(long_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Pivot long target rows into date x stock matrix for one value column."""
    if value_column not in long_df.columns:
        raise ValueError(f"Target field '{value_column}' not found in parquet data")

    matrix = long_df.pivot(index="date", columns="stock", values=value_column)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index()
    matrix.columns = matrix.columns.astype(str)
    return matrix


def build_entry_date_index(
    trading_dates: Iterable[pd.Timestamp] | pd.Index,
    signal_to_entry_days: int,
) -> pd.DatetimeIndex:
    """Return entry dates by shifting along trading-day calendar steps."""
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_dates)))
    if signal_to_entry_days < 0:
        raise ValueError("signal_to_entry_days must be >= 0")
    if signal_to_entry_days == 0:
        return dates
    if len(dates) <= signal_to_entry_days:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates[signal_to_entry_days:])


def align_target_to_signal_dates(
    target_matrix: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp] | pd.Index,
    signal_to_entry_days: int,
) -> pd.DataFrame:
    """Align target matrix to signal dates using trading-day offset.

    The returned frame index is the entry dates (signal date + offset by trading day).
    """
    signal_idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(signal_dates)))
    entry_dates = build_entry_date_index(signal_idx, signal_to_entry_days)
    if len(entry_dates) == 0:
        return pd.DataFrame(columns=target_matrix.columns)
    return target_matrix.reindex(entry_dates)
