#!/usr/bin/env python3
"""Build daily 2D factors from intraday minute-level long data.

Input format (long table):
- required columns: date, stock, minute
- numeric columns: arbitrary minute-level features (e.g. close, volume)

Output:
- dill pickle close dataframe (date x stock)
- dill pickle factor bundle:
  {
    "factors": Dict[str, DataFrame(date x stock)],
    "metadata": Dict,
    "benchmarks": {}
  }

This is a first practical implementation of "3D (stock/day/minute) -> 2D" reduction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import dill
import numpy as np
import pandas as pd


DEFAULT_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

# First batch of practical reducers for intraday->daily conversion.
REDUCERS = {
    "mean": lambda s: float(np.nanmean(s.values)) if len(s) else 0.0,
    "std": lambda s: float(np.nanstd(s.values)) if len(s) else 0.0,
    "max": lambda s: float(np.nanmax(s.values)) if len(s) else 0.0,
    "min": lambda s: float(np.nanmin(s.values)) if len(s) else 0.0,
    "last": lambda s: float(s.iloc[-1]) if len(s) else 0.0,
    "first": lambda s: float(s.iloc[0]) if len(s) else 0.0,
    "sum": lambda s: float(np.nansum(s.values)) if len(s) else 0.0,
    "q10": lambda s: float(np.nanpercentile(s.values, 10)) if len(s) else 0.0,
    "q50": lambda s: float(np.nanpercentile(s.values, 50)) if len(s) else 0.0,
    "q90": lambda s: float(np.nanpercentile(s.values, 90)) if len(s) else 0.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily factors from intraday long data")
    parser.add_argument("--input", required=True, help="Input parquet/csv path of long intraday data")
    parser.add_argument("--output-close", required=True, help="Output close pickle path")
    parser.add_argument("--output-factors", required=True, help="Output factors pickle path")
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated numeric feature columns to aggregate",
    )
    parser.add_argument(
        "--reducers",
        default="mean,std,max,min,last,first,sum,q10,q50,q90",
        help="Comma-separated reducer names",
    )
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--stock-col", default="stock", help="Stock column name")
    parser.add_argument("--minute-col", default="minute", help="Minute column name")
    return parser.parse_args()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {suffix}")


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _pivot_to_2d(frame: pd.DataFrame, date_col: str, stock_col: str, value_col: str) -> pd.DataFrame:
    pt = frame.pivot(index=date_col, columns=stock_col, values=value_col)
    pt.index = pd.to_datetime(pt.index)
    pt = pt.sort_index().sort_index(axis=1)
    pt = pt.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return pt


def _aggregate_daily(
    frame: pd.DataFrame,
    date_col: str,
    stock_col: str,
    minute_col: str,
    features: List[str],
    reducers: List[str],
) -> Dict[str, pd.DataFrame]:
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    frame = frame.sort_values([date_col, stock_col, minute_col])

    factors: Dict[str, pd.DataFrame] = {}

    grouped = frame.groupby([date_col, stock_col], sort=True)

    for feature in features:
        series_group = grouped[feature]
        for reducer_name in reducers:
            reducer = REDUCERS[reducer_name]
            daily = series_group.apply(reducer).reset_index(name="value")
            factor_name = f"hf_{feature}_{reducer_name}"
            factors[factor_name] = _pivot_to_2d(daily, date_col, stock_col, "value")

        # Add a practical range reducer for each feature.
        daily_range = series_group.apply(
            lambda s: float(np.nanmax(s.values) - np.nanmin(s.values)) if len(s) else 0.0
        ).reset_index(name="value")
        factors[f"hf_{feature}_range"] = _pivot_to_2d(daily_range, date_col, stock_col, "value")

    return factors


def _dump_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        dill.dump(obj, f)


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    output_close = Path(args.output_close)
    output_factors = Path(args.output_factors)

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    reducers = [x.strip() for x in args.reducers.split(",") if x.strip()]

    unknown_reducers = [r for r in reducers if r not in REDUCERS]
    if unknown_reducers:
        raise ValueError(f"Unknown reducers: {unknown_reducers}; allowed={sorted(REDUCERS)}")

    frame = _load_table(input_path)
    _ensure_columns(frame, [args.date_col, args.stock_col, args.minute_col, *features])

    frame[args.date_col] = pd.to_datetime(frame[args.date_col]).dt.normalize()

    daily_factors = _aggregate_daily(
        frame=frame,
        date_col=args.date_col,
        stock_col=args.stock_col,
        minute_col=args.minute_col,
        features=features,
        reducers=reducers,
    )

    # Use hf_close_last as daily close proxy.
    close_key = "hf_close_last"
    if close_key not in daily_factors:
        raise ValueError("hf_close_last is missing; include close feature and last reducer")
    close_df = daily_factors[close_key]

    metadata = {
        "job": "intraday_to_daily_reduce",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(input_path),
        "factor_count": len(daily_factors),
        "shape": {
            "dates": int(close_df.shape[0]),
            "stocks": int(close_df.shape[1]),
        },
        "features": features,
        "reducers": reducers,
        "note": "3D intraday tensor reduced to 2D daily factors via per-day stock-level aggregations",
    }

    _dump_pickle(output_close, close_df)
    _dump_pickle(
        output_factors,
        {
            "factors": daily_factors,
            "metadata": metadata,
            "benchmarks": {},
        },
    )

    print("=== Intraday -> Daily build done ===")
    print(f"input: {input_path}")
    print(f"output close: {output_close}")
    print(f"output factors: {output_factors}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
