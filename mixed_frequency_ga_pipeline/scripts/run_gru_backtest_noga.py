#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.pipelines.gru_backtester import GRUBacktester, load_gru_backtest_config
from src.pipelines.gru_forecaster import GRUFactorForecaster, load_gru_config
from src.utils.config_loader import set_config_override
from src.utils.data_io import load_price_factor_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRU-only rolling backtest using existing factor expressions and no GA refresh."
    )
    parser.add_argument("--config", dest="config_path", required=True, help="Path to config JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_config_override(args.config_path)

    config = load_gru_config()
    backtest_config = load_gru_backtest_config()
    backtest_config.run_ga_every_n_months = 0

    model_type = str(getattr(config, "model_type", "gru") or "gru").strip().lower()
    model_label = model_type.upper()

    print("[GRU-ONLY-BACKTEST] starting")
    print(f"[GRU-ONLY-BACKTEST] config={args.config_path}")
    print(
        "[GRU-ONLY-BACKTEST] "
        f"fit_window_months={getattr(config, 'fit_window_months', None)}, "
        f"train_ratio={getattr(config, 'train_ratio', None)}, "
        f"validation_ratio={getattr(config, 'validation_ratio', None)}, "
        f"ga_refresh_disabled={backtest_config.run_ga_every_n_months}"
    )

    forecaster = GRUFactorForecaster(config)
    price_df, factors = forecaster.load_data()
    expressions = forecaster.load_factor_expressions()
    X, y, dates = forecaster.create_dataset(price_df, factors, expressions)
    _, _, metadata, benchmarks = load_price_factor_bundle(config.price_file, config.factor_file)

    universe_name = str(metadata.get("universe", "")).lower()
    benchmark_name = None
    if universe_name in benchmarks:
        benchmark_name = universe_name
    elif universe_name == "zz500_zz1000" and "zz500" in benchmarks:
        benchmark_name = "zz500"
    elif "zz500" in benchmarks:
        benchmark_name = "zz500"
    elif "hs300" in benchmarks:
        benchmark_name = "hs300"

    backtester = GRUBacktester(
        config,
        backtest_config,
        list(price_df.columns),
        benchmark_series=benchmarks,
        benchmark_name=benchmark_name,
    )
    print(
        "[GRU-ONLY-BACKTEST] "
        f"model={model_label}, samples={len(dates)}, "
        f"sample_dates={dates[0]} -> {dates[-1]}, "
        f"top_n={backtest_config.top_n}, "
        f"retrain_every_n_rebalances={backtest_config.retrain_every_n_rebalances}, "
        f"train_window_months={backtest_config.train_window_months}"
    )

    result, summary = backtester.run(X, y, dates)

    report_dir = Path(config.log_output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    curve_path = report_dir / f"gru_only_backtest_no_ga_{timestamp}.csv"
    summary_path = report_dir / f"gru_only_backtest_no_ga_summary_{timestamp}.json"

    result.to_csv(curve_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[GRU-ONLY-BACKTEST] curve={curve_path}")
    print(f"[GRU-ONLY-BACKTEST] summary={summary_path}")
    print("[GRU-ONLY-BACKTEST] done")


if __name__ == "__main__":
    main()
