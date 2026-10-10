import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipelines.gru_forecaster import GRUFactorForecaster, load_gru_config


def main():
    cfg = load_gru_config()
    forecaster = GRUFactorForecaster(cfg)

    print("Loading price and factor data...")
    price_df, factor_dict = forecaster.load_data()
    print(f"Price shape: {price_df.shape}, #factors: {len(factor_dict)}")

    last_prices = price_df.iloc[-1]
    print("Last prices sample:\n", last_prices.head(10))

    print("Loading expressions...")
    expressions = forecaster.load_factor_expressions()
    print(f"Loaded {len(expressions)} expressions (showing up to 10):\n", expressions[:10])

    print("Building dataset (this may take a moment)...")
    X, y, dates = forecaster.create_dataset(price_df, factor_dict, expressions)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    # compute some target stats
    mean_y = np.nanmean(y, axis=0)
    median_y = np.nanmedian(y, axis=0)

    # align last_prices order with mean_y (they should be same stock order)
    # create pandas Series for convenient correlations
    stocks = list(price_df.columns)
    s_last = pd.Series(last_prices.values, index=stocks)
    s_meany = pd.Series(mean_y, index=stocks)
    s_medy = pd.Series(median_y, index=stocks)

    print('\n--- Diagnostics ---')
    try:
        p_corr = s_last.corr(s_meany, method='pearson')
        s_corr = s_last.corr(s_meany, method='spearman')
        print(f"Correlation between last price and mean(target): Pearson={p_corr:.4f}, Spearman={s_corr:.4f}")
    except Exception as e:
        print("Failed to compute correlation:", e)

    try:
        p_corr2 = s_last.corr(s_medy, method='pearson')
        s_corr2 = s_last.corr(s_medy, method='spearman')
        print(f"Correlation between last price and median(target): Pearson={p_corr2:.4f}, Spearman={s_corr2:.4f}")
    except Exception as e:
        print("Failed to compute correlation:", e)

    print('\nShow sample of target (mean) for first 10 stocks:')
    print(s_meany.head(10))

    # If a model exists, try to load and predict
    model_path = Path(cfg.model_output_dir) / 'gru_model_latest.pt'
    if model_path.exists():
        print(f"Found model at {model_path}, attempting to load and predict latest sequence...")
        latest_seq, stock_names = forecaster.prepare_latest_sequence(price_df, factor_dict, expressions)
        try:
            forecaster.load_model(latest_seq, stock_names, model_path)
            preds = forecaster.predict(latest_seq)
            s_preds = pd.Series(preds, index=stock_names)
            print('Predictions sample (first 10):')
            print(s_preds.head(10))
            print('Correlation between predictions and last price: ',
                  s_preds.corr(s_last, method='spearman'))
            print('Correlation between predictions and mean(target): ',
                  s_preds.corr(s_meany, method='spearman'))
        except Exception as e:
            print('Failed to load/predict with model:', e)
    else:
        print(f"No model found at {model_path} — skip model prediction step.")


if __name__ == '__main__':
    main()
