"""Reproduce the whole study: download -> validate/clean -> features -> walk-forward models -> backtest.

    python run_pipeline.py            # downloads fresh data, writes CSVs to reports/
    python run_pipeline.py --cost 0.002
"""
import argparse
from pathlib import Path

import pandas as pd

from src.backtest import buy_and_hold, run_backtest
from src.data import TICKERS, clean_prices, download_all, quality_report
from src.features import FEATURES_CROSS, FEATURES_OWN, build_features, other_closes_first
from src.train import score_classifiers, score_regressors, walk_forward

NAME_TO_TICKER = {v: k for k, v in TICKERS.items()}
# Which second stock feeds the cross-stock features for each target
PAIRS = {"Airtel Africa": "MTN Group", "MTN Group": "Airtel Africa", "Vodacom": "MTN Group"}
FEATURE_SETS = {"own": FEATURES_OWN, "own+cross": FEATURES_OWN + FEATURES_CROSS}


def run_all(frames, cost=0.001):
    """frames: {company name: raw OHLCV DataFrame}. Returns a dict of result tables."""
    clean, flagged = {}, []
    for name, df in frames.items():
        clean[name], bad = clean_prices(df)
        flagged.append(bad.assign(stock=name))
    flagged = pd.concat(flagged) if flagged else pd.DataFrame()

    cls_rows, reg_rows, bt_rows = [], [], []
    for name, df in clean.items():
        other = PAIRS[name]
        same_day = other_closes_first(NAME_TO_TICKER[name], NAME_TO_TICKER[other])
        feat = build_features(df, other_close=clean[other]["Close"], other_same_day=same_day)

        bh_equity, bh_stats = None, None
        for fs_name, cols in FEATURE_SETS.items():
            p_cls = walk_forward(feat, cols, "classification")
            p_reg = walk_forward(feat, cols, "regression")
            cls_rows.append(score_classifiers(feat, p_cls).assign(stock=name, feature_set=fs_name))
            reg_rows.append(score_regressors(feat, p_reg).assign(stock=name, feature_set=fs_name))

            nxt = feat.loc[p_cls.index, "next_ret"]
            if bh_stats is None:
                _, bh_stats = buy_and_hold(nxt, cost)
                bt_rows.append({"stock": name, "feature_set": "-", "strategy": "buy_and_hold", **bh_stats})
            for m in ["logistic", "random_forest"]:
                _, s = run_backtest(nxt, p_cls[m] > 0.5, cost)
                bt_rows.append({"stock": name, "feature_set": fs_name, "strategy": f"{m}_classifier", **s})
            nxt_r = feat.loc[p_reg.index, "next_ret"]
            for m in ["ridge", "random_forest"]:
                _, s = run_backtest(nxt_r, p_reg[m] > 0, cost)
                bt_rows.append({"stock": name, "feature_set": fs_name, "strategy": f"{m}_regressor", **s})

    return {
        "data_quality_raw": quality_report(frames).reset_index(),
        "flagged_ticks": flagged.reset_index(),
        "classification": pd.concat(cls_rows).reset_index(),
        "regression": pd.concat(reg_rows).reset_index(),
        "backtest": pd.DataFrame(bt_rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.001, help="cost per position change (default 0.1%%)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    frames = download_all(raw_dir="data/raw")
    tables = run_all(frames, cost=args.cost)
    out = Path(args.out); out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
        print(f"\n=== {name} ===\n{table.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")


if __name__ == "__main__":
    main()
