import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import run_all
from src.data import clean_prices, flag_bad_ticks
from src.features import FEATURES_CROSS, FEATURES_OWN, build_features


def synth(n=900, seed=0, start="2020-01-02"):
    r = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="Date")
    c = 100 * np.exp(np.cumsum(r.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"Open": c * (1 + r.normal(0, .003, n)), "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": r.integers(1000, 5_000_000, n)}, index=idx)


def test_features_do_not_use_future_prices():
    df, other = synth(seed=1), synth(seed=2)
    cut = df.index[600]
    base = build_features(df, other["Close"], other_same_day=True)

    df2, other2 = df.copy(), other.copy()
    for d in (df2, other2):  # scramble everything strictly after the cut date
        after = d.index > cut
        d.loc[after, ["Open", "High", "Low", "Close"]] *= np.random.default_rng(9).uniform(0.5, 1.5, (after.sum(), 1))
        d.loc[after, "Volume"] = 1
    changed = build_features(df2, other2["Close"], other_same_day=True)

    cols = FEATURES_OWN + FEATURES_CROSS
    pd.testing.assert_frame_equal(base.loc[:cut, cols], changed.loc[:cut, cols])


def test_other_stock_is_lagged_when_it_closes_later():
    df, other = synth(seed=1), synth(seed=2)
    lagged = build_features(df, other["Close"], other_same_day=False)
    same = build_features(df, other["Close"], other_same_day=True)
    common = lagged.index.intersection(same.index)[5:]
    # lagged value on day t equals the same-day value on day t-1
    prev = same["xret_1"].shift(1)
    assert np.allclose(lagged.loc[common, "xret_1"].iloc[1:], prev.loc[common].iloc[1:], equal_nan=False)


def test_bad_tick_is_flagged_and_removed():
    df = synth(seed=3)
    bad_day = df.index[400]
    df.loc[bad_day, ["Open", "High", "Low", "Close"]] *= 0.01          # 100x unit glitch
    flagged = flag_bad_ticks(df)
    assert list(flagged.index) == [bad_day]
    clean, _ = clean_prices(df)
    assert bad_day not in clean.index
    assert clean["Close"].pct_change().abs().max() < 0.2


def test_genuine_level_shift_is_kept():
    df = synth(seed=4)
    df.loc[df.index[400]:, ["Open", "High", "Low", "Close"]] *= 0.6   # lasting -40% repricing
    assert flag_bad_ticks(df).empty


def test_pipeline_runs_end_to_end_and_survives_a_glitch():
    frames = {"Airtel Africa": synth(seed=1), "MTN Group": synth(seed=2), "Vodacom": synth(seed=3)}
    g = frames["Vodacom"].index[500]
    frames["Vodacom"].loc[g, ["Open", "High", "Low", "Close"]] *= 0.01
    t = run_all(frames)
    assert len(t["flagged_ticks"]) == 1
    bt = t["backtest"]
    assert bt["max_drawdown"].min() > -0.95          # no phantom -99% drawdowns after cleaning
    assert bt["total_return"].max() < 20             # no phantom +10,000% returns
    assert set(t["classification"]["feature_set"]) == {"own", "own+cross"}
    assert {"MAE", "RMSE", "oos_r2_vs_zero"} <= set(t["regression"].columns)
