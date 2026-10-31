"""Feature engineering. Every feature on row t uses only information known at the close of day t."""
import numpy as np
import pandas as pd

FEATURES_OWN = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20",
                "dist_ma20", "dist_ma200", "rel_vol", "hl_range", "oc_ret"]
FEATURES_CROSS = ["xret_1", "xret_5"]

# Approximate order in which exchanges close within a calendar day.
# JSE closes 17:00 SAST (15:00 UTC); LSE closes 16:30 London time (15:30-16:30 UTC).
CLOSE_ORDER = {"JO": 0, "L": 1}


def other_closes_first(target_ticker, other_ticker):
    """True if the other stock's close on day t is already known when the target closes."""
    ex = lambda t: t.rsplit(".", 1)[-1]
    return CLOSE_ORDER[ex(other_ticker)] < CLOSE_ORDER[ex(target_ticker)]


def build_features(df, other_close=None, other_same_day=False):
    """Features, next-day return and direction target for one stock.

    other_close: Close series of a second stock, mapped onto this stock's trading
        calendar with a forward fill (so the other market's holidays give a flat day).
    other_same_day: use the other stock's day-t move only if its market closes before
        this one (see other_closes_first); otherwise it is lagged one day to avoid look-ahead.
    """
    c = df["Close"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_1"], f["ret_5"], f["ret_20"] = r, c.pct_change(5), c.pct_change(20)
    f["vol_5"], f["vol_20"] = r.rolling(5).std(), r.rolling(20).std()
    f["dist_ma20"] = c / c.rolling(20).mean() - 1
    f["dist_ma200"] = c / c.rolling(200).mean() - 1
    f["rel_vol"] = df["Volume"] / df["Volume"].rolling(20).mean()
    f["hl_range"] = (df["High"] - df["Low"]) / c
    f["oc_ret"] = c / df["Open"] - 1

    if other_close is not None:
        o = other_close.reindex(other_close.index.union(df.index)).ffill().reindex(df.index)
        x1, x5 = o.pct_change(), o.pct_change(5)
        if not other_same_day:
            x1, x5 = x1.shift(1), x5.shift(1)
        f["xret_1"], f["xret_5"] = x1, x5

    f["next_ret"] = r.shift(-1)          # tomorrow's return: evaluation only, never a feature
    f["target"] = (f["next_ret"] > 0).astype(int)
    return f.replace([np.inf, -np.inf], np.nan).dropna()  # drops warm-up rows and the final row
