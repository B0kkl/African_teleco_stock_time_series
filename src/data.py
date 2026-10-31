"""Download, validate and clean daily OHLCV data from Yahoo Finance."""
from pathlib import Path

import pandas as pd

TICKERS = {"AAF.L": "Airtel Africa", "MTN.JO": "MTN Group", "VOD.JO": "Vodacom"}
START = "2020-01-01"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_one(ticker, start=START, end=None):
    import yfinance as yf  # imported here so the rest of the package works offline

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df[OHLCV].dropna(subset=["Close"])


def download_all(tickers=TICKERS, start=START, end=None, raw_dir=None):
    """Return {company name: DataFrame}. Optionally save raw CSVs to raw_dir."""
    frames = {}
    for ticker, name in tickers.items():
        df = download_one(ticker, start, end)
        if raw_dir is not None:
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(raw_dir) / f"{ticker}.csv")
        frames[name] = df
    return frames


def quality_report(frames):
    rows = []
    for name, df in frames.items():
        r = df["Close"].pct_change()
        gaps = df.index.to_series().diff().dt.days
        rows.append({
            "stock": name, "rows": len(df),
            "first": df.index.min().date(), "last": df.index.max().date(),
            "zero-volume days": int((df["Volume"] == 0).sum()),
            "gaps > 5 days": int((gaps > 5).sum()),
            "|daily move| > 15%": int((r.abs() > 0.15).sum()),
            "duplicate dates": int(df.index.duplicated().sum()),
        })
    return pd.DataFrame(rows).set_index("stock")


def flag_bad_ticks(df, window=11, lo=0.5, hi=2.0, max_intraday_range=0.5):
    """Flag rows that look like data errors rather than real price moves.

    A close is suspect if it sits far outside the centred rolling median of its
    neighbours (catches isolated spikes/collapses of up to window//2 days while
    leaving genuine, lasting level shifts alone), or if the high-low range is
    implausibly wide for a single day.
    """
    close = df["Close"]
    med = close.rolling(window, center=True, min_periods=window // 2 + 1).median()
    ratio = close / med
    rng = (df["High"] - df["Low"]) / close
    bad = (ratio < lo) | (ratio > hi) | (rng > max_intraday_range)
    out = pd.DataFrame({"Close": close, "local_median": med, "ratio": ratio, "intraday_range": rng})
    return out[bad]


def clean_prices(df, **kwargs):
    """Drop flagged rows. Returns (clean_df, flagged_rows)."""
    flagged = flag_bad_ticks(df, **kwargs)
    return df.drop(index=flagged.index), flagged
