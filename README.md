# African Telecom Stocks: Time-Series Analysis and Prediction Pipeline

A small, reproducible quantitative-research project on **Airtel Africa** (`AAF.L`, London), **MTN Group** (`MTN.JO`, Johannesburg) and **Vodacom** (`VOD.JO`, Johannesburg), using daily data from Yahoo Finance (`yfinance`).

**Part 1** is exploratory time-series analysis with Pandas: prices, resampling, 20/200-day moving averages, volume, plus a single-quarter snapshot of the Malawi-listed TNM and Airtel Malawi from the MSE Q3 2026 report.
**Part 2** asks whether anything known at today's close predicts tomorrow's direction or return, using walk-forward validation, simple baselines and a cost-aware backtest.

> **Headline result so far:** no evidence that simple models can predict next-day direction better than chance for any of the three stocks. A data error in Yahoo's Vodacom history also produced absurd backtest numbers (-99% drawdown, +15,000% returns) until it was detected and removed. See [Results](#results) and [Data quality](#data-quality).

## Why these tickers
yfinance does not carry Malawi Stock Exchange shares (TNM, Airtel Malawi), so the analysis uses Airtel Africa (Airtel Malawi's parent) and the two largest listed telecoms on the JSE as regional comparables.

| Company | Ticker | Exchange | Quote unit |
|---|---|---|---|
| Airtel Africa | `AAF.L` | London | pence |
| MTN Group | `MTN.JO` | Johannesburg | South African cents |
| Vodacom | `VOD.JO` | Johannesburg | South African cents |

## Repository layout
```
├── notebooks/
│   └── 01_african_telecom_analysis_and_modeling.ipynb   # Part 1 analysis + Part 2 walkthrough
├── src/
│   ├── data.py        # download, quality report, bad-tick detection and cleaning
│   ├── features.py    # leakage-safe features, targets, cross-stock alignment
│   ├── train.py       # walk-forward classification and regression, scoring
│   └── backtest.py    # long-or-flat backtest with transaction costs
├── tests/test_pipeline.py   # leakage, cleaning and end-to-end checks
├── run_pipeline.py          # reproduces everything, writes reports/*.csv
├── data/raw, data/processed # downloaded and cleaned price CSVs
├── reports/                 # result tables written by the pipeline
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q            # 5 tests, no network needed
python run_pipeline.py         # downloads data, runs all models, writes reports/*.csv
jupyter notebook notebooks/    # the narrative version
```
`python run_pipeline.py --cost 0.002` changes the assumed cost per position change (default 0.1%).

## Method
1. **Data.** Daily OHLCV from 2020-01-02 to the latest session, `auto_adjust=True`. Raw CSVs are saved to `data/raw/`.
2. **Validation and cleaning.** Row counts, gaps, zero-volume days, large daily moves and duplicates are reported. Rows whose close lies far outside the median of neighbouring days (or with an impossibly wide intraday range) are dropped as probable data errors; lasting price changes are kept. The cleaning step is covered by tests.
3. **Features** (all known at the close of day *t*): 1/5/20-day returns, 5/20-day volatility, distance from 20- and 200-day averages, relative volume, high-low range, open-close return. **Cross-stock features:** the other stock's 1- and 5-day returns, aligned onto the target stock's calendar. Because the JSE closes before the LSE, MTN's same-day move is used for Airtel Africa, while in the other direction (and for Vodacom/MTN) the other stock is lagged one day to avoid look-ahead. A test verifies that scrambling future prices never changes a feature.
4. **Targets.** Direction: is tomorrow's close above today's? Return: tomorrow's percentage return.
5. **Validation.** Expanding-window walk-forward in 5 chronological folds with a 1-day gap; no shuffling. All reported predictions are out-of-sample.
6. **Models.** Direction: majority-class baseline, Logistic Regression, shallow Random Forest. Return: predict-zero baseline, training-mean baseline, Ridge, shallow Random Forest. Fixed hyperparameters, no tuning.
7. **Metrics.** Direction: accuracy and ROC-AUC. Return: MAE, RMSE, and out-of-sample R² against the zero forecast. Trading: total return, max drawdown, Sharpe, number of trades, share of days invested, versus buy-and-hold, with a cost on every position change.

## Data quality
In the first run, Vodacom's buy-and-hold showed a **-99.4% maximum drawdown** and the models appeared to earn **+13,100% to +15,000%**. The cause was a single Yahoo data point near the start of 2025 where Vodacom's price collapses to almost zero for one day and returns. A strategy that is in cash on the crash day and invested on the rebound day "earns" about 100x from a data error. `flag_bad_ticks` / `clean_prices` in `src/data.py` now detect and remove such points, and the notebook shows the before/after comparison. **Lesson: always sanity-check extreme backtest numbers against the price chart before believing them.**

## Results
Everything below is a snapshot from the **first run** (raw data, 2020-01-02 to 2026-10-29, own features only, 0.1% cost). Re-run `python run_pipeline.py` for the cleaned results with cross-stock features and the regression models; these are written to `reports/classification.csv`, `reports/regression.csv` and `reports/backtest.csv`.

**Direction classification (about 1,260 out-of-sample days per stock):**

| Stock | Majority baseline | Logistic acc. / AUC | Random forest acc. / AUC |
|---|---|---|---|
| Airtel Africa | 52.8% | 51.8% / 0.502 | 50.7% / 0.498 |
| MTN Group | 48.4% | 51.2% / 0.515 | 50.8% / 0.511 |
| Vodacom | 49.5% | 52.0% / 0.527 | 52.0% / 0.516 |

Accuracy has a standard error of roughly 1.4 points and AUC about 0.016, so none of these results is distinguishable from chance. For MTN and Vodacom the majority baseline scored under 50% only because the training-period majority ("down") differed from the test period; always predicting "up" would have scored 50.4-50.5%.

**Backtest (Airtel Africa and MTN; Vodacom's first-run figures were invalid):**

| Stock | Strategy | Total return | Max drawdown | Position changes |
|---|---|---|---|---|
| Airtel Africa | Buy and hold | +178.8% | -46.9% | 1 |
| Airtel Africa | Logistic | +38.2% | -44.7% | 279 |
| Airtel Africa | Random forest | +23.6% | -44.1% | 337 |
| MTN Group | Buy and hold | +41.6% | -65.0% | 1 |
| MTN Group | Logistic | -9.8% | -45.6% | 272 |
| MTN Group | Random forest | -18.2% | -57.8% | 379 |

Nothing beat buy-and-hold on return. Hundreds of position changes at 0.1% each cost roughly a quarter of capital. The models reduced MTN's drawdown but lost money overall.

## Limitations
- Only about 1,500 daily observations per stock and one market regime, including the 2020 shock and a very strong 2025-26 rally in Airtel Africa.
- Yahoo data needed cleaning once; other errors may remain. Adjustments for dividends and splits depend on Yahoo.
- Costs and slippage are approximated. UK stamp duty, brokerage and bid-ask spreads can differ materially from the 0.1% placeholder. Prices are in GBP (LSE) and ZAR (JSE), so currency moves are embedded in returns.
- Fixed hyperparameters and a single feature recipe; a tuned model might do somewhat better, and a null result here does not prove markets are unpredictable.
- Nothing in this repository is investment advice.

## Possible next steps
- Add volume-based and calendar features, and ZAR/GBP exchange-rate series.
- Test longer horizons (weekly direction), where signal-to-noise is usually better.
- Nested walk-forward hyperparameter tuning.
- Add the Nigerian listings (MTN Nigeria, Airtel Africa on the NGX) if a reliable data source is found.
