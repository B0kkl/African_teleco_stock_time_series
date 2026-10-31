"""Simple long-or-flat backtest with transaction costs."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity):
    return float((equity / equity.cummax() - 1).min())


def run_backtest(next_ret, position, cost=0.001):
    """next_ret: return earned from today's close to tomorrow's close.
    position: 1 (invested) or 0 (cash), decided at today's close.
    cost: charged on every change of position (including the first entry)."""
    position = position.astype(float)
    trades = position.diff().abs().fillna(position.iloc[0])
    ret = position * next_ret - trades * cost
    equity = (1 + ret).cumprod()
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    return equity, {"total_return": float(equity.iloc[-1] - 1), "max_drawdown": max_drawdown(equity),
                    "sharpe": float(sharpe), "trades": int(trades.sum()),
                    "days_invested": float(position.mean())}


def buy_and_hold(next_ret, cost=0.001):
    return run_backtest(next_ret, pd.Series(1.0, index=next_ret.index), cost)
