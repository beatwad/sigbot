import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

risk = 0.0026
leverage = 4
min_free_balance = 0.2
open_comission = 0.00036
close_comission = 0.001
slippage = 0.002


def calculate_profit(
    target: int,
    quantity: float,
    first_price: float,
    last_price: float,
    tp: float,
    sl: float,
) -> Tuple[float, float]:
    """Calculate profit change according to trade success and commissions."""
    if target == 1:
        if last_price > 0:
            # price didn't reach TP or SL — calculate from entry/exit prices
            trade_profit = quantity * (1 + (last_price - first_price) / first_price * leverage)
        else:
            # TP reached
            trade_profit = quantity * (1 + tp * leverage)
    else:
        # SL reached
        trade_profit = quantity * (1 - sl * leverage)

    trade_profit *= 1 - close_comission
    profit = trade_profit - quantity
    return profit, trade_profit


def cap_max_num_simult_trades(df: pd.DataFrame, max_num_simult_trades: int) -> pd.DataFrame:
    """Cap the maximum number of simultaneously opened trades."""
    df["row_number"] = (
        df.sort_values("pred", ascending=False).groupby("time")["target"].cumcount() + 1
    )
    df = df[df["row_number"] <= max_num_simult_trades]
    return df


def backtest(
    df: pd.DataFrame,
    oof: np.ndarray,
    val_idxs: list,
    high_bound: float,
    tp: float,
    sl: float,
    show_progress: bool = False,
    max_num_simult_trades: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """Run a backtest on model predictions."""
    backtest_df = df.loc[
        val_idxs,
        ["target", "max_price_deviation", "time", "close_time", "first_price", "last_price"],
    ]
    backtest_df["pred"] = oof
    backtest_df = backtest_df[backtest_df["pred"] >= high_bound]
    # The simulation is path-dependent (compounding balance, free-balance position sizing,
    # and time-based trade closing), so rows must be processed chronologically. val_idxs is
    # built by tiling folds backwards in time, so sort here before iterating.
    backtest_df = backtest_df.sort_values("time")
    if max_num_simult_trades > 0:
        backtest_df = cap_max_num_simult_trades(backtest_df, max_num_simult_trades)
    backtest_df = backtest_df.reset_index(drop=True)

    n = len(backtest_df)
    if n == 0:
        for col in ("balance", "free_balance", "profit", "trade_profit", "quantity"):
            backtest_df[col] = pd.Series(dtype=float)
        backtest_df["profit_count"] = pd.Series(dtype=int)
        return 0, backtest_df

    # Work on numpy arrays instead of per-cell .loc reads/writes, and track open trades in a
    # min-heap keyed on close time. Rows are already sorted by time, so each step pops every
    # trade whose close time has passed in O(log n) instead of scanning all open trades.
    signal_times = backtest_df["time"].to_numpy().astype("int64")
    targets = backtest_df["target"].to_numpy()
    first_prices = backtest_df["first_price"].to_numpy()
    last_prices = backtest_df["last_price"].to_numpy()
    close_dt = backtest_df["close_time"].to_numpy()
    close_is_nat = np.isnat(close_dt)
    close_times = close_dt.astype("int64")

    balance_arr = np.empty(n)
    free_balance_arr = np.empty(n)
    profit_arr = np.zeros(n)
    trade_profit_arr = np.zeros(n)
    quantity_arr = np.zeros(n)
    profit_count_arr = np.zeros(n, dtype=int)

    open_trades: List[Tuple[int, int]] = []  # min-heap of (close_time_ns, row_idx)

    iterator = tqdm(range(n)) if show_progress else range(n)

    for i in iterator:
        balance = balance_arr[i - 1] if i > 0 else 1.0
        free_balance = free_balance_arr[i - 1] if i > 0 else 1.0

        signal_time = signal_times[i]
        while open_trades and open_trades[0][0] <= signal_time:
            _, j = heapq.heappop(open_trades)
            free_balance += trade_profit_arr[j]
            profit_count_arr[i] += 1

        if free_balance >= min_free_balance * balance:
            quantity = free_balance * risk / (sl * leverage)
            balance -= quantity * open_comission
            free_balance -= quantity * (1 + open_comission)
            profit, trade_profit = calculate_profit(
                targets[i], quantity, first_prices[i], last_prices[i], tp, sl
            )
        else:
            profit, trade_profit = 0.0, 0.0
            quantity = 0.0

        quantity_arr[i] = quantity
        balance_arr[i] = balance + profit
        free_balance_arr[i] = free_balance
        profit_arr[i] = profit
        trade_profit_arr[i] = trade_profit

        # NaT close times never satisfy the close condition (matching the original >= NaT == False
        # behaviour), so they are simply never queued and stay open until the end.
        if not close_is_nat[i]:
            heapq.heappush(open_trades, (close_times[i], i))

    backtest_df["balance"] = balance_arr
    backtest_df["free_balance"] = free_balance_arr
    backtest_df["profit"] = profit_arr
    backtest_df["trade_profit"] = trade_profit_arr
    backtest_df["quantity"] = quantity_arr
    backtest_df["profit_count"] = profit_count_arr

    result = round(balance_arr[-1] * 100, 2)
    return result, backtest_df


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms."""
    if (y_pred <= 0).any() or (y_pred >= 1).any():
        raise ValueError("y_pred must be between 0 and 1")
    if ((y_true != 0) & (y_true != 1)).any():
        raise ValueError("y_true must be 0 or 1")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
