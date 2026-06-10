from typing import Tuple

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
    if max_num_simult_trades > 0:
        backtest_df = cap_max_num_simult_trades(backtest_df, max_num_simult_trades)
    backtest_df = backtest_df.reset_index(drop=True)
    backtest_df["balance"] = 1.0
    backtest_df["free_balance"] = 1.0
    backtest_df["profit"] = 0.0
    backtest_df["trade_profit"] = 0.0
    backtest_df["quantity"] = 0.0
    backtest_df["profit_count"] = 0

    not_closed_trade_idxs = set()

    generator = (
        tqdm(backtest_df.iterrows(), total=len(backtest_df))
        if show_progress
        else backtest_df.iterrows()
    )

    for i, row in generator:
        j = i - 1
        if j >= 0:
            balance = backtest_df.loc[j, "balance"]
            free_balance = backtest_df.loc[j, "free_balance"]
        else:
            balance = free_balance = 1

        signal_time = row["time"]
        target = row["target"]
        first_price = row["first_price"]
        last_price = row["last_price"]

        closed_trade_idxs = []

        for j in not_closed_trade_idxs:
            prev_signal_close_time = backtest_df.loc[j, "close_time"]
            if signal_time >= prev_signal_close_time:
                free_balance += backtest_df.loc[j, "trade_profit"]
                backtest_df.loc[i, "trade_profit"] += backtest_df.loc[j, "trade_profit"]
                backtest_df.loc[i, "profit_count"] += 1
                closed_trade_idxs.append(j)

        for j in closed_trade_idxs:
            not_closed_trade_idxs.remove(j)

        if free_balance >= min_free_balance * balance:
            quantity = free_balance * risk / (sl * leverage)
            balance -= quantity * open_comission
            free_balance -= quantity * (1 + open_comission)
            profit, trade_profit = calculate_profit(
                target, quantity, first_price, last_price, tp, sl
            )
        else:
            profit, trade_profit = 0, 0
            quantity = 0

        backtest_df.loc[i, "quantity"] = quantity
        backtest_df.loc[i, "balance"] = balance + profit
        backtest_df.loc[i, "free_balance"] = free_balance
        backtest_df.loc[i, "profit"] = profit
        backtest_df.loc[i, "trade_profit"] = trade_profit

        not_closed_trade_idxs.add(i)

    if len(backtest_df) > 0:
        result = round(backtest_df["balance"].iloc[-1] * 100, 2)
    else:
        result = 0
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
