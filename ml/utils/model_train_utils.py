import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score

from .feature_selection_utils import prepare_features


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q90(x):
    return x.quantile(0.9)


def _trust_interval(row, z=1.95):
    """Wilson-style trust interval for a Bernoulli proportion (lower, upper)."""
    sum_, val1 = row["total"], row["count"]
    val2 = sum_ - val1
    n = val1 + val2
    p = val1 / n
    low_bound = p - z * np.sqrt(p * (1 - p) / n)
    high_bound = p + z * np.sqrt(p * (1 - p) / n)
    return round(low_bound, 4), round(high_bound, 4)


def _profitable_hours_ti(df: pd.DataFrame, TI_low_bound: float) -> list:
    """Profitable hours by Trust Interval (TI).

    Pivots `df` by hour-of-day and target, computes the trust interval of the
    profitable ratio per hour, and keeps hours whose lower bound is at least
    `TI_low_bound`.
    """
    pvt = df[["target", "pattern", "time", "max_price_deviation"]].copy()
    pvt["hour"] = pvt["time"].dt.hour
    pvt = pvt.pivot_table(
        index=["hour", "target"],
        values=["pattern", "max_price_deviation"],
        aggfunc={
            "pattern": "count",
            "max_price_deviation": ["median", q10, q20, q30, q90],
        },
    ).reset_index()
    pvt.columns = [
        "hour",
        "target",
        "max_price_dev_q50",
        "max_price_dev_q10",
        "max_price_dev_q20",
        "max_price_dev_q30",
        "max_price_dev_q90",
        "pattern",
    ]
    pvt["total"] = pvt.groupby("hour")["pattern"].transform("sum")
    pvt = pvt.rename(columns={"pattern": "count"})
    pvt = pvt[pvt["target"] == 1]
    pvt["trust_interval"] = pvt.apply(_trust_interval, axis=1)
    mask = pvt["trust_interval"].apply(lambda x: x[0]) >= TI_low_bound
    return pvt.loc[mask, "hour"].tolist()


def _profitable_hours_rm(df: pd.DataFrame, RM_percent_above_0_5: float) -> list:
    """Profitable hours by Rolling Mean (RM).

    For each hour-of-day, takes the 168-period rolling mean of the target and
    keeps hours whose rolling mean stays above 0.5 for at least
    `RM_percent_above_0_5` percent of the time.
    """
    pvt = df[["time", "target"]].copy()
    pvt["hour"] = pvt["time"].dt.hour
    pvt = pvt.pivot_table(index="time", columns="hour", values="target", aggfunc="mean")

    results = []
    for hour in range(24):
        if hour in pvt.columns:
            valid_rolling = pvt[hour].dropna().rolling(window=168).mean().dropna()
            pct_above = (valid_rolling > 0.5).mean() * 100 if len(valid_rolling) > 0 else np.nan
        else:
            pct_above = np.nan
        results.append({"hour": hour, "percent_above_0_5": pct_above})

    summary = pd.DataFrame(results)
    return summary.query("percent_above_0_5 >= @RM_percent_above_0_5")["hour"].tolist()


def get_profitable_hours(
    df: pd.DataFrame, TI_low_bound: float, RM_percent_above_0_5: float
) -> Tuple[list, list]:
    """Compute profitable buy/sell hours from a (train) dataframe.

    For each ``ttype`` the hours found by Trust Interval (TI) and Rolling Mean (RM) are intersected,
    using the ``TI_low_bound`` and ``RM_percent_above_0_5`` thresholds.
    """
    df_buy = df[df["ttype"] == "buy"]
    df_sell = df[df["ttype"] == "sell"]
    buy_hours = sorted(
        set(_profitable_hours_rm(df_buy, RM_percent_above_0_5))
        & set(_profitable_hours_ti(df_buy, TI_low_bound))
    )
    sell_hours = sorted(
        set(_profitable_hours_rm(df_sell, RM_percent_above_0_5))
        & set(_profitable_hours_ti(df_sell, TI_low_bound))
    )
    return buy_hours, sell_hours


def _filter_idx_by_hours(df: pd.DataFrame, idx: list, buy_hours: list, sell_hours: list) -> list:
    """Keep only indices whose signal fired during a profitable hour for its ttype."""
    sub = df.loc[idx]
    hour = sub["time"].dt.hour
    buy_mask = (sub["ttype"] == "buy") & hour.isin(buy_hours)
    sell_mask = (sub["ttype"] == "sell") & hour.isin(sell_hours)
    return sub[buy_mask | sell_mask].index.tolist()


def load_train_data(train_path: str, test_time_days: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Load the training dataframe and compute the test cutoff date.

    Returns the dataframe together with the test cutoff date (``test_time_days`` before the last
    timestamp); rows after that date are held out as the test period. Profitable-hours filtering is
    applied per-fold on train data inside :func:`model_train`, not here.
    """
    train_df = pd.read_pickle(train_path).reset_index(drop=True)

    # all data for the last test_time_days days are test
    test_date = train_df["time"].max() - pd.to_timedelta(test_time_days, unit="D")

    return train_df, test_date


def load_selected_features(
    train_df: pd.DataFrame, params: dict, fi: pd.DataFrame
) -> Tuple[List[str], dict]:
    """Select model features from the feature-importance table.

    Consumes the feature-selection hyperparameters (``feature_num``, ``corr_thresh``) from
    ``params`` when present, so the returned ``params`` holds only LightGBM parameters.
    """
    if "feature_num" in params:
        feature_num = params.pop("feature_num")
        corr_thresh = params.pop("corr_thresh")

    features, feature_dict = prepare_features(train_df, fi, feature_num, corr_thresh)

    assert len(features) == len(set(features))

    return features, feature_dict


def prepare_model_params(
    train_df: pd.DataFrame, params: dict
) -> Tuple[dict, float, float, Optional[bool], float, float]:
    """Finalize LightGBM ``params`` and derive prediction bounds and sample weights.

    Consumes ``high_bound``/``low_bound``, ``sample_weight`` and the profitable-hours thresholds
    (``TI_low_bound``/``RM_percent_above_0_5``) from ``params`` so they don't leak into LightGBM,
    and sets the fixed LightGBM training options. When a ``sample_weight`` scheme ("cos" or
    "linear") is requested, a time-based ``weight`` column is added to ``train_df`` in place.

    Returns ``(params, high_bound, low_bound, sample_weight, TI_low_bound, RM_percent_above_0_5)``
    where ``sample_weight`` is ``True`` if weighting is enabled and ``None`` otherwise. The
    profitable-hours thresholds fall back to ``model_train``'s defaults when absent from ``params``
    (e.g. optuna rows produced before they were added to the search).
    """
    # set high and low bound for model predictions
    # p > high_bound -> 1, p < low_bound -> 0
    if "high_bound" in params:
        high_bound = params.pop("high_bound")
        del params["low_bound"]
    low_bound = 0

    # profitable-hours thresholds are model_train args, not LightGBM params
    TI_low_bound = params.pop("TI_low_bound", 0.5)
    RM_percent_above_0_5 = params.pop("RM_percent_above_0_5", 75)

    # add object weights
    if "sample_weight" in params:
        sample_weight = params.pop("sample_weight")
        train_df["weight"] = train_df["time"].astype(np.int64) / int(1e6)
    else:
        sample_weight = None

    if sample_weight == "cos":
        train_df["weight"] = (
            (train_df["weight"].max() - train_df["weight"])
            / (train_df["weight"].max() - train_df["weight"].min())
            * np.pi
            / 2
        )
        train_df["weight"] = np.cos(train_df["weight"])
        sample_weight = True
    elif sample_weight == "linear":
        train_df["weight"] = (train_df["weight"].max() - train_df["weight"]) / (
            train_df["weight"].max() - train_df["weight"].min()
        )
        sample_weight = True

    params["objective"] = "binary"
    params["verbosity"] = -1
    if params["boosting_type"] != "goss":
        params["subsample_freq"] = 1
    else:
        params["subsample"] = None
        params["subsample_freq"] = None
    params["importance_type"] = "gain"
    params["metric"] = "average_precison"

    return params, high_bound, low_bound, sample_weight, TI_low_bound, RM_percent_above_0_5


def conf_ppv_npv_acc_score(
    y: np.ndarray, oof: np.ndarray, _low_bound: float, high_bound: float
) -> Tuple[float, float, float]:
    """
    Consider only high confident objects and low confident
    objects for PPV and NPV score calculation
    """
    pred_conf = np.zeros_like(oof)
    pred_conf[oof >= high_bound] = 1
    pred_conf = pred_conf[(oof >= high_bound)]
    y_conf = y.values.reshape(-1, 1)[(oof >= high_bound)]
    if y_conf.shape[0] == 0:
        return 0, 0, 0
    return precision_score(y_conf, pred_conf), y_conf.shape[0], y_conf.shape[0] / y.shape[0]


def _fit_score_fold(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    fit_idx: list,
    val_idx: list,
    params: dict,
    sample_weight: Optional[Union[list, bool]],
    low_bound: float,
    high_bound: float,
    oof: np.ndarray,
    seen: set,
    conf_scores: list,
    conf_object_nums: list,
    callbacks: list,
    verbose: bool,
) -> lgb.LGBMClassifier:
    """Fit one fold, write its predictions into ``oof`` and append confident-object metrics.

    Only oof rows not already written by an earlier fold are filled (``seen`` tracks them); folds
    iterate newest-first, so a later, older-trained fold never overwrites a fresher prediction.
    The confident-object metrics are still computed on the fold's full validation window.
    """
    sw = df.loc[fit_idx, "weight"] if sample_weight is not None else None

    model_lgb = lgb.LGBMClassifier(**params)
    model_lgb.fit(
        X.iloc[fit_idx],
        y.iloc[fit_idx],
        sample_weight=sw,
        eval_set=[(X.iloc[fit_idx], y.iloc[fit_idx]), (X.iloc[val_idx], y.iloc[val_idx])],
        eval_metric="logloss",
        callbacks=callbacks,
    )

    val_preds = model_lgb.predict_proba(X.iloc[val_idx])
    # First-writer-wins: fill only rows not yet written by an earlier (more recent) fold, so an
    # older-trained fold can't overwrite a fresher prediction in the overlapping inner windows.
    novel_idx, novel_pos = [], []
    for pos, idx in enumerate(val_idx):
        if idx not in seen:
            novel_idx.append(idx)
            novel_pos.append(pos)
    oof[novel_idx, 0] = val_preds[novel_pos, 1]
    seen.update(novel_idx)

    val_score = log_loss(y.iloc[val_idx], val_preds)
    conf_score, conf_obj_num, conf_obj_pct = conf_ppv_npv_acc_score(
        y.iloc[val_idx], val_preds[:, 1], low_bound, high_bound
    )
    conf_scores.append(conf_score)
    conf_object_nums.append(conf_obj_num)

    if verbose:
        print(
            f"Logloss: {val_score}, Confident objects score: {conf_score}\n"
            f"Number of confident objects {conf_obj_num}, % of confident objects: {conf_obj_pct}\n"
            f"Number of profitable objects: {round((2 * conf_score - 1) * conf_obj_num)}"
        )

    return model_lgb


def _window_indices(
    time: pd.Series,
    df: pd.DataFrame,
    bybit_tickers: Optional[List[str]],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[list, list]:
    """Return (fit_idx, val_idx) for the given calendar train/validation windows."""
    fit_idx = time[(time > train_start) & (time <= train_end)].index.tolist()
    if bybit_tickers is not None:
        val_idx = time[
            (time > val_start) & (time <= val_end) & (df["ticker"].isin(bybit_tickers))
        ].index.tolist()
    else:
        val_idx = time[(time > val_start) & (time <= val_end)].index.tolist()
    return fit_idx, val_idx


def model_train(
    df: pd.DataFrame,
    features: List[str],
    params: dict,
    sample_weight: Optional[Union[list, bool]],
    n_folds: int,
    low_bound: float,
    high_bound: float,
    train_test: str,
    bybit_tickers: Optional[List[str]],
    loop: str = "outer",
    train_time_years: int = 2,
    test_time_days: Optional[int] = None,
    verbose: bool = False,
    TI_low_bound: float = 0.5,
    RM_percent_above_0_5: float = 75,
    profitable_hours_path: str = "data/context/profitable_hours.csv",
) -> Tuple[lgb.LGBMClassifier, list, list, np.ndarray, list]:
    """
    Train/validate model, return:
        - model
        - list of precisions for confident objects by folds (if train_test == "fold")
        - list of profitable objects by folds (if train_test == "fold")

    All windows are calendar-based and anchored on the dataset's last date; the gap between
    train and validation is 2 weeks. Each train window spans ``train_time_years`` years
    (default 2). When ``train_test == "fold"``, ``loop`` selects the cross-validation scheme:
        - "outer": ``n_folds`` folds whose 3-month validation windows tile backwards from the
          dataset's last date; each fold trains on the ``train_time_years`` years ending 2 weeks
          before its validation window.
        - "inner": nested validation. For each of the ``n_folds`` outer folds, build 3 inner
          folds whose 3-month validation windows slide back one calendar month each (for the
          default 2-year block: months 22-24, 21-23, 20-22 of the outer fold's train block);
          each inner fold trains on the ``train_time_years`` years ending 2 weeks before its
          validation window. Results are aggregated across all inner folds of all outer folds.
          Where windows overlap, oof keeps the prediction of the first (most recent) fold that
          validated each row.

    When ``train_test == "inference"`` and ``test_time_days`` is set, the last ``test_time_days`` days are
    held out as test data and the model is trained on the ``train_time_years`` years ending where
    that test period begins; if ``test_time_days`` is None, it trains on the ``train_time_years``
    years ending at the dataset's last date and appends the profitable hours found on that train
    window to ``profitable_hours_path`` (the CSV the live bot reads).
    """
    # Fold indices are used both positionally (X.iloc / oof) and by label (df.loc), so the
    # index must be a clean 0..n-1 range. Callers pass boolean-filtered slices, so reset it here.
    df = df.reset_index(drop=True)

    X, time = df[features], df["time"]
    y = df["target"]
    val_idxs = []
    seen: set = set()  # oof rows already written, so later folds don't overwrite fresher preds
    conf_scores = []
    conf_object_nums = []

    if train_test == "fold":
        oof = np.zeros([len(df), 1])
        last_date = time.max()

        if verbose:
            print(f"Training with {len(features)} features")

        if loop == "outer":
            for fold in range(n_folds):
                # Validation windows tile backwards from the dataset's last date.
                val_end = last_date - pd.DateOffset(months=3 * fold)
                val_start = val_end - pd.DateOffset(months=3)
                train_end = val_start - pd.Timedelta(weeks=2)
                train_start = train_end - pd.DateOffset(years=train_time_years)

                fit_idx, val_idx = _window_indices(
                    time, df, bybit_tickers, train_start, train_end, val_start, val_end
                )

                # determine profitable hours on this fold's train data and filter train/val by them
                buy_hours, sell_hours = get_profitable_hours(
                    df.loc[fit_idx], TI_low_bound, RM_percent_above_0_5
                )
                fit_idx = _filter_idx_by_hours(df, fit_idx, buy_hours, sell_hours)
                val_idx = _filter_idx_by_hours(df, val_idx, buy_hours, sell_hours)

                val_idxs.extend(val_idx)

                if verbose:
                    print(f"Fold #{fold + 1}")
                    print(y.iloc[val_idx].value_counts(normalize=True))
                    print(df.loc[val_idx[0], "time"])
                    print(df.loc[val_idx[-1], "time"])

                    plt.plot(
                        df.loc[fit_idx, "time"],
                        [fold + 1] * len(fit_idx),
                        label=f"Train {fold + 1}",
                        color="blue",
                    )
                    plt.plot(
                        df.loc[val_idx, "time"],
                        [fold + 1] * len(val_idx),
                        label=f"Test {fold + 1}",
                        color="red",
                    )

                    callbacks = [lgb.log_evaluation(100)]
                else:
                    callbacks = []

                model_lgb = _fit_score_fold(
                    X,
                    y,
                    df,
                    fit_idx,
                    val_idx,
                    params,
                    sample_weight,
                    low_bound,
                    high_bound,
                    oof,
                    seen,
                    conf_scores,
                    conf_object_nums,
                    callbacks,
                    verbose,
                )

        elif loop == "inner":
            plot_pos = 0
            for outer_fold in range(n_folds):
                # Inner folds validate within this outer fold's 2-year train block, which ends
                # 2 weeks before the outer validation window.
                outer_val_start = last_date - pd.DateOffset(months=3 * outer_fold + 3)
                anchor = outer_val_start - pd.Timedelta(weeks=2)

                for inner_fold in range(3):
                    val_end = anchor - pd.DateOffset(months=inner_fold)
                    val_start = val_end - pd.DateOffset(months=3)
                    train_end = val_start - pd.Timedelta(weeks=2)
                    train_start = train_end - pd.DateOffset(years=train_time_years)

                    fit_idx, val_idx = _window_indices(
                        time, df, bybit_tickers, train_start, train_end, val_start, val_end
                    )

                    # determine profitable hours on this fold's train data and filter train/val
                    buy_hours, sell_hours = get_profitable_hours(
                        df.loc[fit_idx], TI_low_bound, RM_percent_above_0_5
                    )
                    fit_idx = _filter_idx_by_hours(df, fit_idx, buy_hours, sell_hours)
                    val_idx = _filter_idx_by_hours(df, val_idx, buy_hours, sell_hours)

                    val_idxs.extend(val_idx)

                    if verbose:
                        print(f"Outer fold #{outer_fold + 1}, inner fold #{inner_fold + 1}")
                        print(y.iloc[val_idx].value_counts(normalize=True))
                        print(df.loc[val_idx[0], "time"])
                        print(df.loc[val_idx[-1], "time"])

                        plot_pos += 1
                        plt.plot(
                            df.loc[fit_idx, "time"],
                            [plot_pos] * len(fit_idx),
                            label=f"Train {plot_pos}",
                            color="blue",
                        )
                        plt.plot(
                            df.loc[val_idx, "time"],
                            [plot_pos] * len(val_idx),
                            label=f"Test {plot_pos}",
                            color="red",
                        )

                        callbacks = [lgb.log_evaluation(100)]
                    else:
                        callbacks = []

                    model_lgb = _fit_score_fold(
                        X,
                        y,
                        df,
                        fit_idx,
                        val_idx,
                        params,
                        sample_weight,
                        low_bound,
                        high_bound,
                        oof,
                        seen,
                        conf_scores,
                        conf_object_nums,
                        callbacks,
                        verbose,
                    )

        if verbose:
            plt.ylim(0.5, (n_folds if loop == "outer" else n_folds * 3) + 0.5)
            plt.xlabel("Time")
            plt.xticks(rotation=45)
            plt.ylabel("Fold")
            split_name = "Time-Series Split" if loop == "outer" else "Nested Time-Series Split"
            plt.title(f"Train/Test Distribution for {split_name}")
            plt.legend(["Train", "Test"], loc="lower right")
            plt.show()

        # Inner folds of different outer folds can validate overlapping calendar periods, so the
        # accumulated val_idxs may contain duplicates (oof holds the first/newest fold's prediction
        # for such rows). Collapse to unique indices so the pooled oof/backtest never double-counts.
        val_idxs = sorted(set(val_idxs))

        return model_lgb, conf_scores, conf_object_nums, oof, val_idxs

    elif train_test == "inference":
        # When test_time_days is set, hold out the last test_time_days days as test data and anchor the
        # train window on the end of the remaining (train) data; otherwise use all data.
        if test_time_days is not None:
            train_end = time.max() - pd.Timedelta(days=test_time_days)
        else:
            train_end = time.max()
        print(f"Train on the last {train_time_years} years of data up to {train_end}")
        train_start = train_end - pd.DateOffset(years=train_time_years)
        inf_mask = (time > train_start) & (time <= train_end)

        # determine profitable hours on the train window and filter the train rows by them
        buy_hours, sell_hours = get_profitable_hours(
            df.loc[inf_mask], TI_low_bound, RM_percent_above_0_5
        )
        train_idx = _filter_idx_by_hours(df, df.index[inf_mask].tolist(), buy_hours, sell_hours)

        X_inf, y_inf = df.loc[train_idx, features], df.loc[train_idx, "target"]
        sw = df.loc[train_idx, "weight"] if sample_weight is not None else None
        model_lgb = lgb.LGBMClassifier(**params)
        model_lgb.fit(
            X_inf,
            y_inf,
            sample_weight=sw,
            eval_set=[(X_inf, y_inf)],
            eval_metric="logloss",
            callbacks=[lgb.log_evaluation(100)],
        )

        # held-out test indices (after train_end), filtered by the same profitable hours so the
        # caller can backtest on the same hour set the model was trained on
        if test_time_days is not None:
            test_mask = time > train_end
            if bybit_tickers is not None:
                test_mask = test_mask & df["ticker"].isin(bybit_tickers)
            test_idx = _filter_idx_by_hours(df, df.index[test_mask].tolist(), buy_hours, sell_hours)
        else:
            test_idx = []

        # When training the final model on all data (no held-out test), persist the profitable
        # hours so the live bot (ml/inference.py) predicts only during them. Mirrors the row layout
        # of the former find_profitable_hours, minus the per-method (RM/TI) hour lists.
        if test_time_days is None:
            new_row = pd.DataFrame(
                [
                    {
                        "time": datetime.now(),
                        "test_date": None,
                        "TI_low_bound": TI_low_bound,
                        "RM_percent_above_0_5": RM_percent_above_0_5,
                        "profitable_buy_hours": buy_hours,
                        "profitable_sell_hours": sell_hours,
                    }
                ]
            )
            os.makedirs(os.path.dirname(profitable_hours_path), exist_ok=True)
            if os.path.exists(profitable_hours_path):
                new_row = pd.concat(
                    [pd.read_csv(profitable_hours_path), new_row], ignore_index=True
                )
            new_row.to_csv(profitable_hours_path, index=False)

        return model_lgb, [], [], np.array([]), np.array(test_idx)
