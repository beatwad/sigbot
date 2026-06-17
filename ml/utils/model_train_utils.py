from typing import List, Optional, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score

from .feature_selection_utils import prepare_features


def load_train_data(train_path: str, test_time_days: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Load the training dataframe and compute the test cutoff date.

    Returns the dataframe together with the test cutoff date (``test_time_days`` before the last
    timestamp); rows after that date are held out as the test period.
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
) -> Tuple[dict, float, float, Optional[bool]]:
    """Finalize LightGBM ``params`` and derive prediction bounds and sample weights.

    Consumes ``high_bound``/``low_bound``, ``sample_weight`` from ``params`` so they don't leak into LightGBM,
    and sets the fixed LightGBM training options. When a ``sample_weight`` scheme ("cos" or
    "linear") is requested, a time-based ``weight`` column is added to ``train_df`` in place.

    Returns ``(params, high_bound, low_bound, sample_weight)``
    where ``sample_weight`` is ``True`` if weighting is enabled and ``None`` otherwise.
    """
    # set high and low bound for model predictions
    # p > high_bound -> 1, p < low_bound -> 0
    if "high_bound" in params:
        high_bound = params.pop("high_bound")
        del params["low_bound"]
    low_bound = 0

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

    return params, high_bound, low_bound, sample_weight


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
    years ending at the dataset's last date.
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
        train_idx = df.index[inf_mask].tolist()

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

        # held-out test indices (after train_end) for the caller to backtest on
        if test_time_days is not None:
            test_mask = time > train_end
            if bybit_tickers is not None:
                test_mask = test_mask & df["ticker"].isin(bybit_tickers)
            test_idx = df.index[test_mask].tolist()
        else:
            test_idx = []

        return model_lgb, [], [], np.array([]), np.array(test_idx)
