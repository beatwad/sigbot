from typing import List, Optional, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score
from sklearn.model_selection import TimeSeriesSplit


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
    max_train_size: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[lgb.LGBMClassifier, list, list, np.ndarray, list]:
    """
    Train/validate model, return:
        - model
        - list of precisions for confident objects by folds (if train_test == "fold")
        - list of profitable objects by folds (if train_test == "fold")
    """
    X, time = df[features], df["time"]
    y = df["target"]
    val_idxs = []
    conf_scores = []
    conf_object_nums = []
    max_train_size = int(len(df) * max_train_size) if max_train_size is not None else None

    if train_test == "fold":
        oof = np.zeros([len(df), 1])

        tss = TimeSeriesSplit(
            gap=0,
            max_train_size=max_train_size,
            n_splits=n_folds,
            test_size=(len(df) * 2) // (n_folds * 3),
        )

        if verbose:
            print(f"Training with {len(features)} features")

        for fold, (fit_idx, val_idx) in enumerate(tss.split(time)):
            max_train_time = time[fit_idx].max() + pd.to_timedelta(96, unit="h")
            max_val_time = time[val_idx].max()

            if bybit_tickers is not None:
                val_idx = time[
                    (time > max_train_time)
                    & (time <= max_val_time)
                    & (df["ticker"].isin(bybit_tickers))
                ].index.tolist()
            else:
                val_idx = time[(time > max_train_time) & (time <= max_val_time)].index.tolist()
            val_idxs.extend(val_idx)

            if verbose:
                print(f"Fold #{fold + 1}")
                print(y.iloc[val_idx].value_counts(normalize=True))
                print(df.loc[val_idx[0], "time"])
                print(df.loc[val_idx[-1], "time"])

                plt.plot(
                    df.index[fit_idx],
                    [fold + 1] * len(fit_idx),
                    label=f"Train {fold + 1}",
                    color="blue",
                )
                plt.plot(
                    df.index[val_idx],
                    [fold + 1] * len(val_idx),
                    label=f"Test {fold + 1}",
                    color="red",
                )

                callbacks = [lgb.log_evaluation(100)]
            else:
                callbacks = []

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
            oof[val_idx, 0] = val_preds[:, 1]

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

        if verbose:
            plt.ylim(0.5, n_folds + 0.5)
            plt.xlabel("Index")
            plt.ylabel("Fold")
            plt.title("Train/Test Distribution for Time-Series Split")
            plt.legend(["Train", "Test"], loc="lower right")
            plt.show()

        return model_lgb, conf_scores, conf_object_nums, oof, val_idxs

    elif train_test == "inference":
        print("Train on the latest data")
        train_slice = df.iloc[-max_train_size:] if max_train_size else df
        X_inf, y_inf = train_slice[features], train_slice["target"]
        sw = train_slice["weight"] if sample_weight is not None else None
        model_lgb = lgb.LGBMClassifier(**params)
        model_lgb.fit(
            X_inf,
            y_inf,
            sample_weight=sw,
            eval_set=[(X_inf, y_inf)],
            eval_metric="logloss",
            callbacks=[lgb.log_evaluation(100)],
        )

        return model_lgb, [], [], np.array([]), np.array([])
