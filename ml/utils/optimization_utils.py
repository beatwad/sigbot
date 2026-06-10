from typing import Callable, List

import numpy as np
import optuna
import pandas as pd
from scipy.stats import ttest_rel

from ml.utils.feature_selection_utils import prepare_features
from ml.utils.model_test_utils import backtest
from ml.utils.model_train_utils import conf_ppv_npv_acc_score, model_train


def load_best_params(row_num: int = 0) -> dict:
    """
    Load the best parameters from the Optuna study.

    Parameters
    ----------
    row_num : int, optional
        The row number to load the parameters from (default is 0).

    Returns
    -------
    dict
        A dictionary containing the best parameters for the LightGBM model.
    """
    try:
        optuna_df = pd.read_csv("optuna/optuna_lgbm.csv")
    except FileNotFoundError:
        return {}
    columns = [c for c in optuna_df.columns if c.startswith("params_")]
    row = optuna_df[columns].iloc[row_num]
    params = {key[7:]: value for key, value in row.to_dict().items()}
    if params["sample_weight"] != params["sample_weight"]:
        params["sample_weight"] = None
    if params["is_unbalance"] is True:
        if "class_weight" not in params or params["class_weight"] != params["class_weight"]:
            params["class_weight"] = None
        else:
            params["class_weight"] = "balanced"
    return params


def make_objective(
    train_df: pd.DataFrame,
    test_date: pd.Timestamp,
    fi: pd.DataFrame,
    bybit_tickers: List[str],
    tp: float,
    sl: float,
    n_folds: int = 8,
    optimize_alpha: float = 0.2,
    min_precision: float = 0.5,
) -> Callable[[optuna.trial.Trial], float]:
    """Return an Optuna objective function for LightGBM hyperparameter tuning."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precison",
            "boosting_type": trial.suggest_categorical("boosting_type", ["dart", "goss", "gbdt"]),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 3e-1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 4, 512),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "max_bin": trial.suggest_int("max_bin", 32, 255),
            "subsample_freq": 1,
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
            "verbose": -1,
            "importance_type": "gain",
            "high_bound": trial.suggest_float("high_bound", 0.3, 0.65),
            "low_bound": trial.suggest_float("low_bound", 0.0, 0.1),
            "feature_num": trial.suggest_int("feature_num", 30, 600),
            "corr_thresh": trial.suggest_float("corr_thresh", 0.5, 0.99),
            "sample_weight": trial.suggest_categorical("sample_weight", [None, "cos", "linear"]),
        }

        if params["boosting_type"] != "goss":
            params["subsample"] = trial.suggest_float("subsample", 0.3, 0.9)

        if params["is_unbalance"] == "True":
            params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
        else:
            params["class_weight"] = None

        high_bound = params.pop("high_bound")
        low_bound = params.pop("low_bound")
        corr_thresh = params.pop("corr_thresh")
        sample_weight = params.pop("sample_weight")
        feature_num = params.pop("feature_num")

        df = train_df.copy()
        if sample_weight:
            df["weight"] = df["time"].astype(np.int64) / int(1e6)
            if sample_weight == "cos":
                df["weight"] = (
                    (df["weight"].max() - df["weight"])
                    / (df["weight"].max() - df["weight"].min())
                    * np.pi
                    / 2
                )
                df["weight"] = np.cos(df["weight"])
            else:
                df["weight"] = (df["weight"].max() - df["weight"]) / (
                    df["weight"].max() - df["weight"].min()
                )
            sample_weight = True

        features, _ = prepare_features(df, fi, feature_num, corr_thresh)

        _, conf_scores, conf_object_nums, oof, val_idxs = model_train(
            df[df["time"] < test_date],
            features,
            params,
            sample_weight,
            n_folds=n_folds,
            low_bound=low_bound,
            high_bound=high_bound,
            train_test="fold",
            bybit_tickers=bybit_tickers,
            loop="inner",
            verbose=False,
        )

        y = df["target"].iloc[val_idxs]
        oof_slice = oof[val_idxs]
        oof_conf_score, oof_conf_obj_num, _ = conf_ppv_npv_acc_score(
            y, oof_slice, low_bound, high_bound
        )
        backtest_result, _ = backtest(df, oof_slice, val_idxs, high_bound, tp, sl)
        result = backtest_result * oof_conf_score

        scores = [
            conf_object_num * (conf_score - min_precision) * 100
            for conf_object_num, conf_score in zip(conf_object_nums, conf_scores)
        ]

        df_optuna_more_info = pd.read_csv("ml/model/optuna/optuna_lgbm_info.csv")
        profit_objects = round(oof_conf_obj_num * (2 * oof_conf_score - 1))

        if df_optuna_more_info.shape[0] > 0:
            best_result, best_scores_raw = df_optuna_more_info.query("result == result.max()")[
                ["result", "scores"]
            ].values[0]
            best_scores = [float(b) for b in best_scores_raw[1:-1].split(", ")]
            if result > best_result:
                p_value = ttest_rel(scores, best_scores, alternative="greater").pvalue
                if p_value >= optimize_alpha:
                    print(
                        f"avg conf score {result} is better than best score {best_result}, "
                        f"but p-value {p_value} is more than alpha {optimize_alpha}"
                    )
                    result = best_result + np.abs(result - best_result) * (1 - p_value)

        tmp = pd.DataFrame(
            {
                "result": [result],
                "backtest_result": [backtest_result],
                "oof_conf_score": [oof_conf_score],
                "profit_objects": [profit_objects],
                "oof_conf_obj_num": [oof_conf_obj_num],
                "scores": [scores],
            }
        )

        df_optuna_more_info = pd.concat([df_optuna_more_info, tmp])
        df_optuna_more_info.to_csv("ml/model/optuna/optuna_lgbm_info.csv", index=False)
        return result

    return objective
