import os
from typing import Callable, List

import numpy as np
import optuna
import pandas as pd
from scipy.stats import ttest_rel

from ml.utils.feature_selection_utils import prepare_features
from ml.utils.model_test_utils import backtest
from ml.utils.model_train_utils import conf_ppv_npv_acc_score, model_train

# Cache for the per-trial feature correlation matrix. Keyed on the feature set only: delete
# this file if you rebuild the training data without changing the top-feature list.
CORR_MATRIX_PATH = "model/optuna/corr_matrix.pkl"


def _load_or_compute_corr_matrix(
    train_df: pd.DataFrame, corr_candidates: List[str]
) -> pd.DataFrame:
    """Return the abs-correlation matrix for ``corr_candidates``, cached on disk.

    Reuses the cached matrix when it exists and was built on the same set of features;
    otherwise recomputes it and overwrites the cache.
    Recompute takes ~30 sec for 600 features and ~100k rows
    """
    if os.path.exists(CORR_MATRIX_PATH):
        cached = pd.read_pickle(CORR_MATRIX_PATH)
        if set(cached.columns) == set(corr_candidates):
            return cached

    corr_matrix = train_df[corr_candidates].corr().abs()
    os.makedirs(os.path.dirname(CORR_MATRIX_PATH), exist_ok=True)
    corr_matrix.to_pickle(CORR_MATRIX_PATH)
    return corr_matrix


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
        optuna_df = pd.read_csv("model/optuna/optuna_lgbm.csv")
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
    TP: float,
    SL: float,
    n_folds: int = 8,
    optimize_alpha: float = 0.2,
    min_precision: float = 0.5,
) -> Callable[[optuna.trial.Trial], float]:
    """Return an Optuna objective function for LightGBM hyperparameter tuning."""

    # Pairwise feature correlations don't change between trials (only corr_thresh / feature_num
    # do), so compute the abs-correlation matrix once over the largest candidate set the search
    # can reach (top `max_feature_num` ranked features) and reuse it in every trial. The matrix
    # is cached on disk and only recomputed when missing or built on a different feature set.
    max_feature_num = 600
    corr_candidates = [f for f in fi["Feature"][:max_feature_num] if f in train_df.columns]
    corr_matrix = _load_or_compute_corr_matrix(train_df, corr_candidates)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": trial.suggest_categorical("boosting_type", ["dart", "goss", "gbdt"]),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True),
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

        # goss is not supported by the CUDA backend; run it on CPU, GPU otherwise
        # params["device_type"] = "cpu" if params["boosting_type"] == "goss" else "cuda"

        if params["boosting_type"] != "goss":
            params["subsample"] = trial.suggest_float("subsample", 0.3, 0.9)

        if params["is_unbalance"] is True:
            params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
        else:
            params["class_weight"] = None

        high_bound = params.pop("high_bound")
        low_bound = params.pop("low_bound")
        corr_thresh = params.pop("corr_thresh")
        sample_weight = params.pop("sample_weight")
        feature_num = params.pop("feature_num")

        # Only deep-copy train_df when we need to attach a weight column; otherwise read it
        # directly (prepare_features and model_train do not mutate it).
        if sample_weight:
            df = train_df.copy()
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
        else:
            df = train_df

        features, _ = prepare_features(df, fi, feature_num, corr_thresh, corr_matrix=corr_matrix)

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
        backtest_result, _ = backtest(df, oof_slice, val_idxs, high_bound, TP, SL)
        result = backtest_result * oof_conf_score

        scores = [
            conf_object_num * (conf_score - min_precision) * 100
            for conf_object_num, conf_score in zip(conf_object_nums, conf_scores)
        ]

        df_optuna_more_info = pd.read_csv("model/optuna/optuna_lgbm_info.csv")
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
        df_optuna_more_info.to_csv("model/optuna/optuna_lgbm_info.csv", index=False)
        return result

    return objective
