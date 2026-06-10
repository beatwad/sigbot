from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from shaphypetune import BoostBoruta
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from eli5.sklearn import PermutationImportance
except ImportError:
    PermutationImportance = None


def ppv_npv_acc(y_true, y_pred):
    """Calculate confusion matrix and return harmonic mean score of Positive Predictive Value (PPV, precisoin) and Negative Predictive Value (NPV)"""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return 0
    return (tp + tn) / (tp + fp + tn + fn + 1e-8)


def ppv_npv_acc_lgbm(y_true, y_pred):
    """Metric for LGBM"""
    return "ppv_npv_acc", ppv_npv_acc(y_true, y_pred), False


def boruta_selction(df, features, params, n_folds=4):
    boruta_df_ = pd.DataFrame()

    X, y, time = df[features], df["target"], df["time"]

    tss = TimeSeriesSplit(
        gap=0, max_train_size=None, n_splits=n_folds, test_size=(len(X) * 2) // (n_folds * 3)
    )

    # Stratify based on Class and Alpha (3 types of conditions)
    for fold, (train_idx, val_idx) in enumerate(tss.split(time)):
        print(f"Fold: {fold}")
        # Split the dataset according to the fold indexes.
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        clf = lgb.LGBMClassifier(**params)
        model = BoostBoruta(
            clf, importance_type="shap_importances", train_importance=False, max_iter=1000, perc=99
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="logloss",
                callbacks=[lgb.log_evaluation(100)],
            )
        except RuntimeError as re:
            print(re)
            break

        boruta_importance_df = pd.DataFrame(
            {"importance": model.ranking_}, index=X_train.columns
        ).sort_index()
        if boruta_df_.shape[0] == 0:
            boruta_df_ = boruta_importance_df.copy()
        else:
            boruta_df_ += boruta_importance_df

    boruta_df_ = boruta_df_.sort_values("importance")
    boruta_df_ = boruta_df_.reset_index().rename({"index": "Feature"}, axis=1)
    boruta_df_.to_csv("model/features/boruta_importances.csv", index=False)
    return boruta_df_


def lgbm_tuning(df, features, params, bybit_tickers, n_folds=4, n_repeats=1, permut=False):
    outer_cv_score = []

    perm_df_ = pd.DataFrame()
    feature_importances_ = pd.DataFrame()

    X, y, time = df[features], df["target"], df["time"]

    for repeat in range(n_repeats):
        print(f"Repeat #{repeat + 1}")

        tss = TimeSeriesSplit(
            gap=0, max_train_size=None, n_splits=n_folds, test_size=(len(X) * 2) // (n_folds * 3)
        )

        oof = np.zeros(len(df))

        for fold, (fit_idx, val_idx) in enumerate(tss.split(time)):
            if fold == 0:
                first_val_idx = val_idx[0]

            max_train_time = time[fit_idx].max() + pd.to_timedelta(96, unit="h")
            max_val_time = time[val_idx].max()
            val_idx = time[
                (time > max_train_time)
                & (time <= max_val_time)
                & (df["ticker"].isin(bybit_tickers))
            ].index.tolist()

            X_train = X.iloc[fit_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[fit_idx]
            y_val = y.iloc[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="logloss",
                callbacks=[lgb.log_evaluation(100)],
            )

            val_preds = clf.predict_proba(X_val)[:, 1]
            val_score = log_loss(y_val, val_preds)

            oof[val_idx] = val_preds
            best_iter = clf.best_iteration_

            print(f"Fold: {fold + 1:>3} | loss: {val_score:.5f} | Best iteration: {best_iter:>4}")

            f_i = pd.DataFrame(
                sorted(zip(clf.feature_importances_, X.columns), reverse=True, key=lambda x: x[1]),
                columns=["Value", "Feature"],
            )

            if feature_importances_.shape[0] == 0:
                feature_importances_ = f_i.copy()
            else:
                feature_importances_["Value"] += f_i["Value"]

            if permut:
                if PermutationImportance is None:
                    raise ImportError("eli5 is required for permutation importance")
                perm = PermutationImportance(
                    clf, scoring=None, n_iter=1, random_state=42, cv=None, refit=False
                ).fit(X_val, y_val)

                perm_importance_df = pd.DataFrame(
                    {"importance": perm.feature_importances_}, index=X_val.columns
                ).sort_index()

                if perm_df_.shape[0] == 0:
                    perm_df_ = perm_importance_df.copy()
                else:
                    perm_df_ += perm_importance_df

        outer_cv = log_loss(y[first_val_idx:], oof[first_val_idx:])
        outer_cv_score.append(outer_cv)

    print(f"Outer Holdout avg score: log_loss: {np.mean(outer_cv_score):.5f}")
    print(f"{'*' * 50}\n")

    if permut:
        perm_df_ = perm_df_.sort_values("importance", ascending=False)
        perm_df_ = perm_df_.reset_index().rename({"index": "Feature"}, axis=1)

    feature_importances_ = feature_importances_.sort_values("Value", ascending=False).reset_index(
        drop=True
    )

    feature_importances_.to_csv("model/features/lgbm_feature_importances.csv", index=False)

    if permut:
        perm_df_.to_csv("model/features/permutation_importances.csv", index=False)

    return perm_df_, feature_importances_, np.mean(outer_cv_score)


def rfe_selection(df, features):
    rfe_params = {
        "penalty": "l2",
        "max_iter": 10000,
        "C": 1,
    }

    scaler = StandardScaler()
    X, y = df[features], df["target"]
    X = scaler.fit_transform(X)

    estimator = LogisticRegression(**rfe_params)
    selector = RFECV(estimator, min_features_to_select=50, step=0.025, cv=4, verbose=1)
    selector = selector.fit(X, y)
    rfe_df_ = pd.DataFrame({"importance": selector.ranking_}, index=features).sort_index()
    rfe_df_ = rfe_df_.reset_index().rename({"index": "Feature"}, axis=1)
    rfe_df_.to_csv("model/features/rfe_importances.csv", index=False)
    return rfe_df_


def exclude_corr_features(df, fi, features, corr_thresh):
    features_to_select = features.copy()
    correlations = (
        df.loc[:, features_to_select]
        .corr()
        .abs()
        .unstack()
        .sort_values(kind="quicksort", ascending=False)
        .reset_index()
    )
    correlations = correlations[correlations["level_0"] != correlations["level_1"]]
    correlations.columns = ["feature_1", "feature_2", "corr"]

    correlations = pd.merge(
        left=correlations, right=fi, how="left", left_on="feature_1", right_on="Feature"
    )
    correlations = correlations.drop(columns="Feature")
    correlations = correlations.sort_values(["corr", "rank"], ascending=[False, True])
    correlations = correlations[::2]

    features_to_exclude = set()
    correlations = correlations[correlations["corr"] > corr_thresh]

    for _, row in correlations.iterrows():
        feature_1 = row["feature_1"]
        feature_2 = row["feature_2"]

        if feature_1 in features_to_exclude:
            continue

        features_to_exclude.add(feature_2)

    return features_to_exclude


def prepare_features(df, fi, feature_num, corr_thresh):
    """Get features, sort them by their time appearance and return for using in train and inference."""
    fi = fi["Feature"]
    fi = fi[:feature_num]
    feature_dict = defaultdict(list)
    features = list()

    for f in fi:
        if f == "volume_24":
            feature_dict[0].append(f)
            continue
        period = f.split("_")
        if period[-1].isdigit() and period[-2] == "prev":
            feature_dict[int(period[-1])].append("_".join(period[:-2]))
        else:
            feature_dict[0].append(f)

    feature_dict = dict(sorted(feature_dict.items()))

    for item in feature_dict.items():
        if item[0] > 0:
            features.extend([i + f"_prev_{item[0]}" for i in item[1]])
        else:
            features.extend([i for i in item[1]])

    features_to_exclude = exclude_corr_features(
        df,
        fi.reset_index(drop=True).to_frame(name="Feature").assign(rank=range(len(fi))),
        features,
        corr_thresh,
    )
    features = [f for f in features if f not in features_to_exclude]

    feature_dict["features"] = features

    for item in feature_dict.items():
        if not isinstance(item[0], int):
            continue

        features_to_remove = list()

        for f in item[1]:
            if item[0] > 0:
                f_ = f"{f}_prev_{item[0]}"
            else:
                f_ = f

            if f_ not in features:
                assert f_ in features_to_exclude
                features_to_remove.append(f)

        feature_dict[item[0]] = [f for f in feature_dict[item[0]] if f not in features_to_remove]

    empty_list_keys = [key for key in feature_dict if not feature_dict[key]]
    for key in empty_list_keys:
        del feature_dict[key]

    return features, feature_dict


def remove_feature_selection(df, test_date, features, params, high_bound, max_train_size):
    """Remove features that significantly decrease fold score."""
    from tqdm.auto import tqdm

    from ml.utils.model_train_utils import model_train

    _, best_scores, _, _, _ = model_train(
        df[df["time"] < test_date],
        features,
        params,
        None,
        n_folds=5,
        low_bound=0,
        high_bound=high_bound,
        train_test="fold",
        bybit_tickers=None,
        max_train_size=max_train_size,
        verbose=False,
    )

    features_to_remove = []

    for feature in tqdm(features):
        _, scores, _, _, _ = model_train(
            df[df["time"] < test_date],
            [f for f in features if f != feature],
            params,
            None,
            n_folds=5,
            low_bound=0,
            high_bound=high_bound,
            train_test="fold",
            bybit_tickers=None,
            max_train_size=max_train_size,
            verbose=False,
        )
        p_value = ttest_rel(scores, best_scores, alternative="greater").pvalue
        if p_value <= 0.2:
            features_to_remove.append(feature)
            print(feature)

    return features_to_remove
