"""Solution for boosting uncertainty problem"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from lightgbm import LGBMClassifier


@dataclass
class PredictionDict:
    """Class for storing model uncertainty"""

    pred: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    pred_virt: np.ndarray = field(default_factory=lambda: np.array([]))
    lcb: np.ndarray = field(default_factory=lambda: np.array([]))
    ucb: np.ndarray = field(default_factory=lambda: np.array([]))


def virtual_ensemble_iterations(model: LGBMClassifier, k: int = 20) -> List[int]:
    """
    Define number of trees in each model of virtual ensemble

    Parameters
    ----------
    model: LGBMClassifier
        lightgbm classifier model

    k: int
        the number of virtual ensembles (Default value = 20)

    Returns
    -------
    """
    n_estimators = model.n_estimators_
    first_iter = n_estimators // 2 - 1
    iterations = list(range(first_iter, n_estimators, k))
    return iterations


def virtual_ensemble_predict(
    model: LGBMClassifier, X: np.ndarray, k: int = 20
) -> np.ndarray:
    """
    Make prediction for each virtual ensemble

    Parameters
    ----------
    model: LGBMClassifier
        lightgbm classifier model

    X: np.ndarray
        input train data array

    k: int
        the number of virtual ensembles (Default value = 20)

    Returns
    -------
    """
    iterations = virtual_ensemble_iterations(model, k)
    stage_preds = np.array(
        [model.predict_proba(X, num_iteration=i)[:, -1] for i in iterations]
    )
    stage_preds = stage_preds.T
    return stage_preds


def predict_with_uncertainty(
    model: LGBMClassifier, X: np.ndarray, k: int = 20
) -> PredictionDict:
    """Make prediction and calculate model uncertainty
    and lower/upper uncertainty bound for each object

    Parameters
    ----------
    model: LGBMClassifier
        lightgbm classifier model

    X: np.ndarray
        input train data array

    k: int
        the number of virtual ensembles (Default value = 20)

    Returns
    -------
    """
    stage_preds = virtual_ensemble_predict(model, X, k)

    uncertainty = stage_preds.var(axis=1)
    pred_virt = stage_preds.mean(axis=1)

    std = np.sqrt(uncertainty)
    lcb = pred_virt - 3 * std
    ucb = pred_virt + 3 * std

    prediction_dict = PredictionDict(stage_preds, uncertainty, pred_virt, lcb, ucb)
    return prediction_dict
