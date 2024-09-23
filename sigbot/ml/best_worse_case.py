"""Возвращает топ К объектов с наименьшей и наибольшей ошибками."""

from typing import Optional

import numpy as np
import pandas as pd
import residuals


def best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    if mask is not None:
        y_test = y_test[mask].copy()
        y_pred = y_pred[mask].copy()

    func = func or "residuals"
    resid_func = getattr(residuals, func)
    resid = np.abs(resid_func(y_test, y_pred))
    idxs = resid.argsort()[:top_k]

    if mask is not None:
        idxs = np.array(y_test.index)[idxs]

    result = {
        "X_test": X_test.loc[idxs],
        "y_test": y_test[idxs],
        "y_pred": y_pred[idxs],
        "resid": resid[idxs],
    }

    return result

def worst_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k worst cases according to the given function"""
    if mask is not None:
        y_test = y_test[mask].copy()
        y_pred = y_pred[mask].copy()

    func = func or "residuals"
    resid_func = getattr(residuals, func)
    resid = np.abs(resid_func(y_test, y_pred))
    idxs = resid.argsort()[::-1][:top_k]

    if mask is not None:
        idxs = np.array(y_test.index)[idxs]

    result = {
        "X_test": X_test.loc[idxs],
        "y_test": y_test[idxs],
        "y_pred": y_pred[idxs],
        "resid": resid[idxs],
    }

    return result
