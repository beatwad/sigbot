from typing import Tuple, Optional

import numpy as np
from scipy.stats import shapiro, ttest_1samp, bartlett, levene, fligner


def test_normality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    if len(y_true) != len(y_pred):
        raise ValueError("Targets and predictions must have the same length")

    residuals = y_true - y_pred
    _, p_value = shapiro(residuals)

    return p_value, p_value < alpha



def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    if len(y_true) != len(y_pred):
        raise ValueError("Targets and predictions must have the same length")

    if prefer is None or prefer == "two-sided":
        alternative = "two-sided"
    elif prefer == "positive":
        alternative = "greater"
    elif prefer == "negative":
        alternative = "less"
    else:
        raise ValueError("prefer can be None or 'two-sided'\
                          or 'positive' or 'negative'")
    # compute residuals
    residuals = y_true - y_pred
    # compute p-value for t-test for the mean of one group of residuals
    _, p_value = ttest_1samp(residuals, 0, alternative=alternative)

    return p_value, p_value < alpha


def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    func_dict = {None: bartlett, "bartlett": bartlett,
                 "levene": levene, "fligner": fligner}

    if len(y_true) != len(y_pred):
        raise ValueError("Targets and predictions must have the same length")
    if test not in func_dict:
        raise ValueError("Unknown test value, it can be None or 'bartlett'\
                          or 'levene' or 'fligner'")

    # compute the residuals and sort them by y value
    sorted_idxs = y_true.argsort()
    residuals = y_true - y_pred
    residuals = residuals[sorted_idxs]

    # split residuals by bins
    bin_size = len(residuals) // bins
    residuals = [residuals[bin_size * i : bin_size * (i + 1)] 
                 for i in range(bins-1)] + [residuals[bin_size * (bins - 1):]]

    # perform homoscedasticity test
    test_func = func_dict[test]
    _, p_value = test_func(*residuals)

    return p_value, p_value < alpha
