"""
Significance tests for comparing estimators against a reference estimator.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def compare_to_reference(df: pd.DataFrame, estimator: str, reference: str, value_col: str, sample_col: str, paired: bool) -> float:
    """
    Returns the p-value of a two-sided t-test of `estimator` against `reference`.

    If `paired`, samples are matched by `sample_col` (i.e., estimators ran on the same spanning trees)
    and a paired t-test is used. Otherwise, Welch's t-test is used.
    NaN samples are ignored.
    """
    values = df[df.estimator == estimator].set_index(sample_col)[value_col]
    ref_values = df[df.estimator == reference].set_index(sample_col)[value_col]
    if paired:
        joined = pd.concat([values, ref_values], axis=1, keys=['est', 'ref']).dropna()
        return ttest_rel(joined.est, joined.ref).pvalue
    return ttest_ind(values, ref_values, equal_var=False, nan_policy='omit').pvalue

def holm(p_values: dict) -> dict:
    """
    Holm-Bonferroni correction. Returns adjusted p-values for the given dict of p-values.
    NaN p-values are ignored and stay NaN.
    """
    keys = [key for key, p in p_values.items() if not np.isnan(p)]
    keys.sort(key=lambda key: p_values[key])
    m = len(keys)
    adjusted = {key: np.nan for key in p_values}
    running_max = 0
    for i, key in enumerate(keys):
        running_max = max(running_max, min(1, (m - i) * p_values[key]))
        adjusted[key] = running_max
    return adjusted
