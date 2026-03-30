# ======================================================================
# Imports
# ======================================================================

import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.utils import generate_past_data


# ======================================================================
# Parametric Granger causality test
# ======================================================================

def test_granger_causality(
    X,
    Y,
    model_res=LinearRegression(),
    model_full=LinearRegression(),
    alpha=0.05,
    maxlag=1,
    correction='Bonferroni'
):
    """
    Tests Granger causality from X to each column of Y.

    If Y is univariate, performs a simple Granger test without controlling
    for other columns.
    If Y is multivariate, controls for all other columns of Y_present.

    Parameters
    ----------
    X : np.ndarray
        Explanatory variable(s)
    Y : np.ndarray
        Dependent variable(s)
    alpha : float, optional
        Significance level (default=0.05)
    maxlag : int or list of ints
        Maximum lag(s) to test
    correction : str, optional
        Multiple testing correction method ('Bonferroni' or 'none')

    Returns
    -------
    pd.DataFrame
        Columns:
        ['lag', 'Y_column', 'F_statistic', 'df1', 'df2',
         'p_value', 'alpha_used', 'interpretation']
    """

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    # Allow multiple lags
    if not isinstance(maxlag, (list, tuple, np.ndarray)):
        maxlag = [maxlag]

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    all_results = []

    # ------------------------------------------------------------------
    # Loop over lags
    # ------------------------------------------------------------------

    for lag in maxlag:

        # --------------------------------------------------------------
        # Generate lagged predictors
        # --------------------------------------------------------------

        X_past, _ = generate_past_data(X, maxlag=lag, inflag=0)
        Y_past, Y_present = generate_past_data(Y, maxlag=lag, inflag=0)

        n_tests = Y_present.shape[1]
        n = Y_present.shape[0]

        F_stats, P_vals, df1s, df2s = [], [], [], []

        # --------------------------------------------------------------
        # Univariate Y case
        # --------------------------------------------------------------

        if Y.shape[1] == 1:

            y_target = Y_present[:, 0]

            # Restricted model: Y lags only
            predictors_res = Y_past

            # Full model: X lags + Y lags
            predictors_full = np.hstack((X_past, Y_past))

            # Fit models
            lr_res = model_res.fit(predictors_res, y_target)
            lr_full = model_full.fit(predictors_full, y_target)

            # Residual sum of squares
            rss_res = np.sum((y_target - lr_res.predict(predictors_res)) ** 2)
            rss_full = np.sum((y_target - lr_full.predict(predictors_full)) ** 2)

            # Degrees of freedom
            p_res = predictors_res.shape[1]
            p_full = predictors_full.shape[1]
            df1 = p_full - p_res
            df2 = n - p_full

            # F-statistic and p-value
            F_stat = (df2 / df1) * (rss_res - rss_full) / rss_full
            p_val = scipy.stats.f.sf(F_stat, df1, df2)

            # Critical value
            F_crit = scipy.stats.f.ppf(1 - alpha, df1, df2)

            # Multiple-testing correction (kept as in original code)
            if correction.lower() == 'bonferroni':
                adj_alpha = alpha
            else:
                adj_alpha = alpha

            results = pd.DataFrame({
                "lag": [lag],
                "Y_column": [1],
                "F_statistic": [F_stat],
                "df1": [df1],
                "df2": [df2],
                "p_value": [p_val],
                "F_critical": [F_crit],
                "alpha_used": [adj_alpha],
                "interpretation": [
                    f"Reject H₀ (α={alpha:.5f}): X Granger-causes Y (lag={lag})"
                    if p_val < adj_alpha else
                    f"Fail to reject H₀: no Granger causality for Y (lag={lag})"
                ]
            })

            all_results.append(results)
            continue

        # --------------------------------------------------------------
        # Multivariate Y case
        # --------------------------------------------------------------

        for i in range(n_tests):

            y_target = Y_present[:, i]

            # Control for all other Y variables
            Y_controls = np.delete(Y_present, i, axis=1)

            # Restricted and full predictors
            predictors_res = np.hstack((Y_past, Y_controls))
            predictors_full = np.hstack((X_past, Y_past, Y_controls))

            # Fit models
            lr_res = LinearRegression().fit(predictors_res, y_target)
            lr_full = LinearRegression().fit(predictors_full, y_target)

            # Residual sum of squares
            rss_res = np.sum((y_target - lr_res.predict(predictors_res)) ** 2)
            rss_full = np.sum((y_target - lr_full.predict(predictors_full)) ** 2)

            # Degrees of freedom
            p_res = predictors_res.shape[1]
            p_full = predictors_full.shape[1]
            df1 = p_full - p_res
            df2 = n - p_full

            # F-statistic and p-value
            F_stat = (df2 / df1) * (rss_res - rss_full) / rss_full
            p_val = scipy.stats.f.sf(F_stat, df1, df2)

            F_stats.append(F_stat)
            P_vals.append(p_val)
            df1s.append(df1)
            df2s.append(df2)

        # --------------------------------------------------------------
        # Multiple-testing correction
        # --------------------------------------------------------------

        if correction.lower() == 'bonferroni':
            adj_alpha = alpha / n_tests
        else:
            adj_alpha = alpha

        results = pd.DataFrame({
            "lag": lag,
            "Y_column": np.arange(1, n_tests + 1),
            "F_statistic": F_stats,
            "df1": df1s,
            "df2": df2s,
            "p_value": P_vals,
            "F_critical": F_crit,
            "alpha_used": adj_alpha,
            "interpretation": [
                f"Reject H₀ (α={alpha:.5f}): X Granger-causes Y_col_{i+1} (lag={lag})"
                if p < adj_alpha else
                f"Fail to reject H₀: no Granger causality for Y_col_{i+1} (lag={lag})"
                for i, p in enumerate(P_vals)
            ]
        })

        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


# ======================================================================
# Parametric + non-parametric (permutation) Granger causality test
# ======================================================================

def test_granger_causality_non_parametric(
    X,
    Y,
    model_res=LinearRegression(),
    model_full=LinearRegression(),
    alpha=0.05,
    maxlag=1,
    correction='Bonferroni',
    non_parametric=True,
    n_permutations=1000,
    random_state=None
):
    """
    Tests Granger causality from X to each column of Y.
    Optionally computes a non-parametric (permutation-based) p-value.
    """

    # ------------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------------

    if random_state is not None:
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    if not isinstance(maxlag, (list, tuple, np.ndarray)):
        maxlag = [maxlag]

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    all_results = []

    # ------------------------------------------------------------------
    # Loop over lags
    # ------------------------------------------------------------------

    for lag in tqdm(maxlag):

        # Generate lagged predictors
        X_past, _ = generate_past_data(X, maxlag=lag, inflag=0)
        Y_past, Y_present = generate_past_data(Y, maxlag=lag, inflag=0)

        n_tests = Y_present.shape[1]
        n = Y_present.shape[0]

        F_stats = []
        pvals_param = []
        pvals_nonparam = []
        df1s = []
        df2s = []

        # --------------------------------------------------------------
        # Loop over Y variables
        # --------------------------------------------------------------

        for i in range(n_tests):

            y_target = Y_present[:, i]
            Y_controls = np.delete(Y_present, i, axis=1)

            predictors_res = np.hstack((Y_past, Y_controls))
            predictors_full = np.hstack((X_past, Y_past, Y_controls))

            # Restricted model
            lr_res = model_res.fit(predictors_res, y_target)
            rss_res = np.sum((y_target - lr_res.predict(predictors_res)) ** 2)

            # Full model
            lr_full = model_full.fit(predictors_full, y_target)
            rss_full = np.sum((y_target - lr_full.predict(predictors_full)) ** 2)

            # Degrees of freedom
            p_res = predictors_res.shape[1]
            p_full = predictors_full.shape[1]
            df1 = p_full - p_res
            df2 = n - p_full

            # Parametric F-test
            F_stat = (df2 / df1) * (rss_res - rss_full) / rss_full
            p_val_param = scipy.stats.f.sf(F_stat, df1, df2)

            # ----------------------------------------------------------
            # Non-parametric permutation test
            # ----------------------------------------------------------

            if non_parametric:

                null_F = np.zeros(n_permutations)

                for perm in range(n_permutations):

                    X_perm = np.random.permutation(X_past)

                    predictors_full_perm = np.hstack((X_perm, Y_past, Y_controls))
                    predictors_res_perm = np.hstack((Y_past, Y_controls))

                    lr_full_perm = model_full.fit(predictors_full_perm, y_target)
                    rss_full_perm = np.sum(
                        (y_target - lr_full_perm.predict(predictors_full_perm)) ** 2
                    )

                    lr_res_perm = model_full.fit(predictors_res_perm, y_target)
                    rss_res_perm = np.sum(
                        (y_target - lr_res_perm.predict(predictors_res_perm)) ** 2
                    )

                    null_F[perm] = (df2 / df1) * (rss_res_perm - rss_full_perm) / rss_full_perm

                p_val_nonparam = np.mean(null_F >= F_stat)

            else:
                p_val_nonparam = np.nan

            F_stats.append(F_stat)
            pvals_param.append(p_val_param)
            pvals_nonparam.append(p_val_nonparam)
            df1s.append(df1)
            df2s.append(df2)

        # --------------------------------------------------------------
        # Multiple-testing correction
        # --------------------------------------------------------------

        adj_alpha = alpha / n_tests if correction.lower() == 'bonferroni' else alpha

        results = pd.DataFrame({
            "lag": lag,
            "Y_column": np.arange(1, n_tests + 1),
            "F_statistic": F_stats,
            "df1": df1s,
            "df2": df2s,
            "p_value_parametric": pvals_param,
            "p_value_nonparametric": pvals_nonparam,
            "alpha_used": adj_alpha,
            "interpretation": [
                f"Reject H₀ (α={alpha:.5f}): X Granger-causes Y_col_{i+1} (lag={lag})"
                if pvals_param[i] < adj_alpha else
                f"Fail to reject H₀: no Granger causality for Y_col_{i+1} (lag={lag})"
                for i in range(n_tests)
            ]
        })

        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)
