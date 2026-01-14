# ======================================================================
# Imports
# ======================================================================

import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from utils.utils import (
    lag_analysis,
    generate_past_data,
    plot_lag_results
)


# ======================================================================
# Granger PCA
# ======================================================================

class GrangerPCA:
    """
    Granger-causal principal component analysis.

    Identifies directions in Y that are maximally Granger-caused by X
    using likelihood-ratio–type criteria.
    """

    def __init__(
        self,
        maxlag=None,
        maxlags=range(1, 21),
        lag_criterion='AIC',
        method='difference',
        model_res=LinearRegression(),
        model_full=LinearRegression(),
        alpha=0
    ):
        self.method = method
        self.maxlag = maxlag
        self.maxlags = maxlags
        self.lag_criterion = lag_criterion
        self.model_res = model_res
        self.model_full = model_full
        self.alpha = alpha

        return None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, Y, Z=None, plot_lag_analysis=True, verbose=True):
        """
        Fit the GrangerPCA model.

        Optionally selects the optimal lag using information criteria.
        """

        # --------------------------------------------------------------
        # Lag selection
        # --------------------------------------------------------------

        if self.maxlag is None:
            results = lag_analysis(X, Y, n_splits=10, maxlags=self.maxlags)
            self.maxlag = (
                results[results['config'] == 'XY past → Y'][self.lag_criterion].argmin()
                + 1
            )

            if plot_lag_analysis:
                plot_lag_results(results)

        else:
            if plot_lag_analysis:
                print('Parameter maxlag should be set to None to search for optimal lags')

        # --------------------------------------------------------------
        # Generate lagged data
        # --------------------------------------------------------------

        X_past, _ = generate_past_data(X, maxlag=self.maxlag, inflag=0)
        Y_past, Y_present = generate_past_data(Y, maxlag=self.maxlag, inflag=0)

        if Z is not None:
            Z_past, Z_present = generate_past_data(Z, maxlag=self.maxlag, inflag=0)

        # Dimensions
        self.N, self.d = Y_present.shape
        self.p = X_past.shape[1]
        self.rank = np.min([self.d, self.p])

        # --------------------------------------------------------------
        # Restricted and full models
        # --------------------------------------------------------------

        if Z is not None:
            predictors_res = np.hstack((Y_past, Z_past))
            predictors_full = np.hstack((X_past, Y_past, Z_past))
        else:
            predictors_res = Y_past
            predictors_full = np.hstack((X_past, Y_past))

        self.model_res.fit(predictors_res, Y_present)
        self.model_full.fit(predictors_full, Y_present)

        # --------------------------------------------------------------
        # Residuals and covariance matrices
        # --------------------------------------------------------------

        residuals_res = Y_present - self.model_res.predict(predictors_res)
        residuals_full = Y_present - self.model_full.predict(predictors_full)

        Sigma_res = np.cov(residuals_res.T)
        Sigma_full = np.cov(residuals_full.T)

        self.likelihood_ratio_ = (
            np.sqrt(np.linalg.det(Sigma_res))
            / np.sqrt(np.linalg.det(Sigma_full))
        )

        # --------------------------------------------------------------
        # Eigenvalue decomposition
        # --------------------------------------------------------------

        try:
            if self.method == 'ratio':
                u, W = scipy.linalg.eigh(
                    Sigma_res,
                    Sigma_full + self.alpha * np.identity(Sigma_full.shape[0])
                )

            if self.method == 'diff_ratio':
                u, W = scipy.linalg.eigh(
                    Sigma_res - Sigma_full,
                    Sigma_full + self.alpha * np.identity(Sigma_full.shape[0])
                )

            elif self.method == 'difference':
                u, W = np.linalg.eig(Sigma_res - Sigma_full)

        except Exception:
            print("The GEV could not be fitted!")

        idx = np.argsort(u)[::-1][:self.rank]

        # --------------------------------------------------------------
        # Store eigenvalues and eigenvectors
        # --------------------------------------------------------------

        self.eigs_ = u[idx]
        self.eigs_ratio_ = u[idx] / np.sum(u[idx])

        # Initial sign assignment (overwritten below; kept unchanged)
        self.signs = np.sign(
            np.corrcoef(
                np.hstack((X, Y @ W[:, idx])).T
            )[0, 1:]
        )

        # Final sign assignment
        self.signs = np.sign(self.model_full.coef_[:self.p, :].mean())

        self.vects_ = W[:, idx] * self.signs
        self.vects_null_ = W[:, np.argsort(u)[::-1][self.rank:]]

        # --------------------------------------------------------------
        # Output summary
        # --------------------------------------------------------------

        results = pd.DataFrame({
            "eigs": self.eigs_
        })

        if verbose:
            print(results)

    # ------------------------------------------------------------------
    # Statistical test
    # ------------------------------------------------------------------

    def test(self, alpha=0.05, verbose=True, show_plot=True, correction=None):
        """
        Perform Granger causality tests for all retained eigenvalues.
        """

        df1 = self.d
        df2 = self.N - self.p - self.d - 1

        eigs = np.array(self.eigs_)
        n_tests = len(eigs)

        # Multiple-testing correction
        if correction == 'Bonferroni':
            adj_alpha = alpha / n_tests
        else:
            adj_alpha = alpha

        # F-statistics and p-values
        F_stats = np.abs((df2 / df1) * eigs)
        P_vals = scipy.stats.f.sf(F_stats, df1, df2)

        results = pd.DataFrame({
            "eigen_index": np.arange(1, n_tests + 1),
            "eigenvalue": eigs,
            "F_statistic": F_stats,
            "df1": df1,
            "df2": df2,
            "p_value": P_vals,
            "alpha_used": adj_alpha,
            "interpretation": [
                f"Reject H₀ (α={alpha:.5f}): X Granger-causes Y"
                if p < adj_alpha else
                "Fail to reject H₀: no Granger causality"
                for p in P_vals
            ]
        })

        # --------------------------------------------------------------
        # Verbose output
        # --------------------------------------------------------------

        if verbose:
            print("=== Granger Causality Test Results ===\n")

            if correction:
                print(
                    f"Multiple-testing correction applied: {correction} "
                    f"(adjusted α = {adj_alpha:.5f})\n"
                )
            else:
                print(f"No multiple-testing correction applied (α = {alpha})\n")

            print("--- F tests per eigenvalue ---")

            for _, row in results.iterrows():
                print(f"Eigenvalue {int(row['eigen_index'])}:")
                print(f"  eigenvalue     : {row['eigenvalue']:.6f}")
                print(f"  F_statistic    : {row['F_statistic']:.6f}")
                print(f"  df1, df2       : ({df1}, {df2})")
                print(f"  p_value        : {row['p_value']:.6g}")
                print(f"  interpretation : {row['interpretation']}\n")

        # --------------------------------------------------------------
        # Optional visualization
        # --------------------------------------------------------------

        if show_plot:

            plt.figure(figsize=(5, 3))

            x_max = np.max(F_stats) * 1.5
            x = np.linspace(0, x_max, 500)
            pdf = scipy.stats.f.pdf(x, df1, df2)

            plt.plot(x, pdf, lw=2, color='black', label='F-distribution PDF')

            F_crit = scipy.stats.f.ppf(1 - adj_alpha, df1, df2)
            plt.vlines(
                x=F_crit,
                ymin=0,
                ymax=np.max(pdf),
                label='α/m' if correction else 'α',
                color='grey',
                alpha=0.3,
                lw=1.8
            )

            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(F_stats))]

            for i, (Fv, color) in enumerate(zip(F_stats, colors)):
                plt.axvline(
                    Fv,
                    linestyle='--',
                    color=color,
                    lw=1.8,
                    label=f'F{i+1} = {Fv:.3f}',
                    alpha=0.9
                )

            plt.xlabel("F value")
            plt.ylabel("Density")
            plt.grid(alpha=0.3)
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.4),
                ncol=min(4, len(F_stats)),
                fontsize=8,
                frameon=False
            )

            plt.tight_layout()
            plt.show()

        return results

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def transform(self, Y):
        components = Y @ self.vects_
        components = (
            components - components.mean(axis=0)
        ) / components.std(axis=0)
        return components

    def decompose(self, Y):
        P_dec = (
            self.vects_
            @ np.linalg.inv(self.vects_.T @ self.vects_)
            @ self.vects_.T
        )
        return Y @ P_dec

    def fit_transform(self, X, Y, plot_lag_analysis=False, verbose=False):
        self.fit(X, Y, plot_lag_analysis=plot_lag_analysis, verbose=verbose)
        return self.transform(Y)
