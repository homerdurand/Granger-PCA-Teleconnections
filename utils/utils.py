# ======================================================================
# Imports
# ======================================================================

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer


# ======================================================================
# Utilities
# ======================================================================

def generate_past_data(X, maxlag=1, inflag=0):
    """
    Generate lagged versions of an input time series.

    Parameters
    ----------
    X : np.ndarray
        Input array (time, features) or (time,)
    maxlag : int
        Maximum lag
    inflag : int
        Initial lag offset

    Returns
    -------
    X_past : np.ndarray
        Lagged predictor matrix
    X_present : np.ndarray
        Present-time target array
    """
    if len(X.shape) == 1:
        X = X[:, None]

    X_past = np.array([
        X[i + inflag:i + maxlag, :].flatten()
        for i in range(len(X) - maxlag)
    ])

    return X_past, X[maxlag:]


def compute_aic_bic(X, y, model):
    """
    Compute AIC and BIC for a fitted regression model.
    (Toy / placeholder implementation.)
    """
    model.fit(X, y)

    n = len(y)
    k = X.shape[1]

    residual = y - model.predict(X)
    sse = np.sum(residual ** 2)

    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)

    return aic, bic


# ======================================================================
# Lag analysis
# ======================================================================

def lag_analysis(
    X,
    Y,
    n_splits=10,
    maxlags=range(1, 21),
    model=LinearRegression()
):
    """
    Evaluate predictive skill and information criteria as a function of lag.

    Compares:
    - past Y → Y
    - past X → Y
    - past (X, Y) → Y
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for lag in tqdm(maxlags):

        # --------------------------------------------------------------
        # Generate lagged data
        # --------------------------------------------------------------

        Y_past, Y_present = generate_past_data(Y, lag, inflag=0)
        X_past, _ = generate_past_data(X, lag, inflag=0)

        combined = np.hstack([Y_past, X_past])

        # --------------------------------------------------------------
        # Model configurations
        # --------------------------------------------------------------

        configs = {
            "Y past → Y": Y_past,     # only past Y
            "X past → Y": X_past,     # only past X
            "XY past → Y": combined  # past X and Y
        }

        for name, X_data in configs.items():

            # Cross-validated negative R²
            r2_scores = -cross_val_score(
                model,
                X_data,
                Y_present,
                cv=kf,
                scoring="r2"
            )

            # Information criteria
            aic, bic = compute_aic_bic(X_data, Y_present, model)

            results.append({
                "lag": lag,
                "config": name,
                "r2_median": np.median(r2_scores),
                "r2_q25": np.percentile(r2_scores, 25),
                "r2_q75": np.percentile(r2_scores, 75),
                "AIC": aic,
                "BIC": bic
            })

    return pd.DataFrame(results)


# ======================================================================
# Plotting
# ======================================================================

def plot_lag_results(df):
    """
    Plot AIC/BIC and cross-validated negative R² as a function of lag.
    """

    configs = df["config"].unique()
    n_cfg = len(configs)

    # Two rows: (AIC/BIC) and (R²); columns = configurations
    fig, axes = plt.subplots(
        2,
        n_cfg,
        figsize=(3 * n_cfg, 5),
        sharex=True
    )

    # Ensure consistent indexing if only one configuration
    if n_cfg == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    legend_lines = []
    legend_labels = []

    for j, cfg in enumerate(configs):

        sub = df[df["config"] == cfg]

        # --------------------------------------------------------------
        # AIC / BIC (row 0)
        # --------------------------------------------------------------

        ax_ic = axes[0, j]

        l1, = ax_ic.plot(
            sub["lag"],
            sub["AIC"],
            marker="o",
            color="C0"
        )

        l2, = ax_ic.plot(
            sub["lag"],
            sub["BIC"],
            marker="s",
            color="C1"
        )

        ax_ic.set_title(f"AIC/BIC ({cfg})")
        ax_ic.set_xlabel("Lag")
        ax_ic.set_ylabel("Criterion Value")
        ax_ic.grid(True, linestyle="--", alpha=0.6)

        if j == 0:
            legend_lines.extend([l1, l2])
            legend_labels.extend(["AIC", "BIC"])

        # --------------------------------------------------------------
        # R² statistics (row 1)
        # --------------------------------------------------------------

        ax_r2 = axes[1, j]

        l3, = ax_r2.plot(
            sub["lag"],
            sub["r2_median"],
            marker="o",
            color="C2"
        )

        l4 = ax_r2.fill_between(
            sub["lag"],
            sub["r2_q25"],
            sub["r2_q75"],
            alpha=0.3,
            color="C3"
        )

        ax_r2.set_title(f"neg R² Scores ({cfg})")
        ax_r2.set_xlabel("Lag")
        ax_r2.set_ylabel("neg R²")
        ax_r2.grid(True, linestyle="--", alpha=0.6)

        if j == 0:
            legend_lines.extend([l3, l4])
            legend_labels.extend(["Median neg R²", "25–75 percentile"])

    # Global legend
    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        fontsize=9,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()