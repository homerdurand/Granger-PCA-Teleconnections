"""
Synthetic Experiments
=====================
Benchmarking teleconnection methods on toy models: weight-pattern recovery
and Granger causality testing.

Usage
-----
    python synthetic.py experiment_number N m d u_values [u_values ...]

Arguments
---------
experiment_number : int
    1  → Experiment 1 only  (single-run: weight pattern recovery + p-values)
    2  → Experiment 2 only  (grid search over N and m: Type I error & Power)
    12 → Both experiments

N : int
    Time-series length (used by Experiment 1; also sets N_max for Experiment 2)

m : int
    Latent dimension — must be divisible by 5 (Experiment 1) or sets m_max
    for the grid search (Experiment 2)

d : int
    Observed Y dimension

u_values : float(s)
    One or more causal link strengths X → Y.
    • Experiment 1 uses the first value supplied.
    • Experiment 2 uses all supplied values (power sweep); u = 0 is always
      included automatically for the Type I error run.

Examples
--------
    # Experiment 1 only
    python synthetic.py 1 2000 5 20 0.2

    # Experiment 2 only (grid search)
    python synthetic.py 2 500 500 20 0.1 0.2 0.4

    # Both experiments
    python synthetic.py 12 2000 500 20 0.2 0.1 0.4
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")

# ── Data ──────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ── Progress ──────────────────────────────────────────────────────────────────
from tqdm import tqdm

# ── Local package — GPCA + teleconnection baselines ───────────────────────────
from models.granger_pca import GrangerPCA
from models.teleconnection_models import (
    RegressionAnalysis,
    CompositeAnalysis,
    CanonicalCorrelationAnalysis,
    SpatialAveraging,
    PrincipalComponents,
)
from src.granger_causality_test import test_granger_causality_mbb, test_granger_causality
from src.functions_synthetic_experiements import run_experiment, generate_m_values

np.random.seed(2025)

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = "../figures/synthetic"
GRID_FIG_DIR = "../figures/synthetic/grid_experiments"

# ── Publication figure style ──────────────────────────────────────────────────
PUB_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

METHOD_COLORS = {
    "GPCA": "#2c7bb6",
    "CCA": "#d7191c",
    "Composites": "#fdae61",
    "Regression": "#abd9e9",
    "PCA": "#12a15a",
    "Spatial Average": "#353210",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def logistic_func(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def _norm(w):
    """Normalise weight vector to [0, 1]."""
    w = np.abs(np.asarray(w)).ravel()
    return w / w.max()


def plot_rejection_heatmaps(rejection_df, alpha, title, save_path):
    """Plot per-method rejection-rate heatmaps over the (N, m) grid."""
    heatmap_rc = {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 24,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 1.5,
    }

    methods = sorted(rejection_df["Method"].unique())
    with mpl.rc_context(heatmap_rc):
        fig, axes = plt.subplots(
            1,
            len(methods),
            figsize=(6.5 * len(methods), 7.5),
            sharex=True,
            sharey=True,
        )
        if len(methods) == 1:
            axes = [axes]

        cmap = plt.get_cmap("viridis")
        im = None

        for ax, method in zip(axes, methods):
            df_m = rejection_df[rejection_df["Method"] == method]
            pivot = df_m.pivot(index="m", columns="N", values="rejection_rate").sort_index()

            im = ax.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(method, pad=15, fontweight="bold")
            ax.set_xlabel(r"Sample size $N$", labelpad=10)
            ax.set_ylabel(r"Dimensionality $m$", labelpad=10)
            ax.tick_params(axis="both", which="major", width=1.5, length=6)

        cbar = fig.colorbar(
            im, ax=axes, orientation="horizontal", fraction=0.08, pad=-0.3
        )
        cbar.set_label(rf"Rejection rate ($\alpha = {alpha}$)", fontsize=24)
        cbar.ax.tick_params(labelsize=26, width=1.5, length=6)

        fig.suptitle(title, fontsize=28, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}.png", dpi=400, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Single-run toy model
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment1(N, m, d, u, train_ratio=0.80, lag=3,
                    lag_range=range(1, 5), lag_criterion="BIC"):
    """
    Weight-pattern recovery and Granger causality p-values on a single
    synthetic realisation of the toy model.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1 — Single-run toy model")
    print(f"  N={N}, m={m}, d={d}, u={u}")
    print(f"{'='*60}\n")

    assert m % 5 == 0, "m must be divisible by 5 (one block per latent component A–E)"

    t = np.arange(1, N + 1)

    # ── Driving processes ─────────────────────────────────────────────────────
    X  = np.random.normal(0, 1, (N, 1)) + np.sin(0.2 * t)[:, None] + np.cos(0.1 * t)[:, None]
    Z1 = np.random.normal(0, 1, (N, 1)) + np.cos(0.4 * t)[:, None]
    Z2 = np.random.normal(0, 1, (N, 1)) + np.cos(0.05 * t)[:, None]

    # ── Block indices (5 latent components) ──────────────────────────────────
    block = m // 5
    slices = {
        "A": slice(0,         block),
        "B": slice(block,     2 * block),
        "C": slice(2 * block, 3 * block),
        "D": slice(3 * block, 4 * block),
        "E": slice(4 * block, 5 * block),
    }
    A, B, C, D, E = slices["A"], slices["B"], slices["C"], slices["D"], slices["E"]

    # ── Latent dynamics ───────────────────────────────────────────────────────
    W = np.random.randn(N, m)
    for i in range(3, N):
        W[i, A] += (0.2 * W[i-2, A] + u * 0.4 * X[i-1, 0] + u * 0.3 * X[i-2, 0]
                    + u * 0.2 * X[i-3, 0] + 0.2 * Z1[i-1, 0] + 0.2 * Z1[i-2, 0])
        W[i, B] += 0.4 * W[i-1, B] + 0.5 * Z1[i-1, 0]
        W[i, C] += 0.1 * W[i-2, C] + 0.8 * Z1[i-2, 0] + 0.8 * W[i-1, A] + 0.4 * W[i-2, A]
        W[i, D] += 0.1 * W[i-2, D] - 0.7 * Z2[i-3, 0] + 0.8 * W[i-2, C]
        W[i, E] += 0.1 * W[i-1, E] + 0.8 * Z1[i-2, 0] + 0.6 * Z2[i-1, 0] + 0.2 * Z2[i-2, 0]

    Y = W

    # ── Train / test split ────────────────────────────────────────────────────
    train_size = int(N * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    print(f"Train: {train_size}  |  Test: {N - train_size}  |  Y shape: {Y.shape}")

    # ── Model fitting ─────────────────────────────────────────────────────────
    gpca = GrangerPCA(
        maxlag=lag,
        maxlags=lag_range,
        method="diff_ratio",
        lag_criterion=lag_criterion,
    )
    gpca.fit(X_train, Y_train, Z=None, plot_lag_analysis=True, verbose=True)

    ca  = CompositeAnalysis(maxlag=gpca.maxlag)
    ra  = RegressionAnalysis(maxlag=gpca.maxlag)
    cca = CanonicalCorrelationAnalysis(maxlag=gpca.maxlag, n_components=1)
    sa  = SpatialAveraging()
    pc  = PrincipalComponents()

    ca.fit(X_train, Y_train)
    ra.fit(X_train, Y_train)
    cca.fit(X_train, Y_train)
    sa.fit(X_train, Y_train)
    pc.fit(X_train, Y_train)

    print(f"\nSelected lag: {gpca.maxlag}")

    # ── Granger causality tests ───────────────────────────────────────────────
    selected_lag = gpca.maxlag

    cp_gpca = gpca.transform(Y_test)
    cp_ca   = ca.transform(Y_test, type="PosNeg")
    cp_ra   = ra.transform(Y_test)
    cp_cca  = cca.transform(Y_test)
    cp_sa   = sa.transform(Y_test)
    cp_pc   = pc.transform(Y_test)

    results_gpca = test_granger_causality(X_test[:, None], cp_gpca[:, 0], maxlag=[selected_lag])
    results_ra   = test_granger_causality(X_test[:, None], cp_ra,         maxlag=[selected_lag])
    results_ca   = test_granger_causality(X_test[:, None], cp_ca,         maxlag=[selected_lag])
    results_cca  = test_granger_causality(X_test[:, None], cp_cca,        maxlag=[selected_lag])
    results_sa   = test_granger_causality(X_test[:, None], cp_sa,         maxlag=[selected_lag])
    results_pc   = test_granger_causality(X_test[:, None], cp_pc,         maxlag=[selected_lag])

    gc_table = pd.concat(
        [
            results_gpca[results_gpca["lag"] == selected_lag],
            results_cca[results_cca["lag"]   == selected_lag],
            results_ca[results_ca["lag"]     == selected_lag],
            results_ra[results_ra["lag"]     == selected_lag],
            results_sa[results_sa["lag"]     == selected_lag],
            results_pc[results_pc["lag"]     == selected_lag],
        ],
        keys=["GPCA", "CCA", "Composites", "Regression", "Spatial Average", "PCA"],
    ).reset_index(level=0).rename(columns={"level_0": "Method"})

    print("\nGranger causality results:")
    print(gc_table[["Method", "F_statistic", "p_value"]].to_string(index=False))

    # ── Weight pattern heatmap ────────────────────────────────────────────────
    weights_gpca = _norm(gpca.vects_[:, 0])
    weights_ra   = _norm(ra.lr.coef_)
    weights_ca   = _norm(ca.composites["PosNeg"].to_numpy())
    weights_cca  = _norm(cca.cca.y_loadings_[:, 0])
    weights_pc   = _norm(pc.pca.components_[0, :])

    df_weights = pd.DataFrame(
        np.column_stack([weights_gpca, weights_ra, weights_ca, weights_cca, weights_pc]),
        columns=["GPCA", "RA", "CA", "CCA", "PCA"],
    )

    with mpl.rc_context(PUB_RC):
        sns.set_theme(context="paper", style="white", font="serif", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            df_weights, ax=ax, cmap="magma",
            cbar_kws={"shrink": 0.7, "label": "Normalised Weight"},
            linewidths=0.0, yticklabels=False,
        )

        block_size = len(df_weights) / 5
        ax.set_yticks([block_size * (i + 0.5) for i in range(5)])
        ax.set_yticklabels(["A", "B", "C", "D", "E"], rotation=0, fontsize=11)
        ax.set_xlabel("Method")
        ax.set_ylabel("Latent block")

        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/weights_toy_model.pdf", bbox_inches="tight")
        plt.savefig(
            f"{FIG_DIR}/weights_toy_model.png",
            dpi=300, bbox_inches="tight", transparent=True,
        )
        plt.close()
        print(f"Weight heatmap saved → {FIG_DIR}/weights_toy_model.pdf")

    return gpca, df_weights, gc_table


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Grid search over N and m
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment2(N_max, m_max, d, u_values,
                    alpha=0.05, B=2, n_grid=10,
                    N_min=100, m_min=5):
    """
    Type I error and power grid search over (N, m).
    """
    print(f"\n{'='*60}")
    print(f"Experiment 2 — Grid search over N and m")
    print(f"  N_min={N_min}, N_max={N_max}, m_min={m_min}, m_max={m_max}")
    print(f"  d={d}, alpha={alpha}, B={B}, n_grid={n_grid}")
    print(f"  u_values={u_values}")
    print(f"{'='*60}\n")

    N_VALUES = np.unique(
        np.logspace(np.log10(N_min), np.log10(N_max), n_grid).astype(int)
    )
    M_VALUES = generate_m_values(n_points=n_grid, m_min=m_min, m_max=m_max)

    # ── Type I error (u = 0) ──────────────────────────────────────────────────
    results_null = []
    print("Running Type I error experiments (u = 0)...")
    for _ in tqdm(range(B)):
        for N in N_VALUES:
            for m in M_VALUES:
                results_df, _ = run_experiment(N=N, m=m, d=d, u=0)
                for _, row in results_df.iterrows():
                    results_null.append({
                        "N":       N,
                        "m":       m,
                        "Method":  row["Method"],
                        "reject":  row["p_value"] <= alpha,
                        "p_value": row["p_value"],
                        "snr":     row["snr"],
                    })

    rejection_null = (
        pd.DataFrame(results_null)
        .groupby(["N", "m", "Method"])["reject"]
        .mean()
        .reset_index()
        .rename(columns={"reject": "rejection_rate"})
    )
    rejection_null.to_csv(f"{GRID_FIG_DIR}/rejection_rates_null.csv", index=False)
    print(f"Type I error results saved → {GRID_FIG_DIR}/rejection_rates_null.csv")

    # ── Power (u > 0) ─────────────────────────────────────────────────────────
    all_power_results = []
    for u_val in u_values:
        results_power = []
        print(f"Running power experiments (u = {u_val})...")
        for _ in tqdm(range(B)):
            for N in N_VALUES:
                for m in M_VALUES:
                    results_df, _ = run_experiment(N=N, m=m, d=d, u=u_val)
                    for _, row in results_df.iterrows():
                        results_power.append({
                            "u":       u_val,
                            "N":       N,
                            "m":       m,
                            "Method":  row["Method"],
                            "reject":  row["p_value"] <= alpha,
                            "p_value": row["p_value"],
                            "snr":     row["snr"],
                        })

        rejection_power_u = (
            pd.DataFrame(results_power)
            .groupby(["u", "N", "m", "Method"])["reject"]
            .mean()
            .reset_index()
            .rename(columns={"reject": "rejection_rate"})
        )
        csv_name = f"rejection_rates_power_u_{str(u_val).replace('.', 'p')}.csv"
        rejection_power_u.to_csv(f"{GRID_FIG_DIR}/{csv_name}", index=False)
        print(f"  Power results saved → {GRID_FIG_DIR}/{csv_name}")
        all_power_results.append(rejection_power_u)

    rejection_power_all = pd.concat(all_power_results, ignore_index=True)

    # ── Heatmap plots ─────────────────────────────────────────────────────────
    plot_rejection_heatmaps(
        rejection_null,
        alpha=alpha,
        title="Type I Error",
        save_path=f"{GRID_FIG_DIR}/rejection_heatmaps_null",
    )

    for u_val in u_values:
        subset = rejection_power_all[rejection_power_all["u"] == u_val]
        plot_rejection_heatmaps(
            subset,
            alpha=alpha,
            title=f"Power  (u = {u_val})",
            save_path=(
                f"{GRID_FIG_DIR}/rejection_heatmaps_power_u_"
                f"{str(u_val).replace('.', 'p')}"
            ),
        )

    return rejection_null, rejection_power_all


# ─────────────────────────────────────────────────────────────────────────────
# Final composite figure (requires both experiments to have been run)
# ─────────────────────────────────────────────────────────────────────────────

def plot_composite_figure(df_weights, rejection_power_all):
    """
    Two-panel publication figure:
      (A) Normalised weight patterns
      (B) True positive rate vs SNR with logistic fits
    """
    df_snr = rejection_power_all.copy()

    with mpl.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel A — weight heatmap
        ax_A = axes[0]
        sns.heatmap(
            df_weights, ax=ax_A, cmap="magma",
            cbar_kws={"shrink": 0.75, "label": "Normalised weight"},
            linewidths=0.0, yticklabels=False,
        )
        block_size = len(df_weights) / 5
        ax_A.set_yticks([block_size * (i + 0.5) for i in range(5)])
        ax_A.set_yticklabels(["A", "B", "C", "D", "E"], rotation=0, fontsize=10)
        ax_A.set_xlabel("Method")
        ax_A.set_ylabel("Latent block")
        ax_A.set_title("(A) Weight pattern recovery")

        # Panel B — TPR vs SNR with logistic fits
        ax_B = axes[1]
        for method in df_snr["Method"].unique():
            m_data = df_snr[df_snr["Method"] == method].sort_values("snr")
            x = m_data["snr"].values
            y = m_data["rejection_rate"].values
            color = METHOD_COLORS.get(method, "gray")

            try:
                p0 = [max(y), 1.0, float(np.median(x)), float(min(y))]
                popt, _ = curve_fit(logistic_func, x, y, p0=p0, maxfev=5000)
                x_fit = np.linspace(x.min(), x.max(), 500)
                ax_B.scatter(x, y, color=color, alpha=0.35, s=25, edgecolors="none")
                ax_B.plot(
                    x_fit,
                    logistic_func(x_fit, *popt),
                    color=color,
                    label=f"{method} ($x_0$={popt[2]:.2f})",
                )
            except RuntimeError:
                ax_B.scatter(
                    x, y, color=color, alpha=0.5, s=25,
                    edgecolors="none", label=method,
                )

        ax_B.set_xlim(0, 0.1)
        ax_B.set_xlabel("Signal-to-Noise Ratio (SNR)")
        ax_B.set_ylabel("True Positive Rate")
        ax_B.set_title("(B) Power vs SNR")
        ax_B.legend(frameon=False, loc="lower right", fontsize=9)
        ax_B.spines[["top", "right"]].set_visible(False)
        ax_B.tick_params(direction="out", length=3)

        plt.tight_layout()
        plt.savefig(
            f"{FIG_DIR}/synthetic_experiments_publication.pdf",
            dpi=300, bbox_inches="tight",
        )
        plt.savefig(
            f"{FIG_DIR}/synthetic_experiments_publication.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"Final figure saved → {FIG_DIR}/synthetic_experiments_publication.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment_number",
        type=str,
        choices=["1", "2", "12"],
        help="Which experiment(s) to run: 1, 2, or 12 (both).",
    )
    parser.add_argument("N",  type=int, help="Time-series length.")
    parser.add_argument("m",  type=int, help="Latent dimension (Exp 1: must be divisible by 5).")
    parser.add_argument("d",  type=int, help="Observed Y dimension.")
    parser.add_argument(
        "u_values",
        type=float,
        nargs="+",
        help="Causal link strength(s) X → Y. Exp 1 uses the first value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_exp1 = "1" in args.experiment_number
    run_exp2 = "2" in args.experiment_number

    os.makedirs(FIG_DIR,      exist_ok=True)
    os.makedirs(GRID_FIG_DIR, exist_ok=True)

    df_weights          = None
    rejection_power_all = None

    if run_exp1:
        u1 = args.u_values[0]
        _, df_weights, _ = run_experiment1(
            N=args.N, m=args.m, d=args.d, u=u1
        )

    if run_exp2:
        _, rejection_power_all = run_experiment2(
            N_max=args.N,
            m_max=args.m,
            d=args.d,
            u_values=args.u_values,
        )

    # Composite figure only when both experiments provided their outputs
    if run_exp1 and run_exp2 and df_weights is not None and rejection_power_all is not None:
        plot_composite_figure(df_weights, rejection_power_all)


if __name__ == "__main__":
    main()
