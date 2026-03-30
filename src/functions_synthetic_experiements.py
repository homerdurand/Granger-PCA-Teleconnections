import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

from models.granger_pca import GrangerPCA
from models.teleconnection_models import (
    RegressionAnalysis,
    CompositeAnalysis,
    CanonicalCorrelationAnalysis,
    SpatialAveraging,
    PrincipalComponents
)

from src.granger_causality_test import test_granger_causality

def generate_m_values(n_points=20, m_min=5, m_max=2000):
    k_min = m_min // 5
    k_max = m_max // 5
    
    k_vals = np.logspace(
        np.log10(k_min),
        np.log10(k_max),
        n_points
    )
    
    k_vals = np.unique(np.round(k_vals).astype(int))
    m_vals = 5 * k_vals
    
    return m_vals


# ==========================================================
# 1. DATA GENERATION
# ==========================================================

def generate_synthetic_data(N=10000, m=5, d=20, u=0.2, seed=2025):
    """
    Generate synthetic VAR-style teleconnection dataset.
    """
    assert m % 5 == 0, "m must be divisible by 5"
    # np.random.seed(seed)

    t = np.arange(1, N+1)

    # Base signals
    X = np.random.normal(0, 1, (N, 1)) + np.sin(0.2 * t)[:, None] + np.cos(0.1 * t)[:, None]
    Z1 = np.random.normal(0, 1, (N, 1)) + np.cos(0.4 * t)[:, None]
    Z2 = np.random.normal(0, 1, (N, 1)) + np.cos(0.05 * t)[:, None]

    block = m // 5
    A = slice(0, block)
    B = slice(block, 2 * block)
    C = slice(2 * block, 3 * block)
    D = slice(3 * block, 4 * block)
    E = slice(4 * block, 5 * block)

    W = np.random.randn(N, m)

    for i in range(3, N):
        W[i, A] += (
            0.2 * W[i-2, A]
            + u * 0.4 * X[i-1, 0]
            + u * 0.3 * X[i-2, 0]
            + u * 0.2 * X[i-3, 0]
            + 0.2 * Z1[i-1, 0]
            + 0.2 * Z1[i-2, 0]
        )

        W[i, B] += 0.4 * W[i-1, B] + 0.5 * Z1[i-1, 0]
        W[i, C] += 0.1 * W[i-2, C] + 0.8 * Z1[i-2, 0] + 0.8 * W[i-1, A] + 0.4 * W[i-2, A]
        W[i, D] += 0.1 * W[i-2, D] - 0.7 * Z2[i-3, 0] + 0.8 * W[i-2, C]
        W[i, E] += 0.1 * W[i-1, E] + 0.8 * Z1[i-2, 0] + 0.6 * Z2[i-1, 0] + 0.2 * Z2[i-2, 0]

    Y = W
    return X, Y, W


# ==========================================================
# 2. TRAIN / TEST SPLIT
# ==========================================================

def train_test_split_time(X, Y, train_ratio=0.8):
    n = len(Y)
    train_size = int(n * train_ratio)

    return (
        X[:train_size], X[train_size:],
        Y[:train_size], Y[train_size:]
    )


# ==========================================================
# 3. FIT ALL METHODS
# ==========================================================

def fit_methods(X_train, Y_train):
    alphas = np.logspace(-5, 12, 20)
    gpca = GrangerPCA(maxlag=3 , method="diff_ratio", model_res=RidgeCV(alphas=alphas), model_full=RidgeCV(alphas=alphas), alpha=1e-1)
    gpca.fit(X_train, Y_train, Z=None, plot_lag_analysis=False, verbose=False)

    ca = CompositeAnalysis(maxlag=gpca.maxlag)
    ca.fit(X_train, Y_train)

    ra = RegressionAnalysis(maxlag=gpca.maxlag)
    ra.fit(X_train, Y_train)

    cca = CanonicalCorrelationAnalysis(maxlag=gpca.maxlag, n_components=1)
    cca.fit(X_train, Y_train)

    sa = SpatialAveraging()
    sa.fit(X_train, Y_train)

    pc = PrincipalComponents()
    pc.fit(X_train, Y_train)

    return gpca, ca, ra, cca, sa, pc


# ==========================================================
# 4. GRANGER EVALUATION
# ==========================================================

def evaluate_methods(X_test, Y_test, models):
    gpca, ca, ra, cca, sa, pc = models
    lag = gpca.maxlag

    cp_gpca = gpca.transform(Y_test)[:, 0]
    cp_ca   = ca.transform(Y_test, type='PosNeg')
    cp_ra   = ra.transform(Y_test)
    cp_cca  = cca.transform(Y_test)
    cp_sa   = sa.transform(Y_test)
    cp_pc   = pc.transform(Y_test)

    results = {
        "GPCA": test_granger_causality(X_test, cp_gpca, maxlag=[lag]),
        "CCA": test_granger_causality(X_test, cp_cca, maxlag=[lag]),
        "Composites": test_granger_causality(X_test, cp_ca, maxlag=[lag]),
        "Regression": test_granger_causality(X_test, cp_ra, maxlag=[lag]),
        "Spatial Average": test_granger_causality(X_test, cp_sa, maxlag=[lag]),
        "PCA": test_granger_causality(X_test, cp_pc, maxlag=[lag]),
    }

    combined = pd.concat(
        [v[v['lag'] == lag] for v in results.values()],
        keys=results.keys()
    ).reset_index(level=0).rename(columns={'level_0': 'Method'})

    return combined


# ==========================================================
# 5. WEIGHT EXTRACTION
# ==========================================================

def extract_weights(models):
    gpca, ca, ra, cca, sa, pc = models

    weights_gpca = np.ravel(np.abs(gpca.vects_)[:, 0])
    weights_ra = np.ravel(np.abs(ra.lr.coef_))
    weights_ca = np.ravel(np.abs(ca.composites['PosNeg'].to_numpy()))
    weights_cca = np.ravel(np.abs(cca.cca.y_loadings_[:, 0]))
    weights_pc = np.ravel(np.abs(pc.pca.components_.T[:, 0]))

    def normalize(w):
        return w / np.max(np.abs(w))

    df = pd.DataFrame({
        "GPCA": normalize(weights_gpca),
        "RA": normalize(weights_ra),
        "CA": normalize(weights_ca),
        "CCA": normalize(weights_cca),
        "PCA": normalize(weights_pc),
    })

    return df


# ==========================================================
# 6. PLOTTING
# ==========================================================

def plot_heatmap(df_weights, save_path="figures/grid_experiments/weights.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sns.set_theme(context='paper', style='white')
    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        df_weights,
        cmap='magma',
        cbar_kws={'shrink': 0.7, 'label': 'Normalized Weight'},
        yticklabels=False
    )

    labels = ["A", "B", "C", "D", "E"]
    n_blocks = len(labels)
    n_rows = df_weights.shape[0]

    block_size = n_rows / n_blocks
    tick_positions = [block_size*(i + 0.5) for i in range(n_blocks)]

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(labels, rotation=0)

    ax.set_xlabel("Method")
    ax.set_ylabel("Component Block")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================================
# 7. MASTER FUNCTION
# ==========================================================

def run_experiment(N=10000, m=5, d=20, u=0.2):
    os.makedirs("results/synth", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    X, Y, W = generate_synthetic_data(N=N, m=m, d=d, u=u)

    X_train, X_test, Y_train, Y_test = train_test_split_time(X, Y)

    models = fit_methods(X_train, Y_train)

    results_df = evaluate_methods(X_test, Y_test, models)

    weights_df = extract_weights(models)

    # Save LaTeX table
    latex = results_df[['Method', 'F_statistic', 'p_value']].to_latex(
        index=False,
        float_format=lambda x: f"{x:.3e}"
    )
    with open(f"tables/synthetic/synthetic_results_N{N}_m{m}.tex", "w") as f:
        f.write(latex)

    # Save heatmap
    plot_heatmap(weights_df,
                 save_path=f"figures/grid_experiments/weights_null/weights_N{N}_m{m}.png")

    return results_df, weights_df