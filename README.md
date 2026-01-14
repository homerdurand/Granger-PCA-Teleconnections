# Granger-PCA Teleconnection Analysis

This repository contains research code for analyzing **teleconnections in climate data** using **Granger PCA**, with a focus on **Granger-PCA**.
The goal is to identify spatial patterns in multivariate climate fields that are causally influenced by a given driver (typically a climate oscillation).

---

## Code Structure

The repository is organized in a simple, research-oriented way:

- **`models/`**  
  Core implementations of Granger-PCA and alternative teleconnection models (regression, composites, CCA, spatial averaging).

- **`utils/`**  
  Shared utilities for lag analysis, Granger causality tests, data reshaping (xarray ↔ NumPy), preprocessing, and plotting of spatial weights.

- **`figures/`**  
  Generated figures (teleconnection patterns, diagnostics).

- **`tables/`**  
  LaTeX tables containing p-values and statistical results.

- **`results/`**  
  Saved outputs from synthetic experiments.

---

## Notebooks

The main results and experiments are reproduced using the following notebooks:

- **`synthetic.ipynb`**  
  Simple synthetic examples illustrating Granger causality and lag selection.

- **`synthetic_multivariate.ipynb`**  
  Multivariate synthetic experiments validating the Granger-PCA method.

- **`teleconnection_NDVI.ipynb`**  
  Application to NDVI teleconnections.

- **`teleconnection_precipitations.ipynb`**  
  Application to precipitation teleconnections over different regions.

Each notebook is self-contained and generates its own figures and tables.

---

## Notes

This repository is intended as a **research codebase** for transparency and reproducibility, rather than a polished software package.
