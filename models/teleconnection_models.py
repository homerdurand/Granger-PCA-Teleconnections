# ======================================================================
# Imports
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA

from utils.utils import generate_past_data


# ======================================================================
# Composite Analysis
# ======================================================================

class CompositeAnalysis:
    """
    Composite analysis based on thresholded values of X.
    """

    def __init__(self, maxlag=None):
        self.maxlag = maxlag
        return None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, Y, threshold=1):

        # Lag alignment
        if self.maxlag is not None:
            Y = Y[:-self.maxlag, :]
            X = X[self.maxlag:, :]

        # Use first component of X
        X = X[:, 0]

        # --------------------------------------------------------------
        # Threshold-based indices
        # --------------------------------------------------------------

        X_positive_idx = X > threshold
        X_negative_idx = X < -threshold
        X_neutral_idx = (X < threshold) & (X > -threshold)

        # --------------------------------------------------------------
        # Conditional subsets of Y
        # --------------------------------------------------------------

        Y_positive = Y[X_positive_idx, :]
        Y_negative = Y[X_negative_idx, :]
        Y_neutral = Y[X_neutral_idx, :]

        # Temporal means
        Y_positive_mean = Y_positive.mean(axis=0)
        Y_negative_mean = Y_negative.mean(axis=0)
        Y_neutral_mean = Y_neutral.mean(axis=0)

        # --------------------------------------------------------------
        # Composites
        # --------------------------------------------------------------

        composite_posneg = Y_positive_mean - Y_negative_mean
        composite_posneut = Y_positive_mean - Y_neutral_mean
        composite_neutneg = Y_neutral_mean - Y_negative_mean

        self.composites = pd.DataFrame({
            'PosNeg': composite_posneg,
            'PosNeut': composite_posneut,
            'NeutNeg': composite_neutneg
        })

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, Y, type='PosNeg'):
        components = Y @ self.composites[type]
        components = (
            components - components.mean(axis=0)
        ) / components.std(axis=0)
        return components

    def fit_transform(self, X, Y, type='PosNeg'):
        self.fit(X, Y, type=type)
        return self.transform(Y)


# ======================================================================
# Regression Analysis
# ======================================================================

class RegressionAnalysis:
    """
    Linear regression–based projection of Y onto X.
    """

    def __init__(self, maxlag=None):
        self.maxlag = maxlag
        return None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, Y):

        # Lag alignment
        if self.maxlag is not None:
            Y = Y[:-self.maxlag, :]
            X = X[self.maxlag:, :]

        lr = LinearRegression()
        lr.fit(X, Y)
        self.lr = lr

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, Y):
        components = Y @ self.lr.coef_
        components = (
            components - components.mean(axis=0)
        ) / components.std(axis=0)
        return components

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(Y)


# ======================================================================
# Canonical Correlation Analysis
# ======================================================================

class CanonicalCorrelationAnalysis:
    """
    Canonical correlation analysis between lagged X and Y.
    """

    def __init__(self, maxlag=None, n_components=1):
        self.maxlag = maxlag
        self.n_components = n_components
        return None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, Y):

        if self.maxlag is not None:
            X_past, _ = generate_past_data(X, maxlag=self.maxlag, inflag=0)
            _, Y_present = generate_past_data(Y, maxlag=self.maxlag, inflag=0)

        cca = CCA(n_components=self.n_components)
        cca.fit(X_past, Y_present)
        self.cca = cca

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, Y):
        components = Y @ self.cca.y_rotations_
        components = (
            components - components.mean(axis=0)
        ) / components.std(axis=0)
        return components

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(Y)


# ======================================================================
# Spatial Averaging
# ======================================================================

class SpatialAveraging:
    """
    Uniform spatial averaging of Y.
    """

    def __init__(self):
        return None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, Y):
        d = Y.shape[1]
        self.weights_ = np.repeat(1, d) / d

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, Y):
        components = Y @ self.weights_
        components = (
            components - components.mean(axis=0)
        ) / components.std(axis=0)
        return components

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(Y)
