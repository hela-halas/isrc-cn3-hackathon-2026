"""Common Spatial Patterns as an sklearn transformer.

CSP is the spatial half of the pipeline and the part that actually sees the
lateralisation in motor imagery. A band-pass filter treats every electrode
independently, so on its own it cannot represent "right sensorimotor cortex
desynchronised relative to left" -- which is the signal. CSP learns channel
mixtures whose variance is maximally different between the two classes, and
log-variance in those mixtures is the feature the classifier reads.

Implemented directly rather than pulled from MNE to keep the runtime
dependencies to numpy/scipy/sklearn, which is all the game machine needs.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _covariance(epoch: np.ndarray, shrinkage: float) -> np.ndarray:
    """Trace-normalised covariance of one ``(n_channels, n_samples)`` epoch.

    Normalising by the trace makes the estimate invariant to overall amplitude,
    so a trial recorded with slightly better electrode contact does not
    dominate the class average. Shrinkage toward a scaled identity keeps the
    matrix invertible when there are far fewer epochs than the covariance has
    free parameters -- which is the situation with a handful of trials.
    """
    centred = epoch - epoch.mean(axis=1, keepdims=True)
    cov = centred @ centred.T
    trace = np.trace(cov)
    if trace <= 0:
        raise ValueError("epoch has zero variance; check for a dead channel")
    cov /= trace
    if shrinkage > 0:
        n = cov.shape[0]
        cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / n * np.eye(n)
    return cov


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns for two classes.

    Parameters
    ----------
    n_components:
        Number of spatial filters kept, taken in pairs from the two ends of the
        eigenvalue spectrum (most class-A-dominant and most class-B-dominant).
        Must be even. Four is a sensible default for eight channels; more
        components on few trials overfits.
    shrinkage:
        Ledoit-Wolf-style shrinkage applied to each epoch covariance, in
        ``[0, 1]``. With very few trials, some shrinkage is not optional.
    log:
        Return ``log(var)`` of each filtered component. Log-variance is roughly
        normally distributed, which is what LDA assumes.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        The learned spatial filters, applied as ``filters_ @ epoch``.
    patterns_ : ndarray, shape (n_components, n_channels)
        The corresponding spatial patterns. These, not the filters, are what
        you plot on a scalp map to interpret where the signal came from.
    """

    def __init__(self, n_components: int = 4, shrinkage: float = 0.1, log: bool = True):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.log = log

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        """Fit on ``X`` of shape (n_epochs, n_channels, n_samples)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"expected (n_epochs, n_channels, n_samples), got {X.shape}")
        if self.n_components % 2 != 0:
            raise ValueError("n_components must be even (filters are taken in pairs)")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSP is a two-class method, got {len(classes)} classes: {classes}. "
                "Wrap it in OneVsRestClassifier for more."
            )
        n_channels = X.shape[1]
        if self.n_components > n_channels:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_channels={n_channels}"
            )

        self.classes_ = classes
        class_covs = []
        for cls in classes:
            covs = np.array([_covariance(e, self.shrinkage) for e in X[y == cls]])
            class_covs.append(covs.mean(axis=0))

        # Solve C_a w = lambda (C_a + C_b) w. Eigenvalues land in [0, 1]: near 1
        # means the component's variance is almost all class A, near 0 almost
        # all class B. The extremes are therefore the discriminative ones.
        eigenvalues, eigenvectors = linalg.eigh(class_covs[0], class_covs[0] + class_covs[1])
        order = np.argsort(eigenvalues)
        # Interleave from both ends: highest, lowest, second-highest, ...
        picks = np.empty(len(order), dtype=int)
        picks[0::2] = order[::-1][: (len(order) + 1) // 2]
        picks[1::2] = order[: len(order) // 2]
        eigenvectors = eigenvectors[:, picks[: self.n_components]]

        self.filters_ = eigenvectors.T
        self.patterns_ = linalg.pinv(eigenvectors)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project epochs onto the spatial filters and return their power."""
        check_is_fitted(self, "filters_")
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"expected (n_epochs, n_channels, n_samples), got {X.shape}")

        projected = np.asarray([self.filters_ @ epoch for epoch in X])
        power = np.var(projected, axis=2)
        # Normalise per epoch so the feature is the *distribution* of power
        # across components, not the absolute amplitude of the recording.
        power /= power.sum(axis=1, keepdims=True)
        if self.log:
            power = np.log(np.maximum(power, np.finfo(float).tiny))
        return power
