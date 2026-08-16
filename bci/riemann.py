"""Riemannian (tangent space) feature extraction for motor imagery.

What actually goes into the SVM
-------------------------------
Not raw EEG, and not band power.  The chain is:

    epoch (n_channels x n_samples)
      -> spatial covariance matrix (n_channels x n_channels, SPD)
      -> project to the tangent space at the Riemannian mean of the
         training covariances
      -> vector of length n_channels * (n_channels + 1) / 2
      -> standardise -> SVM with RBF kernel

The covariance matrix is the feature.  Motor imagery shows up as event-related
de/synchronisation in the mu and beta bands, which changes the *variance* of
sensorimotor channels and the *correlation* between them.  A covariance matrix
captures both, and unlike per-channel band power it keeps the cross-channel
terms, which is where the C3/C4 lateralisation lives.

Covariance matrices are symmetric positive definite, so they live on a curved
manifold, not in a vector space.  Feeding them to an SVM directly would be a
category error: the straight-line distance between two SPD matrices is not the
meaningful distance between them.  The tangent space projection maps the
manifold to a flat space *local to the training data's mean*, after which
ordinary Euclidean classifiers apply.  That projection is a fitted parameter --
it must be learned on the training fold only.

Bandpass choice
---------------
8-30 Hz by default: mu (8-13) plus beta (13-30), the two rhythms motor imagery
modulates.  Passing wider costs you -- drift and EMG dominate the covariance
and swamp the effect.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import bandpass, notch

# Motor imagery lives here.  Filter bank splits it further.
MI_BAND = (8.0, 30.0)
FILTER_BANK = [(8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0)]


def tangent_space_dim(n_channels: int) -> int:
    """Length of the vector the SVM actually sees, per band."""
    return n_channels * (n_channels + 1) // 2


class BandPassEpochs(BaseEstimator, TransformerMixin):
    """Band-pass a stack of epochs.  Stateless, so safe inside a pipeline."""

    def __init__(self, fs: float = 125.0, low: float = 8.0, high: float = 30.0):
        self.fs, self.low, self.high = fs, low, high

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for i, epoch in enumerate(X):
            filtered = notch(epoch, self.fs)
            filtered = bandpass(filtered, self.fs, self.low, self.high)
            # Common average reference after filtering.
            if filtered.shape[0] > 1:
                filtered = filtered - filtered.mean(axis=0, keepdims=True)
            out[i] = filtered
        return out


def riemann_svm(fs: float = 125.0, band=MI_BAND, C: float = 1.0,
                gamma="scale", estimator: str = "oas") -> Pipeline:
    """The pipeline you asked about: covariance -> tangent space -> SVM RBF.

    `estimator="oas"` is a shrinkage covariance estimator (oracle approximating
    shrinkage).  With 6 channels and 250 samples per window the empirical
    covariance is already well-conditioned, but shrinkage costs nothing and
    keeps the pipeline from breaking if a channel goes flat mid-session --
    a singular covariance is not SPD and the tangent space projection fails
    outright.
    """
    return make_pipeline(
        BandPassEpochs(fs=fs, low=band[0], high=band[1]),
        Covariances(estimator=estimator),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        SVC(C=C, gamma=gamma, kernel="rbf"),
    )


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Tangent space features from several bands, concatenated.

    Multiplies the feature count by the number of bands, which is where
    dimensionality reduction starts to matter.
    """

    def __init__(self, fs: float = 125.0, bands=None, estimator: str = "oas"):
        self.fs = fs
        self.bands = bands or FILTER_BANK
        self.estimator = estimator

    def fit(self, X, y=None):
        self.blocks_ = []
        for low, high in self.bands:
            bp = BandPassEpochs(self.fs, low, high)
            cov = Covariances(estimator=self.estimator)
            ts = TangentSpace(metric="riemann")
            ts.fit(cov.fit_transform(bp.transform(X)))
            self.blocks_.append((bp, cov, ts))
        return self

    def transform(self, X):
        return np.hstack([ts.transform(cov.transform(bp.transform(X)))
                          for bp, cov, ts in self.blocks_])


def epochs_from_recordings(recordings, window_s: float = 2.0, overlap: float = 0.5):
    """Stack recordings into (n_epochs, n_channels, n_samples) + labels + groups.

    Riemannian methods need the raw epoch, not a feature vector, so this
    replaces build_dataset() from bci.features.
    """
    X, y, groups = [], [], []
    for rec in recordings:
        win = int(window_s * rec.fs)
        step = max(1, int(win * (1.0 - overlap)))
        n = rec.data.shape[1]
        for s in range(0, n - win + 1, step):
            X.append(rec.data[:, s : s + win])
            y.append(rec.label)
            groups.append(rec.source)
    return np.asarray(X), np.asarray(y), np.asarray(groups)
