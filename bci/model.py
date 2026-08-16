"""Train, save and load the live classifier.

Kept deliberately small: the model is a scikit-learn pipeline pickled to disk
alongside the metadata the real-time script needs to rebuild identical features
(sampling rate, window length, channel count).  If those disagree between
training and inference the classifier silently produces garbage, so they travel
together.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model.pkl"


@dataclass
class TrainedModel:
    """A fitted pipeline plus everything needed to reproduce its input.

    `feature_mode` decides what the live loop must hand the pipeline:
      "bandpower" -- a feature vector from bci.features.log_band_power
      "riemann"   -- the raw epoch, because bci.riemann.riemann_svm does its
                     own filtering, covariance and tangent space projection
                     inside the pipeline
    """

    pipeline: object
    classes: list[str]
    fs: float
    window_s: float
    n_channels: int
    channels: list[str] = field(default_factory=list)
    notes: str = ""
    feature_mode: str = "bandpower"

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path = DEFAULT_MODEL_PATH) -> "TrainedModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}. Record calibration data with "
                f"record_calibration.py, then train with train_offline.py --save."
            )
        with path.open("rb") as fh:
            return pickle.load(fh)


def default_classifier():
    """Shrinkage LDA.

    LDA is the right default here and not just by tradition: with a 30-feature
    vector and a calibration set of maybe a hundred windows, anything with more
    capacity overfits.  `shrinkage="auto"` (Ledoit-Wolf) regularises the
    covariance estimate, which is exactly the quantity that is badly estimated
    when n_samples is close to n_features.
    """
    return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))


def train(X: np.ndarray, y: np.ndarray, fs: float, window_s: float,
          n_channels: int, channels: list[str] | None = None,
          clf=None, notes: str = "",
          feature_mode: str = "bandpower") -> TrainedModel:
    clf = clf or default_classifier()
    clf.fit(X, y)
    return TrainedModel(
        pipeline=clf,
        classes=sorted(set(y)),
        fs=fs,
        window_s=window_s,
        n_channels=n_channels,
        channels=channels or [],
        notes=notes,
        feature_mode=feature_mode,
    )
