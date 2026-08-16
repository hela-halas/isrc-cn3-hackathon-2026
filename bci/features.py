"""Preprocessing, epoching and feature extraction.

The feature set is log band power per channel, which is the standard baseline
for motor imagery and the thing `class_exampl.ipynb` was reaching for.  Two
differences from that notebook:

  * Band power is computed per channel and kept per channel.  Averaging across
    channels (as the notebook does) throws away exactly the information motor
    imagery lives in -- the lateralisation between C3 and C4.
  * Power is log-transformed.  Raw variance is heavily skewed, which hurts every
    linear model downstream.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),  # mu rhythm over sensorimotor cortex
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


def bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Note `filtfilt`, not `lfilter`: bandpass.ipynb uses lfilter for the low-pass
    leg, which adds a phase shift the high-pass leg does not have.  It also
    re-filters its own output inside the order sweep, so by order 9 the signal
    has been through 45 filter stages.
    """
    nyq = 0.5 * fs
    high = min(high, nyq * 0.95)
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def notch(data: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove mains hum.  UK mains is 50 Hz."""
    if freq >= 0.5 * fs:
        return data
    b, a = signal.iirnotch(freq, q, fs)
    return signal.filtfilt(b, a, data, axis=-1)


def preprocess(data: np.ndarray, fs: float, low: float = 1.0, high: float = 45.0) -> np.ndarray:
    """Notch, band-pass, then common-average-reference."""
    out = notch(data, fs)
    out = bandpass(out, fs, low, high)
    # CAR: subtract the mean across channels at each time point.  Removes
    # session-wide common noise, which is the main thing separating the two
    # recording hosts in this dataset.
    if out.shape[0] > 1:
        out = out - out.mean(axis=0, keepdims=True)
    return out


def epoch(data: np.ndarray, fs: float, window_s: float = 2.0, overlap: float = 0.5):
    """Slice a continuous recording into fixed-length windows."""
    win = int(window_s * fs)
    step = max(1, int(win * (1.0 - overlap)))
    n = data.shape[1]
    return [data[:, s : s + win] for s in range(0, n - win + 1, step)]


def log_band_power(window: np.ndarray, fs: float, bands: dict | None = None) -> np.ndarray:
    """Log power in each band, for each channel, via Welch's method.

    Returns a flat vector of length n_channels * n_bands.
    """
    bands = bands or BANDS
    nperseg = min(window.shape[1], int(fs))
    freqs, psd = signal.welch(window, fs=fs, nperseg=nperseg, axis=-1)

    feats = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            feats.append(np.zeros(window.shape[0]))
            continue
        feats.append(np.log(psd[:, mask].mean(axis=-1) + 1e-12))
    # (n_bands, n_channels) -> flat, channel-major
    return np.concatenate(feats)


def feature_names(channels: list[str], bands: dict | None = None) -> list[str]:
    bands = bands or BANDS
    return [f"{band}_{ch}" for band in bands for ch in channels]


def build_dataset(recordings, window_s: float = 2.0, overlap: float = 0.5):
    """Turn Recordings into (X, y, groups, names) ready for scikit-learn.

    `groups` is the source file, so cross-validation can hold out whole
    recordings.  Windows from one recording overlap and are highly correlated;
    splitting them at random leaks the answer across the train/test boundary.
    """
    X, y, groups = [], [], []
    channels = None

    for rec in recordings:
        clean = preprocess(rec.data, rec.fs)
        for window in epoch(clean, rec.fs, window_s, overlap):
            X.append(log_band_power(window, rec.fs))
            y.append(rec.label)
            groups.append(rec.source)
        channels = rec.channels

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(groups),
        feature_names(channels or []),
    )
