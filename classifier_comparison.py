#!/usr/bin/env python3
"""Compare candidate classifiers on the recorded FlexEEG motor-imagery data.

Run:  python classifier_comparison.py

The point of this script is not to pick a winner once and freeze it -- it is to
give you an honest number you can re-run every time you record more data.

Two things it does that a naive train_test_split does not:

1. Epochs cut from the same recording are highly correlated. If they are allowed
   to land in both the train and the test fold, accuracy is inflated by ~10-20
   points. Cross-validation here is grouped by recording file.
2. It reports the majority-class baseline next to every score, so "80%" on an
   unbalanced set is not mistaken for a working classifier.
"""
from __future__ import annotations

import glob
import os
import re
from io import StringIO

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = "EEG Hackathon data bios and one"

# Measured from relative_time as n_samples / duration. Do not use the median
# inter-sample gap: LSL delivers samples in bursts, which makes it look like 700 Hz.
FS = 125.0

# Ch3 is an order of magnitude noisier than the rest (bad electrode) and Ch9 is
# flat zero (marker channel), so both are dropped.
CHANNELS = ["Ch1", "Ch2", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"]

MU_BETA = (8.0, 30.0)   # the band motor imagery actually modulates
WINDOW_S = 2.0
STEP_S = 0.5

LABELS = {"hands": "hands", "legimagery": "legs", "rest": "rest"}


def load_blocks(path: str) -> list[pd.DataFrame]:
    """Return each LSL export in a file. Rest1.csv holds two concatenated exports."""
    lines = open(path).readlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("timestamp,")]
    blocks = []
    for k, start in enumerate(starts):
        end = starts[k + 1] - 3 if k + 1 < len(starts) else len(lines)
        df = pd.read_csv(StringIO("".join(lines[start:end])))
        if all(c in df.columns for c in CHANNELS) and len(df) > 3 * FS:
            blocks.append(df)
    return blocks


def to_epochs(df: pd.DataFrame) -> np.ndarray:
    """Band-pass to mu+beta and cut sliding windows. Returns (n_epochs, n_ch, n_samples)."""
    x = df[CHANNELS].to_numpy(float).T
    x = x - x.mean(axis=1, keepdims=True)
    b, a = signal.butter(4, [f / (FS / 2) for f in MU_BETA], btype="band")
    x = signal.filtfilt(b, a, x, axis=1)

    n, step = int(WINDOW_S * FS), int(STEP_S * FS)
    return np.array([x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, step)])


def log_variance(epochs: np.ndarray) -> np.ndarray:
    """One feature per channel: log band power in mu+beta. This is the ERD/ERS signal."""
    return np.log(epochs.var(axis=2) + 1e-12)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs, labels, groups = [], [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        name = os.path.basename(path)
        if "demo_eeg" in open(path).readline():
            continue  # Hands4.csv is synthetic loopback data, not a recording
        label = LABELS[re.match(r"[A-Za-z]+", name).group(0).lower()]
        for block_index, df in enumerate(load_blocks(path)):
            ep = to_epochs(df)
            epochs.append(ep)
            labels += [label] * len(ep)
            groups += [f"{name}#{block_index}"] * len(ep)
    return np.concatenate(epochs), np.array(labels), np.array(groups)


def build_models() -> dict:
    return {
        "LDA (shrinkage)": make_pipeline(
            StandardScaler(), LDA(solver="lsqr", shrinkage="auto")),
        "LDA (plain)": make_pipeline(StandardScaler(), LDA()),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")),
        "SVM (linear)": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    }


def evaluate(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    baseline = max(np.unique(y, return_counts=True)[1]) / len(y)
    print(f"--- {name} | {len(y)} epochs, {len(set(groups))} recordings, "
          f"majority-class baseline {baseline:.2f} ---")
    leaky = StratifiedKFold(5, shuffle=True, random_state=0)
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for label, model in build_models().items():
        a = cross_val_score(model, X, y, cv=leaky).mean()
        b = cross_val_score(model, X, y, cv=grouped, groups=groups)
        print(f"  {label:<20} leaky {a:.2f} | honest {b.mean():.2f} +/- {b.std():.2f}")
    print()


def main() -> None:
    epochs, y, groups = load_dataset()
    X = log_variance(epochs)
    print(f"{len(X)} epochs x {X.shape[1]} features from "
          f"{len(set(groups))} recordings at {FS:g} Hz\n")

    mask = np.isin(y, ("hands", "legs"))
    evaluate("hands vs legs", X[mask], y[mask], groups[mask])

    # NOTE: every Rest recording came from a different machine ("Umbara") than
    # every Hands/Legs recording ("DESKTOP-0N5AHDD"), so this comparison is
    # confounded by session. Treat the score as an upper bound, not as evidence.
    evaluate("active vs rest (session-confounded)",
             X, np.where(y == "rest", "rest", "active"), groups)


if __name__ == "__main__":
    main()
