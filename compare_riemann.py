"""Does the Riemannian SVM need dimensionality reduction?  Measured, not guessed.

Run:  python3 compare_riemann.py

Answers two questions:
  1. How many features does the SVM actually see?
  2. Does reducing that number help, hurt, or do nothing?
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bci.data import load_recordings
from bci.riemann import (
    FILTER_BANK,
    BandPassEpochs,
    FilterBankTangentSpace,
    epochs_from_recordings,
    riemann_svm,
    tangent_space_dim,
)

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent / "EEG Hackathon data bios and one"


def score(clf, X, y, groups):
    """Random k-fold (inflated) and leave-one-recording-out (honest)."""
    naive = cross_val_score(
        clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
    ).mean()
    honest = cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=groups).mean()
    return naive, honest


def main():
    recordings = load_recordings(DATA_DIR)
    X, y, groups = epochs_from_recordings(recordings)
    n_ch, n_samp = X.shape[1], X.shape[2]

    print("=" * 74)
    print("WHAT THE SVM ACTUALLY SEES")
    print("=" * 74)
    print(f"\nepoch shape           : {n_ch} channels x {n_samp} samples "
          f"= {n_ch * n_samp} raw numbers")
    print(f"covariance matrix     : {n_ch} x {n_ch} (symmetric, positive definite)")
    print(f"tangent space vector  : {tangent_space_dim(n_ch)} features "
          f"= {n_ch}*({n_ch}+1)/2")
    print(f"  -> this is the SVM's input, one vector per 2s window")
    print(f"filter bank ({len(FILTER_BANK)} bands) : "
          f"{tangent_space_dim(n_ch) * len(FILTER_BANK)} features")

    # Samples-per-feature ratio is the number that decides whether reduction
    # is even a question.
    print(f"\ntraining windows      : {len(y)}")
    print(f"samples per feature   : {len(y) / tangent_space_dim(n_ch):.1f}  "
          f"(single band)")
    print(f"                        {len(y) / (tangent_space_dim(n_ch) * len(FILTER_BANK)):.1f}  "
          f"(filter bank)")

    # Covariance conditioning: the tangent space projection needs SPD input.
    bp = BandPassEpochs(fs=125.0, low=8.0, high=30.0)
    covs = Covariances(estimator="oas").fit_transform(bp.transform(X))
    conds = np.linalg.cond(covs)
    print(f"\ncovariance condition number: median {np.median(conds):.1f}, "
          f"max {conds.max():.1f}")
    print("  (samples per window >> channels, so these are well-conditioned;")
    print("   a number in the thousands would mean rank-deficient covariances)")

    # ---------------- the honest task: hands vs legs, same session ----------
    mask = np.isin(y, ["hands", "legs"])
    Xh, yh, gh = X[mask], y[mask], groups[mask]
    chance = max((yh == c).mean() for c in set(yh))

    print("\n" + "=" * 74)
    print("DOES DIMENSIONALITY REDUCTION HELP?  (hands vs legs, same session)")
    print("=" * 74)
    print(f"\n{'pipeline':44s} {'5-fold':>10s} {'LORO':>10s}")
    print("-" * 74)

    ts = lambda: make_pipeline(  # noqa: E731
        BandPassEpochs(125.0, 8.0, 30.0),
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
    )

    candidates = {
        f"Riemann TS + SVM-RBF (no reduction, {tangent_space_dim(n_ch)}f)":
            riemann_svm(),
        "Riemann TS + PCA(10) + SVM-RBF":
            make_pipeline(ts(), StandardScaler(), PCA(n_components=10),
                          SVC(kernel="rbf")),
        "Riemann TS + PCA(5) + SVM-RBF":
            make_pipeline(ts(), StandardScaler(), PCA(n_components=5),
                          SVC(kernel="rbf")),
        "Riemann TS + SelectKBest(10) + SVM-RBF":
            make_pipeline(ts(), StandardScaler(),
                          SelectKBest(f_classif, k=10), SVC(kernel="rbf")),
        "Riemann TS + LDA (no SVM)":
            make_pipeline(ts(), StandardScaler(),
                          LDA(solver="lsqr", shrinkage="auto")),
        f"Filter bank TS + SVM-RBF ({tangent_space_dim(n_ch) * len(FILTER_BANK)}f)":
            make_pipeline(FilterBankTangentSpace(), StandardScaler(),
                          SVC(kernel="rbf")),
        "Filter bank TS + PCA(15) + SVM-RBF":
            make_pipeline(FilterBankTangentSpace(), StandardScaler(),
                          PCA(n_components=15), SVC(kernel="rbf")),
    }

    for name, clf in candidates.items():
        try:
            naive, honest = score(clf, Xh, yh, gh)
            print(f"{name:44s} {naive:9.1%} {honest:9.1%}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:44s} {'failed':>10s}  {type(exc).__name__}: {exc}")

    print("-" * 74)
    print(f"{'chance':44s} {chance:9.1%} {chance:9.1%}")

    # ---------------- how many PCA components carry the variance -----------
    tsf = ts().fit_transform(Xh)
    tsf = StandardScaler().fit_transform(tsf)
    pca = PCA().fit(tsf)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA on the {tsf.shape[1]} tangent-space features:")
    for target in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, target) + 1)
        print(f"  {target:.0%} of variance in {k} components")

    print("\n" + "=" * 74)
    print("Caution: the numbers above are near chance because this dataset has")
    print("almost no motor-imagery signal (see diagnose_confound.py). Treat the")
    print("dimensionality arithmetic as valid and the accuracies as a null result.")
    print("=" * 74)


if __name__ == "__main__":
    main()
