#!/usr/bin/env python3
"""Train on 4 recordings per class, test on the held-out 1.

Run:  python train_holdout.py                # hold out the last file of each class
      python train_holdout.py --rotate       # repeat, holding out each file in turn
      python train_holdout.py --save model.joblib

The split is by FILE, never by epoch. Sliding windows cut from one recording are
near-duplicates of each other, so an epoch-level split would put copies of the
test data into training and report an accuracy you cannot reproduce live.

Caveat worth remembering when you read the numbers: one held-out file per class
is ~25 test epochs, so a single run has a margin of error around +/-10 points.
Use --rotate to see how much the score moves with the choice of held-out file.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifier_comparison import load_dataset, log_variance

# Hands has only 4 usable recordings (Hands4.csv is synthetic loopback data), so
# that class trains on 3. Legs and rest train on 4.
CLASS_ORDER = ["hands", "legs", "rest"]


def build_classifier():
    """Shrinkage LDA: the linear models were the only ones not overfitting 9 recordings."""
    return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))


def holdout_masks(y: np.ndarray, groups: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one recording per class. `fold` selects which one, wrapping per class."""
    test = np.zeros(len(y), bool)
    for label in np.unique(y):
        recordings = sorted(set(groups[y == label]))
        held_out = recordings[fold % len(recordings)]
        test |= groups == held_out
    return ~test, test


def run_fold(X, y, groups, fold: int, verbose: bool) -> float:
    train, test = holdout_masks(y, groups, fold)
    classes = [c for c in CLASS_ORDER if c in set(y)]

    model = build_classifier()
    model.fit(X[train], y[train])
    predicted = model.predict(X[test])
    accuracy = accuracy_score(y[test], predicted)

    if verbose:
        print("held out:", ", ".join(sorted(set(groups[test]))))
        print(f"train {train.sum()} epochs from {len(set(groups[train]))} recordings | "
              f"test {test.sum()} epochs from {len(set(groups[test]))} recordings")
        print(f"\naccuracy {accuracy:.2f} "
              f"(chance {1 / len(classes):.2f})\n")
        print("confusion matrix (rows = true, cols = predicted)")
        print(f"{'':>8}" + "".join(f"{c:>8}" for c in classes))
        for label, row in zip(classes, confusion_matrix(y[test], predicted, labels=classes)):
            print(f"{label:>8}" + "".join(f"{v:>8}" for v in row))
        print()
        print(classification_report(y[test], predicted, labels=classes, zero_division=0))
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rotate", action="store_true",
                        help="repeat the split holding out each recording in turn")
    parser.add_argument("--hands-vs-legs", action="store_true",
                        help="drop rest, which was recorded on a different machine")
    parser.add_argument("--save", metavar="PATH",
                        help="fit on every recording and write the model to PATH")
    args = parser.parse_args()

    epochs, y, groups = load_dataset()
    X = log_variance(epochs)

    if args.hands_vs_legs:
        keep = np.isin(y, ("hands", "legs"))
        X, y, groups = X[keep], y[keep], groups[keep]

    for label in np.unique(y):
        print(f"{label:>6}: {len(set(groups[y == label]))} recordings, "
              f"{(y == label).sum()} epochs")
    print()

    if args.rotate:
        n_folds = max(len(set(groups[y == label])) for label in np.unique(y))
        scores = [run_fold(X, y, groups, fold, verbose=False) for fold in range(n_folds)]
        for fold, score in enumerate(scores):
            print(f"  held-out set {fold + 1}: accuracy {score:.2f}")
        print(f"\nmean {np.mean(scores):.2f} +/- {np.std(scores):.2f} over {n_folds} splits")
    else:
        run_fold(X, y, groups, fold=-1, verbose=True)

    if args.save:
        import joblib
        model = build_classifier().fit(X, y)
        joblib.dump({"model": model, "classes": list(model.classes_)}, args.save)
        print(f"fitted on all {len(X)} epochs and saved to {args.save}")


if __name__ == "__main__":
    main()
