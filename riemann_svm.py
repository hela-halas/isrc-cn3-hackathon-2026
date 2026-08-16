#!/usr/bin/env python3
"""Riemannian SVM for the motor-imagery classifier.

Run:  python riemann_svm.py                 # 4-train / 1-test holdout per class
      python riemann_svm.py --rotate        # every held-out choice, mean +/- spread
      python riemann_svm.py --tune          # nested grid search over C / metric
      python riemann_svm.py --save model.joblib

Pipeline
--------
    epochs (n_epochs, n_channels, n_samples)
        -> Covariances()   spatial covariance matrix per epoch, 7x7 SPD
        -> SVC(metric=...) SVM with a Riemannian kernel on the SPD manifold

A note on ordering, since it comes up: LDA cannot go in front of this. LDA's
transform outputs n_classes - 1 dimensions, so for a binary flap decision it
returns a single number per epoch. The covariance of a 1-channel signal is a
scalar, not a matrix, and the manifold of 1x1 SPD matrices is just the positive
reals -- the Riemannian geometry this pipeline is built to exploit disappears
entirely. To reduce dimensionality before the SVM, use --csp, which applies
Riemannian CSP and returns smaller SPD matrices, keeping the geometry intact.
"""
from __future__ import annotations

import argparse

import numpy as np
from pyriemann.classification import SVC as RiemannSVC
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP as RiemannCSP
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from classifier_comparison import load_dataset
from train_holdout import CLASS_ORDER, holdout_masks

# Chosen by nested grouped CV on the recorded hands-vs-legs data.
BEST = {"estimator": "scm", "C": 10.0, "metric": "logeuclid"}


def build_classifier(csp_filters: int | None = None):
    """Covariances -> (optional Riemannian CSP) -> Riemannian-kernel SVM."""
    steps = [Covariances(estimator=BEST["estimator"])]
    if csp_filters:
        steps.append(RiemannCSP(nfilter=csp_filters, log=False))
    steps.append(RiemannSVC(C=BEST["C"], metric=BEST["metric"]))
    return make_pipeline(*steps)


def tune(E, y, groups):
    """Nested grid search. Re-run this whenever you add recordings."""
    grid = {
        "covariances__estimator": ["oas", "lwf", "scm"],
        "svc__C": [0.01, 0.1, 1, 10, 100],
        "svc__metric": ["riemann", "logeuclid"],
    }
    search = GridSearchCV(
        make_pipeline(Covariances(), RiemannSVC()), grid,
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1), n_jobs=-1)
    search.fit(E, y, groups=groups)
    print(f"best params {search.best_params_} (inner score {search.best_score_:.2f})")
    return search.best_estimator_


def run_fold(model, E, y, groups, fold, verbose):
    train, test = holdout_masks(y, groups, fold)
    classes = [c for c in CLASS_ORDER if c in set(y)]
    model.fit(E[train], y[train])
    predicted = model.predict(E[test])
    accuracy = accuracy_score(y[test], predicted)

    if verbose:
        print("held out:", ", ".join(sorted(set(groups[test]))))
        print(f"train {train.sum()} epochs | test {test.sum()} epochs")
        print(f"\naccuracy {accuracy:.2f} (chance {1 / len(classes):.2f})\n")
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
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--tune", action="store_true", help="nested grid search first")
    parser.add_argument("--csp", type=int, metavar="N",
                        help="reduce to N Riemannian CSP filters before the SVM")
    parser.add_argument("--all-classes", action="store_true",
                        help="keep rest, which was recorded on a different machine")
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    E, y, groups = load_dataset()
    if not args.all_classes:
        keep = np.isin(y, ("hands", "legs"))
        E, y, groups = E[keep], y[keep], groups[keep]

    for label in np.unique(y):
        print(f"{label:>6}: {len(set(groups[y == label]))} recordings, "
              f"{(y == label).sum()} epochs")
    print()

    model = tune(E, y, groups) if args.tune else build_classifier(args.csp)

    if args.rotate:
        n_folds = max(len(set(groups[y == label])) for label in np.unique(y))
        scores = [run_fold(model, E, y, groups, f, False) for f in range(n_folds)]
        for fold, score in enumerate(scores):
            print(f"  held-out set {fold + 1}: accuracy {score:.2f}")
        print(f"\nmean {np.mean(scores):.2f} +/- {np.std(scores):.2f} over {n_folds} splits")
    else:
        run_fold(model, E, y, groups, -1, True)

    if args.save:
        import joblib
        model.fit(E, y)
        joblib.dump({"model": model, "classes": list(np.unique(y))}, args.save)
        print(f"fitted on all {len(E)} epochs and saved to {args.save}")


if __name__ == "__main__":
    main()
