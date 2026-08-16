"""Train and honestly evaluate motor-imagery classifiers on the hackathon CSVs.

Run:
    python3 train_offline.py

The point of this script is the *comparison between two validation schemes*:

  * Random k-fold over windows -- what class_exampl.ipynb does with
    train_test_split.  Windows from one recording overlap, so near-identical
    windows land on both sides of the split and the score is inflated.
  * Leave-one-recording-out -- the honest number, and the one that predicts
    whether the classifier will work live on the day.

If those two numbers disagree wildly, believe the second one.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bci.data import load_recordings, summarise
from bci.features import build_dataset
from bci.model import train

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "EEG Hackathon data bios and one"


def classifiers():
    """The four candidates from pipeline-layout.ipynb, plus logistic regression."""
    return {
        "LDA (shrinkage)": make_pipeline(
            StandardScaler(), LDA(solver="lsqr", shrinkage="auto")
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    }


def evaluate(X, y, groups, task_name):
    print(f"\n{'=' * 72}\n{task_name}\n{'=' * 72}")
    classes, counts = np.unique(y, return_counts=True)
    majority = counts.max() / counts.sum()
    print(f"windows={len(y)}  classes={dict(zip(classes, counts))}")
    print(f"chance (majority class) = {majority:.1%}")
    print(f"recordings held out one at a time: {len(np.unique(groups))}")

    print(f"\n{'classifier':22s} {'random 5-fold':>15s} {'leave-1-recording-out':>23s}")
    print("-" * 72)

    results = {}
    for name, clf in classifiers().items():
        naive = cross_val_score(
            clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
        ).mean()
        honest = cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=groups).mean()
        results[name] = (naive, honest)
        print(f"{name:22s} {naive:14.1%} {honest:22.1%}")

    print("-" * 72)
    print(f"{'chance':22s} {majority:14.1%} {majority:22.1%}")
    return results


def confusion_for_best(X, y, groups, name="LDA (shrinkage)"):
    """Leave-one-recording-out confusion matrix for a single classifier."""
    clf = classifiers()[name]
    preds, truth = [], []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        clf.fit(X[train], y[train])
        preds.extend(clf.predict(X[test]))
        truth.extend(y[test])

    labels = sorted(np.unique(y))
    print(f"\nLeave-one-recording-out confusion matrix -- {name}")
    print(f"accuracy = {accuracy_score(truth, preds):.1%}")
    print("rows = true, cols = predicted")
    cm = confusion_matrix(truth, preds, labels=labels)
    print(f"{'':8s}" + "".join(f"{l:>10s}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:8s}" + "".join(f"{v:10d}" for v in row))
    print()
    print(classification_report(truth, preds, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=2.0, help="window length, seconds")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--keep-artifact-channels",
        action="store_true",
        help="keep Ch2/Ch3 (the artifact-dominated channels)",
    )
    parser.add_argument("--data-dir", default=None,
                        help="directory of CSVs (default: the supplied dataset)")
    parser.add_argument("--save", nargs="?", const="model.pkl", default=None,
                        help="fit on all data and save to this path (default model.pkl)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    recordings = load_recordings(
        data_dir, drop_artifact_channels=not args.keep_artifact_channels
    )
    if not recordings:
        raise SystemExit(f"No usable recordings found in {data_dir}")
    print(summarise(recordings))

    X, y, groups, names = build_dataset(recordings, args.window, args.overlap)
    print(f"\nfeature vector: {X.shape[1]} features "
          f"({len(names) // 5} channels x 5 bands)")

    evaluate(X, y, groups, f"ALL CLASSES: {' vs '.join(sorted(set(y)))}")

    # On the supplied dataset only, also run the confound-free comparison.
    # Rest was recorded on a different machine ("Umbara") from hands/legs
    # ("DESKTOP-0N5AHDD"), so any rest-vs-imagery accuracy is unfalsifiable:
    # session differences alone would produce it.  Hands vs legs shares a
    # session, so it is the only honest test of motor imagery in that data.
    hosts = {r.host for r in recordings}
    mask = np.isin(y, ["hands", "legs"])
    if len(hosts) > 1 and mask.sum() > 10:
        print("\n!! More than one recording host present: "
              f"{sorted(hosts)}. Labels may be confounded with session.")
        print("   Run diagnose_confound.py for the full picture.")
        evaluate(
            X[mask], y[mask], groups[mask],
            "SAME-SESSION SUBSET: hands vs legs (the honest test)",
        )
        confusion_for_best(X[mask], y[mask], groups[mask])
    else:
        confusion_for_best(X, y, groups)

    if args.save:
        rec = recordings[0]
        model = train(
            X, y, fs=rec.fs, window_s=args.window,
            n_channels=len(rec.channels), channels=rec.channels,
            notes=f"trained on {data_dir}",
        )
        path = model.save(args.save)
        print(f"\nSaved model to {path}")
        print(f"Run it live:  python3 realtime_classifier.py --model {path}")


if __name__ == "__main__":
    main()
