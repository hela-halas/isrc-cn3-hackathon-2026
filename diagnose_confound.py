"""Check whether 'rest vs imagery' is really 'Umbara vs DESKTOP-0N5AHDD'.

Every rest recording came off a different machine than every hands/legs
recording.  Label and recording session are therefore perfectly confounded: a
classifier can score well on the three-class task without learning anything
about motor imagery at all.

This script quantifies that.  Run:  python3 diagnose_confound.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bci.data import load_recordings
from bci.features import build_dataset

DATA_DIR = Path(__file__).parent / "EEG Hackathon data bios and one"


def main():
    recordings = load_recordings(DATA_DIR)
    X, y, groups, _ = build_dataset(recordings)

    host_by_source = {r.source: r.host for r in recordings}
    hosts = np.asarray([host_by_source[g] for g in groups])

    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    cv = LeaveOneGroupOut()

    print("=" * 68)
    print("Is the label confounded with the recording session?")
    print("=" * 68)
    print("\nlabel x host contingency (windows):")
    labels, hs = sorted(set(y)), sorted(set(hosts))
    print(f"{'':8s}" + "".join(f"{h:>20s}" for h in hs))
    for label in labels:
        row = [int(((y == label) & (hosts == h)).sum()) for h in hs]
        print(f"{label:8s}" + "".join(f"{v:20d}" for v in row))

    print("\nIf a column has exactly one non-zero row, the label IS the session.")

    # 1. Can we predict the recording machine from the EEG features?
    host_acc = cross_val_score(clf, X, hosts, cv=cv, groups=groups).mean()
    host_chance = max((hosts == h).mean() for h in hs)
    print(f"\n1. Predicting the HOST from EEG features: {host_acc:.1%} "
          f"(chance {host_chance:.1%})")
    print("   A high score means the two sessions are trivially distinguishable,")
    print("   so any rest-vs-imagery accuracy could be session, not brain state.")

    # 2. Rest vs imagery -- crosses the session boundary, so it is unfalsifiable.
    binary = np.where(y == "rest", "rest", "imagery")
    rest_acc = cross_val_score(clf, X, binary, cv=cv, groups=groups).mean()
    rest_chance = max((binary == c).mean() for c in set(binary))
    print(f"\n2. rest vs imagery (crosses sessions): {rest_acc:.1%} "
          f"(chance {rest_chance:.1%})")

    # 3. Hands vs legs -- same session, so this one is real.
    mask = np.isin(y, ["hands", "legs"])
    hl_acc = cross_val_score(clf, X[mask], y[mask], cv=cv, groups=groups[mask]).mean()
    hl_chance = max((y[mask] == c).mean() for c in ["hands", "legs"])
    print(f"\n3. hands vs legs (same session):       {hl_acc:.1%} "
          f"(chance {hl_chance:.1%})")

    print("\n" + "=" * 68)
    print("Read (2) and (3) together.  (2) is inflated by the session it cannot")
    print("separate from the label.  (3) is the trustworthy measurement, and it")
    print("is the one that tells you how much signal this data really has.")
    print("=" * 68)

    # 4. How much data do we actually have?
    total = sum(r.duration for r in recordings)
    print(f"\nTotal usable EEG: {total:.0f}s across {len(recordings)} recordings")
    print(f"Windows for training: {len(y)} (2s windows, 50% overlap)")
    print("For reference, a typical motor-imagery training set is 20-40 minutes")
    print("per subject with 100+ trials per class.")


if __name__ == "__main__":
    main()
