#!/usr/bin/env python3
"""Train the motor-imagery classifier on the recorded CSVs.

    python train_classifier.py audit
    python train_classifier.py train
    python train_classifier.py train --labels hands legimagery --band 8 30
    python train_classifier.py sweep

``audit`` prints what is actually in the data directory and why some files get
excluded; run it first. ``train`` fits the pipeline, reports leave-one-trial-out
accuracy, and writes ``model.npz`` for ``realtime_bci.py`` to load. ``sweep``
tries a grid of bands and epoch lengths so you can pick them on evidence rather
than by feel.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from bci.data import audit, load_dataset
from bci.pipeline import (
    DEFAULT_EPOCH_SECONDS,
    DEFAULT_STEP_SECONDS,
    epoch_dataset,
    leave_one_trial_out,
    make_pipeline,
)

DEFAULT_DATA_DIR = Path(__file__).parent / "EEG Hackathon data bios and one"
DEFAULT_MODEL = Path(__file__).parent / "model.pkl"


def cmd_audit(args: argparse.Namespace) -> None:
    print(audit(args.data_dir))


def _load(args: argparse.Namespace):
    trials = load_dataset(args.data_dir, labels=args.labels)
    print(f"Loaded {len(trials)} recordings from {args.data_dir}")
    for trial in trials:
        notes = f"   [{'; '.join(trial.notes)}]" if trial.notes else ""
        print(
            f"  {trial.name:<18} {trial.label:<12} "
            f"{trial.data.shape[0]}ch x {trial.duration:5.1f}s @ {trial.fs:5.1f}Hz{notes}"
        )
    return trials


def cmd_train(args: argparse.Namespace) -> None:
    trials = _load(args)

    X, y, groups = epoch_dataset(
        trials,
        band=tuple(args.band),
        epoch_seconds=args.epoch_seconds,
        step_seconds=args.step_seconds,
    )
    print(
        f"\nEpoched into {X.shape[0]} x ({X.shape[1]} channels, {X.shape[2]} samples) "
        f"at {args.epoch_seconds:g}s / {args.step_seconds:g}s step, band {args.band[0]:g}-{args.band[1]:g} Hz"
    )

    result = leave_one_trial_out(
        X, y, groups, lambda: make_pipeline(args.n_components, args.shrinkage)
    )
    print()
    print(result.report())

    model = make_pipeline(args.n_components, args.shrinkage)
    model.fit(X, y)

    fs = float(np.mean([t.fs for t in trials]))
    payload = {
        "model": model,
        "fs": fs,
        "band": tuple(args.band),
        "epoch_seconds": args.epoch_seconds,
        "n_channels": X.shape[1],
        "channels": trials[0].channels,
        "classes": sorted(set(y.tolist())),
        "cv_accuracy": result.accuracy,
    }
    args.out.write_bytes(pickle.dumps(payload))
    print(f"\nFitted on all {X.shape[0]} epochs and saved to {args.out}")
    if result.accuracy <= result.chance + 0.05:
        print(
            "\nWARNING: cross-validated accuracy is at or near chance. Do not expect "
            "this model to fly the bird. Record more calibration data before relying on it."
        )


def cmd_sweep(args: argparse.Namespace) -> None:
    trials = _load(args)
    bands = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (8.0, 30.0), (1.0, 50.0)]
    lengths = [1.0, 2.0, 3.0]

    print(f"\n{'band':<14}{'epoch':<8}{'epochs':>8}{'acc':>9}{'trial':>9}{'chance':>9}{'p':>8}")
    print("-" * 65)
    for band in bands:
        for length in lengths:
            try:
                X, y, groups = epoch_dataset(
                    trials, band=band, epoch_seconds=length, step_seconds=args.step_seconds
                )
                result = leave_one_trial_out(
                    X, y, groups, lambda: make_pipeline(args.n_components, args.shrinkage)
                )
            except ValueError as exc:
                print(f"{f'{band[0]:g}-{band[1]:g} Hz':<14}{length:<8g}  skipped: {exc}")
                continue
            print(
                f"{f'{band[0]:g}-{band[1]:g} Hz':<14}{length:<8g}{result.n_epochs:>8}"
                f"{result.accuracy:>9.1%}{result.trial_accuracy:>9.1%}"
                f"{result.chance:>9.1%}{result.p_value:>8.3f}"
            )
    print(
        "\nPick on the trial column and the p-value, not the epoch column: with "
        "\noverlapping epochs the epoch accuracy is the more optimistic of the two."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    common.add_argument(
        "--labels", nargs="*", default=["hands", "legimagery"],
        help="classes to use; default is the two motor-imagery classes recorded in the same session",
    )
    common.add_argument("--band", nargs=2, type=float, default=[8.0, 30.0], metavar=("LOW", "HIGH"))
    common.add_argument("--epoch-seconds", type=float, default=DEFAULT_EPOCH_SECONDS)
    common.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS)
    common.add_argument("--n-components", type=int, default=4)
    common.add_argument("--shrinkage", type=float, default=0.1)

    audit_cmd = sub.add_parser("audit", help="report what is in the data directory")
    audit_cmd.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    audit_cmd.set_defaults(func=cmd_audit)

    train_cmd = sub.add_parser("train", parents=[common], help="train and cross-validate")
    train_cmd.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train_cmd.set_defaults(func=cmd_train)

    sweep_cmd = sub.add_parser("sweep", parents=[common], help="compare bands and epoch lengths")
    sweep_cmd.set_defaults(func=cmd_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
