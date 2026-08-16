"""Cued calibration recorder -- run this FIRST on hackathon day.

The supplied CSVs cannot train a usable classifier (see diagnose_confound.py:
every rest recording is from a different machine than every imagery recording,
so the label and the session are the same variable).  You need your own data,
recorded from one subject, in one sitting, with the classes interleaved.

This script cues the subject through alternating trials and writes one CSV per
trial, in the same format as the supplied data, so the rest of the pipeline
reads it unchanged.

Usage:
    python3 record_calibration.py --subject alice --classes rest,hands
    python3 record_calibration.py --subject alice --trials 30 --trial-s 6

Design notes, which matter more than the code:
  * Classes are INTERLEAVED, never blocked.  Recording all the rest trials and
    then all the hands trials reintroduces exactly the confound that ruins the
    supplied data -- drift, impedance changes and fatigue all track time.
  * Every trial has a cue period and a rest period.  The classifier is trained
    on the cue period only.
  * Keep the headset on for the whole session.  Removing and replacing it
    changes the montage and invalidates everything recorded before.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from pylsl import StreamInlet, resolve_streams
except ImportError:  # pragma: no cover
    StreamInlet = resolve_streams = None


def resolve_eeg(name: str | None, timeout: float) -> "StreamInlet":
    """Connect to an EEG stream by name, or the sole stream of type EEG."""
    if resolve_streams is None:
        raise SystemExit("pylsl is not installed. Run: pip install pylsl")

    deadline = time.monotonic() + timeout
    target = f"named {name!r}" if name else "with type 'EEG'"
    print(f"Waiting for an EEG LSL stream {target} ...")
    while time.monotonic() < deadline:
        for stream in resolve_streams(wait_time=1.0):
            match = stream.name() == name if name else stream.type().lower() == "eeg"
            if match:
                print(f"Connected to {stream.name()!r}: {stream.channel_count()} "
                      f"channels at {stream.nominal_srate():g} Hz")
                return StreamInlet(stream, max_buflen=60, processing_flags=0)
    raise TimeoutError(f"No LSL stream {target} appeared within {timeout:g}s")


def countdown(message: str, seconds: float) -> None:
    for remaining in range(int(seconds), 0, -1):
        print(f"\r{message} {remaining}s ", end="", flush=True)
        time.sleep(1)
    print(f"\r{message} now!    ", flush=True)


def record_window(inlet, seconds: float):
    """Pull `seconds` of samples, returning (timestamps, samples)."""
    inlet.flush()
    samples, stamps = [], []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chunk, ts = inlet.pull_chunk(timeout=0.2, max_samples=256)
        if chunk:
            samples.extend(chunk)
            stamps.extend(ts)
    return np.asarray(stamps), np.asarray(samples, dtype=float)


def write_csv(path: Path, stamps, samples, label: str, subject: str) -> None:
    """Write in the same shape as the supplied FlexEEG exports."""
    n_ch = samples.shape[1] if samples.ndim == 2 else 0
    header_cols = ",".join(f"Ch{i + 1}" for i in range(n_ch))
    t0 = stamps[0] if len(stamps) else 0.0
    with path.open("w") as fh:
        fh.write(f"# Stream: FlexEEG (EEG) calibration subject={subject} label={label}\n")
        fh.write(f"# Recorded: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# Samples: {len(stamps)}\n")
        fh.write(f"timestamp,relative_time,{header_cols}\n")
        for t, row in zip(stamps, samples):
            values = ",".join(f"{v:g}" for v in row)
            fh.write(f"{t:.7f},{t - t0:.4f},{values}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="subject id, used in filenames")
    parser.add_argument("--classes", default="rest,hands",
                        help="comma-separated class labels to cue")
    parser.add_argument("--trials", type=int, default=20,
                        help="total trials, split evenly across classes")
    parser.add_argument("--trial-s", type=float, default=5.0,
                        help="seconds of imagery recorded per trial")
    parser.add_argument("--rest-s", type=float, default=3.0,
                        help="seconds of rest between trials")
    parser.add_argument("--out", default="calibration", help="output directory")
    parser.add_argument("--eeg-stream", help="LSL stream name; omit to auto-select")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) < 2:
        raise SystemExit("Need at least two classes, e.g. --classes rest,hands")

    out_dir = Path(args.out) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    inlet = resolve_eeg(args.eeg_stream, args.timeout)

    # Interleave, then shuffle within the balanced sequence.
    per_class = max(1, args.trials // len(classes))
    order = classes * per_class
    random.shuffle(order)

    print(f"\n{len(order)} trials, {args.trial_s:g}s each, classes: {classes}")
    print("Keep the headset on for the whole session. Ctrl-C to stop early.\n")
    input("Press Enter when the subject is settled and ready...")

    try:
        for i, label in enumerate(order, 1):
            countdown(f"[{i}/{len(order)}] Relax...", args.rest_s)
            print(f"[{i}/{len(order)}] IMAGINE: {label.upper()} "
                  f"({args.trial_s:g}s) -- go")
            stamps, samples = record_window(inlet, args.trial_s)

            if len(stamps) < 10:
                print("  !! almost no samples arrived -- check the headset link")
                continue

            path = out_dir / f"{label}{i:03d}.csv"
            write_csv(path, stamps, samples, label, args.subject)
            rate = len(stamps) / (stamps[-1] - stamps[0]) if stamps[-1] > stamps[0] else 0
            print(f"  saved {path.name}  ({len(stamps)} samples, ~{rate:.0f} Hz)")
    except KeyboardInterrupt:
        print("\nStopped early.")

    print(f"\nDone. Now train on it:\n  python3 train_offline.py --data-dir {out_dir} --save")


if __name__ == "__main__":
    main()
