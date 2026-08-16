"""Loading and auditing the hackathon's FlexEEG CSV recordings.

The CSVs are LSL buffer dumps with a three-line ``#`` comment header followed
by ``timestamp,relative_time,Ch1..Ch9``. Two things about them matter enough to
be handled here rather than left to the caller:

* The real recordings run at ~125.5 Hz, not the 250 Hz assumed by the original
  bandpass notebook. Getting this wrong halves every filter cutoff.
* ``Ch9`` is identically zero in every real recording, and one or two channels
  per recording sit 10-40x above the others in variance (a railing or
  disconnected electrode). Both wreck CSP if fed in unchecked.

Use :func:`audit` to see all of that for a directory before trusting it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# From the original notebook: 24-bit ADC over a 4.5 V range, expressed in µV.
# CSP and log-variance features are scale-invariant, so this is for readability
# and for plots, not for accuracy.
ADC_TO_MICROVOLTS = 1_000_000 * 4.5 / 16_777_215

_LABEL_RE = re.compile(r"^([A-Za-z]+?)\d", re.ASCII)


@dataclass
class Trial:
    """One continuous recording: ``data`` is (n_channels, n_samples)."""

    name: str
    label: str
    data: np.ndarray
    channels: list[str]
    fs: float
    t0: float
    source: Path
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.fs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Trial({self.name!r}, label={self.label!r}, "
            f"{self.data.shape[0]}ch x {self.data.shape[1]} @ {self.fs:.1f}Hz)"
        )


def label_from_filename(path: Path) -> str:
    """``LegImagery3.csv`` -> ``legimagery``. Case is folded because the data
    contains both ``LegImagery2`` and ``Legimagery2`` spellings."""
    match = _LABEL_RE.match(path.stem)
    stem = match.group(1) if match else path.stem
    return stem.lower()


def load_csv(path: str | Path) -> Trial:
    """Read one LSL CSV dump into a :class:`Trial`.

    Sampling rate is measured from the timestamp column rather than assumed.
    LSL delivers samples in chunks, so consecutive timestamp differences are
    bursty and useless; the mean rate across the whole file is what is stable.
    """
    path = Path(path)
    header: list[str] | None = None
    stamps: list[float] = []
    rows: list[list[float]] = []

    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if header is None and parts and parts[0] == "timestamp":
                header = parts[2:]
                continue
            if len(parts) < 3:
                continue
            try:
                values = [float(x) for x in parts]
            except ValueError:
                # A concatenated dump can repeat its header mid-file.
                continue
            stamps.append(values[0])
            rows.append(values[2:])

    if not rows:
        raise ValueError(f"{path} contains no data rows")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"{path} has ragged rows")

    stamp_array = np.asarray(stamps, dtype=float)
    data = np.asarray(rows, dtype=float).T
    span = stamp_array[-1] - stamp_array[0]
    if span <= 0:
        raise ValueError(f"{path} has a non-increasing timestamp column")
    fs = len(stamp_array) / span

    if header is None:
        header = [f"Ch{i + 1}" for i in range(width)]

    return Trial(
        name=path.stem,
        label=label_from_filename(path),
        data=data,
        channels=list(header),
        fs=fs,
        t0=float(stamp_array[0]),
        source=path,
    )


def _content_digest(trial: Trial) -> str:
    """Hash the timestamps and samples, so duplicates are caught even when the
    files differ in line endings or trailing whitespace."""
    payload = np.ascontiguousarray(np.round(trial.data, 6)).tobytes()
    return hashlib.sha1(f"{trial.t0:.6f}".encode() + payload).hexdigest()


def flag_bad_channels(
    data: np.ndarray, zero_tol: float = 1e-12, ratio: float = 5.0
) -> list[int]:
    """Indices of channels that are flat or wildly out of scale.

    A channel is bad if its standard deviation is ~zero (dead or aux, e.g.
    ``Ch9``), or more than ``ratio`` times the median channel deviation (a
    railing or floating electrode, e.g. ``Ch3`` in these recordings). CSP
    maximises variance ratios, so a single railing channel will otherwise
    dominate every component it produces.
    """
    deviations = data.std(axis=1)
    bad = set(np.flatnonzero(deviations <= zero_tol).tolist())
    live = deviations[deviations > zero_tol]
    if live.size:
        threshold = ratio * float(np.median(live))
        bad.update(np.flatnonzero(deviations > threshold).tolist())
    return sorted(bad)


def load_dataset(
    directory: str | Path,
    labels: list[str] | None = None,
    drop_bad_channels: bool = True,
    to_microvolts: bool = True,
    expected_channels: int = 9,
    fs_tolerance: float = 0.15,
) -> list[Trial]:
    """Load every CSV in ``directory`` that belongs to a usable recording.

    Recordings are excluded, with the reason recorded, when they are exact
    duplicates of an earlier file or do not match the FlexEEG channel layout
    (which is how the synthetic ``demo_eeg`` dump is filtered out).

    ``labels`` restricts the returned classes, e.g. ``["hands", "legimagery"]``.
    Bad channels are unioned across the kept trials so that every trial ends up
    with the same channel set -- CSP needs a fixed channel count.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV files under {directory}")

    kept: list[Trial] = []
    seen: dict[str, str] = {}

    for path in paths:
        trial = load_csv(path)

        if len(trial.channels) != expected_channels:
            trial.notes.append(
                f"excluded: {len(trial.channels)} channels "
                f"({','.join(trial.channels)}), not the {expected_channels}-channel FlexEEG layout"
            )
            continue

        digest = _content_digest(trial)
        if digest in seen:
            trial.notes.append(f"excluded: identical content to {seen[digest]}")
            continue
        seen[digest] = trial.name

        if labels is not None and trial.label not in labels:
            continue

        kept.append(trial)

    if not kept:
        raise ValueError(f"no usable recordings in {directory} for labels={labels}")

    rates = np.array([t.fs for t in kept])
    if float(np.ptp(rates)) / rates.mean() > fs_tolerance:
        detail = ", ".join(f"{t.name}={t.fs:.1f}Hz" for t in kept)
        raise ValueError(f"recordings disagree on sampling rate: {detail}")

    if drop_bad_channels:
        bad = sorted({i for t in kept for i in flag_bad_channels(t.data)})
        good = [i for i in range(expected_channels) if i not in bad]
        if len(good) < 2:
            raise ValueError("fewer than two usable channels survived screening")
        for trial in kept:
            dropped = [trial.channels[i] for i in bad]
            trial.data = trial.data[good]
            trial.channels = [trial.channels[i] for i in good]
            if dropped:
                trial.notes.append(f"dropped channels {','.join(dropped)}")

    if to_microvolts:
        for trial in kept:
            trial.data = trial.data * ADC_TO_MICROVOLTS

    return kept


def audit(directory: str | Path) -> str:
    """A human-readable report of every CSV: rate, clock, channels, exclusions.

    Worth reading before training. It is what surfaces the duplicate ``Rest1``,
    the synthetic ``Hands4``, and the fact that the rest recordings come from a
    different session clock than the motor-imagery ones.
    """
    directory = Path(directory)
    lines = [f"{'file':<20}{'label':<12}{'n':>6}{'rate':>8}{'clock':>11}  channels / notes"]
    lines.append("-" * 96)

    seen: dict[str, str] = {}
    clocks: dict[str, list[float]] = {}

    for path in sorted(directory.glob("*.csv")):
        try:
            trial = load_csv(path)
        except ValueError as exc:
            lines.append(f"{path.name:<20}{'-':<12}{'':>6}{'':>8}{'':>11}  unreadable: {exc}")
            continue

        notes: list[str] = []
        if len(trial.channels) != 9:
            notes.append(f"NOT FlexEEG layout ({','.join(trial.channels)}) -- excluded")

        digest = _content_digest(trial)
        if digest in seen:
            notes.append(f"DUPLICATE of {seen[digest]} -- excluded")
        else:
            seen[digest] = trial.name

        bad = flag_bad_channels(trial.data)
        if bad:
            notes.append("bad channels: " + ",".join(trial.channels[i] for i in bad))

        clocks.setdefault(trial.label, []).append(trial.t0)

        lines.append(
            f"{path.name:<20}{trial.label:<12}{len(trial.data[0]):>6}"
            f"{trial.fs:>8.1f}{trial.t0:>11.0f}  {'; '.join(notes) if notes else 'ok'}"
        )

    spans = {k: (min(v), max(v)) for k, v in clocks.items()}
    lines.append("")
    lines.append("LSL clock range per label (a disjoint range means a separate session):")
    for label, (lo, hi) in sorted(spans.items()):
        lines.append(f"  {label:<12} {lo:>10.0f} .. {hi:<10.0f}")

    overlapping = [
        (a, b)
        for i, (a, (a_lo, a_hi)) in enumerate(sorted(spans.items()))
        for b, (b_lo, b_hi) in sorted(spans.items())[i + 1 :]
        if not (a_hi < b_lo - 60 or b_hi < a_lo - 60)
    ]
    if len(spans) > 1:
        for a, b in overlapping:
            lines.append(f"  -> {a} and {b} share a session: comparable.")
        for a, _ in sorted(spans.items()):
            for b, _ in sorted(spans.items()):
                if a < b and (a, b) not in overlapping:
                    lines.append(
                        f"  -> {a} and {b} are from DIFFERENT sessions: a classifier "
                        f"separating them may be reading session, not intent."
                    )

    return "\n".join(lines)
