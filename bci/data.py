"""Loading and cleaning for the NeuroCONCISE FlexEEG CSV exports.

The recordings in "EEG Hackathon data bios and one/" are LSL buffer dumps, and
they are messier than they look:

  * Samples arrive in bursts, not on a clock.  The median inter-sample gap is
    ~1.4 ms but the *effective* rate is ~125 Hz, so anything that assumes a
    uniform grid (every filter in scipy.signal) has to resample first.
  * One file (Rest1.csv) contains two concatenated recordings, each with its
    own "# Stream:" header block.
  * Ch9 is dead (identically zero in every file).
  * Ch3 carries an amplifier artifact an order of magnitude larger than the
    neural signal, and Ch2 does the same in several of the Rest files.
  * Hands4.csv is from a different device entirely: 250 Hz, eight *named*
    channels (Fp1..O2), amplitudes ~20x smaller.

`load_recordings` normalises all of that into uniformly-sampled segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channels that are unusable in the Ch1..Ch9 recordings.
#   Ch9 - identically zero in every file
#   Ch3 - amplifier artifact, sd ~3000-4000 counts vs ~80 for a clean channel
#   Ch2 - same failure mode, but only in some Rest recordings
DEAD_CHANNELS = ("Ch9",)
ARTIFACT_CHANNELS = ("Ch3", "Ch2")

# ADC -> microvolts, per the conversion used in bandpass.ipynb
ADC_TO_UV = 1_000_000 * 4.5 / 16_777_215


@dataclass
class Recording:
    """One uniformly-resampled segment of EEG."""

    label: str  # "hands" | "legs" | "rest"
    source: str  # file name, used as the cross-validation group
    host: str  # LSL hostname -> proxy for recording session
    channels: list[str]
    data: np.ndarray  # (n_channels, n_samples), microvolts
    fs: float

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.fs


def label_from_filename(name: str) -> str:
    stem = name.lower()
    if stem.startswith("hands"):
        return "hands"
    if stem.startswith("leg"):
        return "legs"
    if stem.startswith("rest"):
        return "rest"
    raise ValueError(f"Cannot infer a label from {name!r}")


def _parse_blocks(path: Path):
    """Split one CSV into (host, channel_names, times, samples) blocks.

    A new block starts at every "# Stream:" header, because Rest1.csv is two
    recordings glued together and they must not be filtered as one signal.
    """
    blocks = []
    host, cols, times, rows = "unknown", None, [], []

    def flush():
        if cols and len(rows) > 10:
            blocks.append((host, cols, np.asarray(times), np.asarray(rows, dtype=float)))

    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Stream:"):
            flush()
            host = line.split("from")[-1].strip()
            cols, times, rows = None, [], []
            continue
        if line.startswith("#"):
            continue
        if line.startswith("timestamp"):
            cols = line.split(",")[2:]
            continue
        if cols is None:
            continue
        parts = line.split(",")
        try:
            t = float(parts[1])
            values = [float(v) for v in parts[2 : 2 + len(cols)]]
        except (ValueError, IndexError):
            continue
        if len(values) != len(cols):
            continue
        times.append(t)
        rows.append(values)

    flush()
    return blocks


def _resample_uniform(times: np.ndarray, values: np.ndarray, fs: float) -> np.ndarray:
    """Interpolate irregular LSL samples onto a uniform grid at `fs` Hz.

    values is (n_samples, n_channels); the result is (n_channels, n_uniform).
    """
    order = np.argsort(times)
    times, values = times[order], values[order]
    # Drop duplicate timestamps -- np.interp requires strictly increasing x.
    keep = np.concatenate(([True], np.diff(times) > 0))
    times, values = times[keep], values[keep]

    grid = np.arange(times[0], times[-1], 1.0 / fs)
    out = np.empty((values.shape[1], grid.size))
    for ch in range(values.shape[1]):
        out[ch] = np.interp(grid, times, values[:, ch])
    return out


def estimate_rate(times: np.ndarray) -> float:
    """Effective rate = samples / wall-clock span, not 1 / median gap.

    The bursty delivery makes the median gap read ~700 Hz when the device is
    really producing ~125 Hz.
    """
    span = times[-1] - times[0]
    return len(times) / span if span > 0 else 0.0


def load_recordings(
    data_dir: str | Path,
    fs: float = 125.0,
    drop_artifact_channels: bool = True,
    exclude_named_montage: bool = True,
) -> list[Recording]:
    """Load every CSV in `data_dir` as one or more uniformly-sampled segments.

    exclude_named_montage skips Hands4.csv, whose Fp1..O2 montage and 250 Hz
    rate make it incomparable with the rest of the set.
    """
    data_dir = Path(data_dir)
    recordings: list[Recording] = []

    for path in sorted(data_dir.glob("*.csv")):
        label = label_from_filename(path.name)
        for host, cols, times, values in _parse_blocks(path):
            named_montage = "Ch1" not in cols
            if named_montage and exclude_named_montage:
                continue

            drop = set(DEAD_CHANNELS)
            if drop_artifact_channels and not named_montage:
                drop |= set(ARTIFACT_CHANNELS)
            keep_idx = [i for i, c in enumerate(cols) if c not in drop]
            if not keep_idx:
                continue

            uniform = _resample_uniform(times, values[:, keep_idx], fs)
            if not named_montage:
                uniform = uniform * ADC_TO_UV

            recordings.append(
                Recording(
                    label=label,
                    source=path.name,
                    host=host,
                    channels=[cols[i] for i in keep_idx],
                    data=uniform,
                    fs=fs,
                )
            )

    return recordings


def summarise(recordings: list[Recording]) -> str:
    lines = [f"{'file':22s} {'label':6s} {'host':18s} {'chans':6s} {'dur':>6s}"]
    for r in recordings:
        lines.append(
            f"{r.source:22s} {r.label:6s} {r.host:18s} "
            f"{len(r.channels):<6d} {r.duration:5.1f}s"
        )
    return "\n".join(lines)
