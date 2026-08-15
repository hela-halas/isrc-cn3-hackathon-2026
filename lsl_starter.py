#!/usr/bin/env python3
"""EEG LSL starter code.

This is intentionally minimal. It only knows how to:
  1. List visible LSL streams on the network.
  2. Connect to an EEG stream by name (or auto-select the only EEG stream).
  3. Pull a fixed-length window of raw samples from that stream.

Everything else -- filtering, feature extraction, classification,
calibration UI, and any output back to Unity or elsewhere -- is left
for you to build on top of this.

Usage:
  python lsl_starter.py streams
  python lsl_starter.py grab --seconds 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pylsl import StreamInlet, resolve_streams

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def visible_streams(wait: float = 2.0):
    """Print every LSL stream currently visible on the network."""
    streams = resolve_streams(wait_time=wait)
    for s in streams:
        print(
            f"name={s.name()!r} type={s.type()!r} channels={s.channel_count()} "
            f"rate={s.nominal_srate():g} source_id={s.source_id()!r}"
        )
    return streams


def resolve_eeg(name: str | None, timeout: float) -> StreamInlet:
    """Find and connect to an EEG stream, either by exact name or by type."""
    deadline = time.monotonic() + timeout
    target = f"named {name!r}" if name else "with type 'EEG'"
    print(f"Waiting for an EEG LSL stream {target} ...")
    while time.monotonic() < deadline:
        streams = resolve_streams(wait_time=1.0)
        matches = [
            stream
            for stream in streams
            if (stream.name() == name if name else stream.type().lower() == "eeg")
        ]
        if matches:
            stream = matches[0]
            print(
                f"Connected to {stream.name()!r}: {stream.channel_count()} channels at "
                f"{stream.nominal_srate():g} Hz"
            )
            return StreamInlet(stream, max_buflen=30, processing_flags=0)
    raise TimeoutError(f"No LSL stream {target} appeared within {timeout:g} s")


def parse_channels(text: str | None, channel_count: int) -> np.ndarray:
    """Turn a comma-separated channel list into an index array (or use all channels)."""
    if not text:
        return np.arange(channel_count, dtype=int)
    channels = np.asarray([int(x.strip()) for x in text.split(",")], dtype=int)
    if channels.min() < 0 or channels.max() >= channel_count:
        raise ValueError(f"Channel indices must be between 0 and {channel_count - 1}")
    return channels


def collect_window(inlet: StreamInlet, seconds: float, fs: float, channels: np.ndarray):
    """Block until a window of `seconds` worth of samples has been collected.

    Returns an array shaped (n_channels, n_samples).
    """
    wanted = int(round(seconds * fs))
    samples: list[list[float]] = []
    deadline = time.monotonic() + seconds + 5.0
    while len(samples) < wanted and time.monotonic() < deadline:
        chunk, _ = inlet.pull_chunk(timeout=0.25, max_samples=wanted - len(samples))
        samples.extend(chunk)
    if len(samples) < wanted:
        raise RuntimeError(f"Only received {len(samples)}/{wanted} EEG samples")
    data = np.asarray(samples[:wanted], dtype=float).T
    return data[channels]


def grab(args):
    """Connect to an EEG stream and pull one window of samples, printing its shape."""
    inlet = resolve_eeg(args.eeg_stream, args.timeout)
    info = inlet.info()
    fs = args.sampling_rate or info.nominal_srate()
    if fs <= 0:
        raise ValueError("The stream has no nominal rate; supply --sampling-rate")
    channels = parse_channels(args.channels, info.channel_count())
    epoch = collect_window(inlet, args.seconds, fs, channels)
    print(f"Collected epoch with shape {epoch.shape} (channels x samples) at {fs:g} Hz")

    # ------------------------------------------------------------------
    # SANITY-CHECK SCAFFOLDING -- delete this block once you trust that
    # data is actually arriving and start building your own pipeline.
    # ------------------------------------------------------------------
    print("First few raw samples per channel:")
    for ch_index, ch_data in zip(channels, epoch):
        preview = ", ".join(f"{v:.2f}" for v in ch_data[:5])
        print(f"  ch{ch_index}: {preview} ...")

    if not args.no_plot:
        if plt is None:
            print("matplotlib is not installed; skipping plot (pip install matplotlib)")
        else:
            time_axis = np.arange(epoch.shape[1]) / fs
            fig, ax = plt.subplots()
            for ch_index, ch_data in zip(channels, epoch):
                ax.plot(time_axis, ch_data, label=f"ch{ch_index}")
            ax.set_xlabel("seconds")
            ax.set_ylabel("amplitude")
            ax.set_title("Raw EEG window (sanity check only)")
            ax.legend(loc="upper right", fontsize="small")
            plt.show()
    # ------------------------------------------------------------------
    # END SANITY-CHECK SCAFFOLDING
    # ------------------------------------------------------------------

    # TODO: this is where your own processing / classification pipeline starts.


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    streams = sub.add_parser("streams", help="list visible LSL streams")
    streams.add_argument("--wait", type=float, default=2.0)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--eeg-stream",
        help="LSL stream name; omit to auto-select the sole stream of type EEG",
    )
    common.add_argument("--timeout", type=float, default=30.0)

    grab_cmd = sub.add_parser("grab", parents=[common], help="pull one window of raw samples")
    grab_cmd.add_argument("--sampling-rate", type=float)
    grab_cmd.add_argument("--channels", help="zero-based indices, e.g. 0,1,2,3,4,5,6,7")
    grab_cmd.add_argument("--seconds", type=float, default=3.0)
    grab_cmd.add_argument(
        "--no-plot", action="store_true", help="skip the sanity-check plot of the raw epoch"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        # No subcommand given (e.g. just pressing "Run") -- default to a
        # quick connect-and-grab so it's obvious the setup is working.
        args = parser.parse_args(["grab"])
    if args.command == "streams":
        visible_streams(args.wait)
    elif args.command == "grab":
        grab(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()