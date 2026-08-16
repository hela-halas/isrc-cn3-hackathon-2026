#!/usr/bin/env python3
"""Real-time motor-imagery classifier: LSL EEG in, flap commands out.

    python realtime_bci.py simulate --csv "EEG Hackathon data bios and one/Hands2.csv"
    python realtime_bci.py run --output udp
    python realtime_bci.py run --output lsl --target-class hands

``simulate`` replays a recorded CSV through the exact live code path, so the
whole loop can be tested without the headset or the game. Get that working
first; the plumbing is what usually breaks on the day, not the maths.

The output protocol matches the dummy senders already in this repo, so Unity
needs no changes:
  * LSL  -- stream ``MotorImagery`` of type ``Markers``, 1.0 = flap, 0.0 = idle
  * UDP  -- the string "1" to 127.0.0.1:5005

The classifier decides on a sliding window, so it emits a decision several
times a second while each decision still reflects a couple of seconds of EEG.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np

from bci.data import ADC_TO_MICROVOLTS, load_csv
from bci.filters import StreamingBandpass

DEFAULT_MODEL = Path(__file__).parent / "model.pkl"
UDP_HOST, UDP_PORT, UDP_FLAP = "127.0.0.1", 5005, "1"
LSL_NAME, LSL_TYPE, LSL_FLAP, LSL_IDLE = "MotorImagery", "Markers", 1.0, 0.0


class Smoother:
    """Exponential moving average over the classifier's probability output.

    Raw per-window predictions are jittery, and a bird that twitches on every
    window is unplayable. This is the one place a Gaussian-ish smoother belongs
    -- on the control signal, not on the EEG. ``alpha`` trades steadiness
    against latency: lower is calmer and laggier.
    """

    def __init__(self, alpha: float = 0.3, initial: float = 0.5) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = initial

    def __call__(self, probability: float) -> float:
        self.value = self.alpha * probability + (1 - self.alpha) * self.value
        return self.value


class Output:
    """Sends flap commands to the game over LSL or UDP."""

    def __init__(self, kind: str, verbose: bool = True) -> None:
        self.kind = kind
        self.verbose = verbose
        self._socket = None
        self._outlet = None

        if kind == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Sending flap commands over UDP to {UDP_HOST}:{UDP_PORT}")
        elif kind == "lsl":
            from pylsl import StreamInfo, StreamOutlet

            info = StreamInfo(LSL_NAME, LSL_TYPE, 1, 0, "float32", "bci-classifier-001")
            self._outlet = StreamOutlet(info)
            print(f"Publishing LSL stream {LSL_NAME!r} (type {LSL_TYPE!r})")
        elif kind == "none":
            print("Output disabled; decisions will only be printed")
        else:
            raise ValueError(f"unknown output {kind!r}")

    def flap(self) -> None:
        if self._socket is not None:
            self._socket.sendto(UDP_FLAP.encode(), (UDP_HOST, UDP_PORT))
        elif self._outlet is not None:
            self._outlet.push_sample([LSL_FLAP])
            time.sleep(0.05)
            self._outlet.push_sample([LSL_IDLE])

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()


class Decoder:
    """Sliding-window decoder: feed it chunks, it yields flap decisions.

    Holds the streaming filter state and the sample ring buffer, so this is the
    single object that has to behave identically offline and live.
    """

    def __init__(
        self,
        model_path: Path,
        target_class: str | None = None,
        threshold: float = 0.6,
        alpha: float = 0.3,
        refractory: float = 0.4,
    ) -> None:
        payload = pickle.loads(model_path.read_bytes())
        self.model = payload["model"]
        self.fs = payload["fs"]
        self.band = payload["band"]
        self.classes = payload["classes"]
        self.n_channels = payload["n_channels"]
        self.window = int(round(payload["epoch_seconds"] * self.fs))

        self.target = target_class or self.classes[0]
        if self.target not in self.classes:
            raise ValueError(f"target {self.target!r} not in {self.classes}")
        self.target_index = list(self.model.classes_).index(self.target)

        self.threshold = threshold
        self.refractory = refractory
        self.bandpass = StreamingBandpass(self.fs, self.n_channels, self.band)
        self.smoother = Smoother(alpha)
        self.buffer = np.zeros((self.n_channels, 0))
        self._last_flap = 0.0

        print(
            f"Loaded {model_path.name}: {self.n_channels} channels @ {self.fs:.1f} Hz, "
            f"band {self.band[0]:g}-{self.band[1]:g} Hz, classes {self.classes}, "
            f"flapping on {self.target!r} (CV accuracy {payload['cv_accuracy']:.1%})"
        )

    def push(self, chunk: np.ndarray) -> None:
        """Filter and buffer a ``(n_channels, n_samples)`` chunk."""
        filtered = self.bandpass(chunk)
        self.buffer = np.concatenate([self.buffer, filtered], axis=1)
        if self.buffer.shape[1] > self.window * 2:
            self.buffer = self.buffer[:, -self.window :]

    def ready(self) -> bool:
        return self.buffer.shape[1] >= self.window

    def decide(self, now: float | None = None) -> tuple[float, bool]:
        """Return ``(smoothed_probability, should_flap)`` for the newest window."""
        if not self.ready():
            raise RuntimeError("not enough samples buffered yet")
        now = time.monotonic() if now is None else now

        epoch = self.buffer[:, -self.window :][None, :, :]
        probability = float(self.model.predict_proba(epoch)[0, self.target_index])
        smoothed = self.smoother(probability)

        should_flap = (
            smoothed >= self.threshold and now - self._last_flap >= self.refractory
        )
        if should_flap:
            self._last_flap = now
        return smoothed, should_flap


def _bar(value: float, width: int = 28) -> str:
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Replay a recorded CSV through the live decoder, chunk by chunk."""
    decoder = Decoder(args.model, args.target_class, args.threshold, args.alpha)
    trial = load_csv(args.csv)

    # Match the channel screening train_classifier.py applied. The model was fit
    # on a specific channel subset, so the live stream has to be cut the same way.
    keep = [i for i, name in enumerate(trial.channels) if name in decoder_channels(args.model)]
    if len(keep) != decoder.n_channels:
        raise SystemExit(
            f"{args.csv.name} has channels {trial.channels}, but the model expects "
            f"{decoder_channels(args.model)}. Retrain, or pass a matching recording."
        )
    data = trial.data[keep] * ADC_TO_MICROVOLTS

    chunk_size = max(1, int(round(args.chunk_seconds * decoder.fs)))
    interval = 1.0 / args.decisions_per_second
    print(
        f"Replaying {trial.name} ({trial.label}) -- {data.shape[1] / decoder.fs:.1f}s, "
        f"{chunk_size}-sample chunks\n"
    )

    flaps = 0
    next_decision = 0.0
    for start in range(0, data.shape[1] - chunk_size + 1, chunk_size):
        decoder.push(data[:, start : start + chunk_size])
        clock = (start + chunk_size) / decoder.fs
        if not decoder.ready() or clock < next_decision:
            continue
        next_decision = clock + interval

        probability, should_flap = decoder.decide(now=clock)
        flaps += should_flap
        print(
            f"  t={clock:5.1f}s  p({decoder.target})={probability:5.2f} "
            f"[{_bar(probability)}] {'FLAP' if should_flap else '    '}"
        )
        if args.realtime:
            time.sleep(chunk_size / decoder.fs)

    print(f"\n{flaps} flaps over {data.shape[1] / decoder.fs:.1f}s of {trial.label!r} data")
    print(
        f"Sanity check: replaying a {decoder.target!r} recording should flap often, "
        f"and replaying the other class should flap rarely."
    )


def decoder_channels(model_path: Path) -> list[str]:
    return pickle.loads(model_path.read_bytes())["channels"]


def cmd_run(args: argparse.Namespace) -> None:
    """Connect to a live LSL EEG stream and drive the game."""
    try:
        from pylsl import StreamInlet, resolve_streams
    except ImportError:
        raise SystemExit("pylsl is not installed. Run: pip install pylsl")

    decoder = Decoder(args.model, args.target_class, args.threshold, args.alpha)
    output = Output(args.output)
    expected = decoder_channels(args.model)

    print(f"Waiting for an EEG stream (up to {args.timeout:g}s) ...")
    deadline = time.monotonic() + args.timeout
    inlet = None
    while time.monotonic() < deadline and inlet is None:
        for stream in resolve_streams(wait_time=1.0):
            if stream.name() == args.eeg_stream or (
                args.eeg_stream is None and stream.type().lower() == "eeg"
            ):
                inlet = StreamInlet(stream, max_buflen=30, processing_flags=0)
                print(
                    f"Connected to {stream.name()!r}: {stream.channel_count()} channels "
                    f"at {stream.nominal_srate():g} Hz"
                )
                break
    if inlet is None:
        raise SystemExit("No EEG stream appeared. Run `python lsl_starter.py streams` to look.")

    rate = inlet.info().nominal_srate()
    if rate > 0 and abs(rate - decoder.fs) / decoder.fs > 0.1:
        print(
            f"\nWARNING: the stream runs at {rate:g} Hz but the model was trained at "
            f"{decoder.fs:.1f} Hz. Every filter cutoff is off by {rate / decoder.fs:.2f}x. "
            f"Retrain on data from this headset.\n"
        )

    channels = list(range(decoder.n_channels))
    if inlet.info().channel_count() < decoder.n_channels:
        raise SystemExit(
            f"Stream has {inlet.info().channel_count()} channels, model needs {decoder.n_channels}"
        )
    print(f"Using stream channels {channels} as {expected}\nRunning. Ctrl-C to stop.\n")

    interval = 1.0 / args.decisions_per_second
    next_decision = time.monotonic() + interval
    try:
        while True:
            samples, _ = inlet.pull_chunk(timeout=0.1)
            if samples:
                decoder.push(np.asarray(samples, dtype=float).T[channels])

            now = time.monotonic()
            if now < next_decision or not decoder.ready():
                continue
            next_decision = now + interval

            probability, should_flap = decoder.decide(now)
            if should_flap:
                output.flap()
            print(
                f"\r  p({decoder.target})={probability:5.2f} [{_bar(probability)}] "
                f"{'FLAP' if should_flap else '    '}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        output.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    common.add_argument("--target-class", help="class that makes the bird flap")
    common.add_argument(
        "--threshold", type=float, default=0.6,
        help="smoothed probability above which a flap is sent (default 0.6)",
    )
    common.add_argument(
        "--alpha", type=float, default=0.3,
        help="EMA smoothing on the probability; lower is calmer and laggier (default 0.3)",
    )
    common.add_argument("--decisions-per-second", type=float, default=4.0)

    sim = sub.add_parser("simulate", parents=[common], help="replay a CSV through the live path")
    sim.add_argument("--csv", type=Path, required=True)
    sim.add_argument("--chunk-seconds", type=float, default=0.1)
    sim.add_argument("--realtime", action="store_true", help="replay at wall-clock speed")
    sim.set_defaults(func=cmd_simulate)

    run = sub.add_parser("run", parents=[common], help="classify a live LSL stream")
    run.add_argument("--eeg-stream", help="stream name; omit to take the sole EEG stream")
    run.add_argument("--output", choices=["lsl", "udp", "none"], default="udp")
    run.add_argument("--timeout", type=float, default=30.0)
    run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Run: python train_classifier.py train")
    args.func(args)


if __name__ == "__main__":
    main()
