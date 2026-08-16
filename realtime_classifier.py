"""Real-time classifier: EEG in over LSL, flap out over LSL or UDP.

This is the piece that lsl_starter.py stops just short of ("TODO: this is where
your own processing / classification pipeline starts").

    python3 realtime_classifier.py --output lsl --flap-class hands
    python3 realtime_classifier.py --output udp --flap-class hands
    python3 realtime_classifier.py --output lsl --dry-run   # print, don't send

The output side matches the dummy senders exactly, so the Unity game cannot
tell this apart from lsl_dummy_input.py / udp_dummy_input.py:
  * LSL: a 1-channel "MotorImagery" Markers stream, 1.0 to flap then 0.0.
  * UDP: the string "1" to 127.0.0.1:5005.

Three things keep it usable rather than merely working:

  * A ring buffer, so features are computed on a sliding window and the decision
    rate is decoupled from the chunk size LSL happens to deliver.
  * Majority voting over the last N predictions.  A per-window classifier at 70%
    accuracy flickers badly; voting trades latency for stability, and in a game
    where a spurious flap costs you a red neuron that is the right trade.
  * A refractory period, so one sustained thought produces one flap rather than
    a continuous burst.
"""

from __future__ import annotations

import argparse
import socket
import time
from collections import deque

import numpy as np

from bci.features import log_band_power, preprocess
from bci.model import TrainedModel

try:
    from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_streams
except ImportError:  # pragma: no cover
    StreamInfo = StreamInlet = StreamOutlet = resolve_streams = None

FLAP_VALUE, IDLE_VALUE = 1.0, 0.0
UDP_HOST, UDP_PORT, UDP_FLAP = "127.0.0.1", 5005, "1"


class FlapSender:
    """Emits a flap over LSL, UDP, or nowhere (dry run)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.outlet = self.sock = None

        if mode == "lsl":
            if StreamOutlet is None:
                raise SystemExit("pylsl is not installed. Run: pip install pylsl")
            info = StreamInfo("MotorImagery", "Markers", 1, 0, "float32", "bci-live-001")
            self.outlet = StreamOutlet(info)
            print("Publishing LSL stream 'MotorImagery' (type 'Markers')")
        elif mode == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Sending flap packets to {UDP_HOST}:{UDP_PORT}")
        else:
            print("Dry run: decisions printed, nothing sent")

    def flap(self) -> None:
        if self.mode == "lsl":
            self.outlet.push_sample([FLAP_VALUE])
            time.sleep(0.05)
            self.outlet.push_sample([IDLE_VALUE])
        elif self.mode == "udp":
            self.sock.sendto(UDP_FLAP.encode("utf-8"), (UDP_HOST, UDP_PORT))

    def close(self) -> None:
        if self.sock:
            self.sock.close()


def resolve_eeg(name: str | None, timeout: float):
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
                return StreamInlet(stream, max_buflen=30, processing_flags=0)
    raise TimeoutError(f"No LSL stream {target} appeared within {timeout:g}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--output", choices=["lsl", "udp", "none"], default="lsl")
    parser.add_argument("--dry-run", action="store_true",
                        help="same as --output none")
    parser.add_argument("--flap-class", help="class that triggers a flap "
                                             "(default: first non-rest class)")
    parser.add_argument("--eeg-stream", help="LSL stream name; omit to auto-select")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--decision-hz", type=float, default=8.0,
                        help="classifications per second")
    parser.add_argument("--vote", type=int, default=5,
                        help="majority vote over this many recent decisions")
    parser.add_argument("--refractory", type=float, default=0.4,
                        help="minimum seconds between flaps")
    parser.add_argument("--channels", help="zero-based indices to keep, e.g. 0,3,4,5")
    args = parser.parse_args()

    model = TrainedModel.load(args.model)
    print(f"Model: classes={model.classes} fs={model.fs:g} "
          f"window={model.window_s:g}s channels={model.n_channels}")

    flap_class = args.flap_class
    if flap_class is None:
        non_rest = [c for c in model.classes if c != "rest"]
        flap_class = non_rest[0] if non_rest else model.classes[0]
    if flap_class not in model.classes:
        raise SystemExit(f"--flap-class {flap_class!r} not in {model.classes}")
    print(f"Flapping on: {flap_class!r}")

    inlet = resolve_eeg(args.eeg_stream, args.timeout)
    sender = FlapSender("none" if args.dry_run else args.output)

    keep = None
    if args.channels:
        keep = [int(x) for x in args.channels.split(",")]

    win_samples = int(model.window_s * model.fs)
    buffer: deque = deque(maxlen=win_samples)
    votes: deque = deque(maxlen=max(1, args.vote))
    interval = 1.0 / args.decision_hz
    next_decision = time.monotonic() + interval
    last_flap = 0.0
    n_flaps = 0

    print("\nRunning. Ctrl-C to stop.\n")
    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.05, max_samples=128)
            if chunk:
                for sample in chunk:
                    row = np.asarray(sample, dtype=float)
                    buffer.append(row[keep] if keep is not None else row)

            now = time.monotonic()
            if now < next_decision or len(buffer) < win_samples:
                continue
            next_decision = now + interval

            window = np.asarray(buffer).T  # (n_channels, n_samples)
            if window.shape[0] != model.n_channels:
                print(f"\n!! stream has {window.shape[0]} channels, model expects "
                      f"{model.n_channels}. Use --channels to select.")
                break

            clean = preprocess(window, model.fs)
            features = log_band_power(clean, model.fs).reshape(1, -1)
            prediction = model.pipeline.predict(features)[0]
            votes.append(prediction)

            # Majority vote, then refractory gate.
            labels, counts = np.unique(votes, return_counts=True)
            decision = labels[counts.argmax()]
            settled = len(votes) == votes.maxlen and decision == flap_class

            if settled and (now - last_flap) >= args.refractory:
                sender.flap()
                last_flap = now
                n_flaps += 1
                print(f"\rFLAP  #{n_flaps:<4d} (vote: {decision})           ",
                      end="", flush=True)
            else:
                print(f"\r      {str(decision):<10s} flaps={n_flaps:<4d}     ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
