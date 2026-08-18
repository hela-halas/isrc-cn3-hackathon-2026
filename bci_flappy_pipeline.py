#!/usr/bin/env python3
"""Integrated EEG-to-Flappy-Bird pipeline for the ISRC CN3 hackathon.

This single file covers the practical path from recorded EEG CSV files to a
real-time game command:

  train     - train a motor-imagery classifier from recorded CSV files
  test-udp  - send manual UDP flap commands to the game
  realtime  - read live EEG through LSL, classify windows, and emit flap commands

The defaults are intentionally simple for a hackathon setting. Tune channels,
window length, classifier, and threshold from the command line instead of
editing the code.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EPS = 1e-12
DEFAULT_EXCLUDE_COLUMNS = {"timestamp", "relative_time", "time", "marker", "label"}

joblib = None
np = None
pd = None
signal = None
LinearDiscriminantAnalysis = None
RandomForestClassifier = None
LogisticRegression = None
accuracy_score = None
balanced_accuracy_score = None
classification_report = None
f1_score = None
GroupKFold = None
StratifiedGroupKFold = None
cross_val_predict = None
make_pipeline = None
StandardScaler = None
SVC = None


def load_science_dependencies() -> None:
    """Load heavy ML dependencies only for train/realtime commands."""
    global joblib, np, pd, signal
    global LinearDiscriminantAnalysis, RandomForestClassifier, LogisticRegression
    global accuracy_score, balanced_accuracy_score, classification_report, f1_score
    global GroupKFold, StratifiedGroupKFold, cross_val_predict
    global make_pipeline, StandardScaler, SVC

    if np is not None:
        return

    try:
        import joblib as _joblib
        import numpy as _np
        import pandas as _pd
        from scipy import signal as _signal
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LinearDiscriminantAnalysis
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.metrics import (
            accuracy_score as _accuracy_score,
            balanced_accuracy_score as _balanced_accuracy_score,
            classification_report as _classification_report,
            f1_score as _f1_score,
        )
        from sklearn.model_selection import (
            GroupKFold as _GroupKFold,
            StratifiedGroupKFold as _StratifiedGroupKFold,
            cross_val_predict as _cross_val_predict,
        )
        from sklearn.pipeline import make_pipeline as _make_pipeline
        from sklearn.preprocessing import StandardScaler as _StandardScaler
        from sklearn.svm import SVC as _SVC
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing Python package: {missing}\n"
            "Install dependencies first:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    joblib = _joblib
    np = _np
    pd = _pd
    signal = _signal
    LinearDiscriminantAnalysis = _LinearDiscriminantAnalysis
    RandomForestClassifier = _RandomForestClassifier
    LogisticRegression = _LogisticRegression
    accuracy_score = _accuracy_score
    balanced_accuracy_score = _balanced_accuracy_score
    classification_report = _classification_report
    f1_score = _f1_score
    GroupKFold = _GroupKFold
    StratifiedGroupKFold = _StratifiedGroupKFold
    cross_val_predict = _cross_val_predict
    make_pipeline = _make_pipeline
    StandardScaler = _StandardScaler
    SVC = _SVC


@dataclass
class FeatureConfig:
    window_seconds: float
    step_seconds: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    notch_hz: float | None
    target_sampling_rate_hz: float | None
    channels: list[str]
    positive_classes: list[str]
    classifier: str


def parse_csv_list(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def stream_header_is_demo(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first = handle.readline().lower()
    except OSError:
        return False
    return "demo_eeg" in first


def infer_label(path: Path, positive_classes: set[str]) -> int | None:
    name = path.name.lower()
    if "rest" in name:
        return 0
    if "hand" in name:
        return 1 if "hands" in positive_classes or "hand" in positive_classes else None
    if "leg" in name or "feet" in name or "foot" in name:
        return 1 if (
            "legs" in positive_classes
            or "leg" in positive_classes
            or "feet" in positive_classes
            or "foot" in positive_classes
        ) else None
    return None


def signal_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for col in df.columns:
        if str(col).strip().lower() in DEFAULT_EXCLUDE_COLUMNS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            columns.append(col)
    return columns


def select_channels(df: pd.DataFrame, channels_arg: str | None, default_count: int = 7) -> list[str]:
    available = signal_columns(df)
    if not available:
        raise ValueError("No numeric EEG-like columns found after excluding timestamp/time columns")

    requested = parse_csv_list(channels_arg)
    if not requested:
        return available[: min(default_count, len(available))]

    selected: list[str] = []
    for item in requested:
        if item in df.columns:
            selected.append(item)
            continue
        try:
            index = int(item)
        except ValueError as exc:
            raise ValueError(f"Unknown channel {item!r}; use column names or zero-based indices") from exc
        if index < 0 or index >= len(available):
            raise ValueError(f"Channel index {index} is outside available range 0..{len(available) - 1}")
        selected.append(available[index])
    return selected


def load_csv_recording(path: Path, channels_arg: str | None) -> tuple[np.ndarray, float, list[str]]:
    df = pd.read_csv(path, comment="#")
    if df.empty:
        raise ValueError(f"{path.name} contains no samples")

    channels = select_channels(df, channels_arg)
    data = df[channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    good_rows = np.isfinite(data).all(axis=1)
    data = data[good_rows]
    if data.shape[0] < 10:
        raise ValueError(f"{path.name} has too few valid samples after dropping NaNs")

    if "relative_time" in df.columns:
        t = pd.to_numeric(df.loc[good_rows, "relative_time"], errors="coerce").to_numpy(dtype=float)
    elif "timestamp" in df.columns:
        t = pd.to_numeric(df.loc[good_rows, "timestamp"], errors="coerce").to_numpy(dtype=float)
    else:
        t = np.arange(data.shape[0], dtype=float)

    finite_t = t[np.isfinite(t)]
    if finite_t.size > 2:
        dt = np.diff(finite_t)
        dt = dt[dt > 0]
        fs = 1.0 / float(np.median(dt)) if dt.size else 125.0
    else:
        fs = 125.0

    return data, fs, channels


def resample_recording(data: np.ndarray, fs: float, target_fs: float | None) -> tuple[np.ndarray, float]:
    if target_fs is None or target_fs <= 0 or abs(fs - target_fs) < 1e-6:
        return data, fs
    target_samples = max(2, int(round(data.shape[0] * target_fs / fs)))
    return signal.resample(data, target_samples, axis=0), float(target_fs)


def apply_filters(data: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    low = max(0.01, cfg.bandpass_low_hz)
    high = min(cfg.bandpass_high_hz, fs * 0.45)
    if low >= high:
        raise ValueError(f"Invalid bandpass range {low:g}-{high:g} Hz for fs={fs:g}")

    filtered = data.astype(float, copy=True)
    sos = signal.butter(cfg.filter_order, [low, high], btype="bandpass", fs=fs, output="sos")
    if filtered.shape[0] > 3 * (2 * cfg.filter_order + 1):
        filtered = signal.sosfiltfilt(sos, filtered, axis=0)
    else:
        filtered = signal.sosfilt(sos, filtered, axis=0)

    if cfg.notch_hz and cfg.notch_hz < fs * 0.45:
        b, a = signal.iirnotch(w0=cfg.notch_hz, Q=30, fs=fs)
        if filtered.shape[0] > max(len(a), len(b)) * 3:
            filtered = signal.filtfilt(b, a, filtered, axis=0)
        else:
            filtered = signal.lfilter(b, a, filtered, axis=0)
    return filtered


def band_power(values: np.ndarray, fs: float, low: float, high: float) -> float:
    nperseg = min(len(values), max(32, int(round(fs))))
    freqs, power = signal.welch(values, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return EPS
    return float(np.trapz(power[mask], freqs[mask]) + EPS)


def extract_features(window: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    filtered = apply_filters(window, fs, cfg)
    features: list[float] = []
    for channel_index in range(filtered.shape[1]):
        values = filtered[:, channel_index]
        features.append(float(np.log(np.var(values) + EPS)))
        features.append(float(np.log(band_power(values, fs, 8.0, 13.0))))
        features.append(float(np.log(band_power(values, fs, 13.0, 30.0))))
    return np.asarray(features, dtype=float)


def window_features(data: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    window_samples = max(8, int(round(cfg.window_seconds * fs)))
    step_samples = max(1, int(round(cfg.step_seconds * fs)))
    rows = []
    for start in range(0, data.shape[0] - window_samples + 1, step_samples):
        rows.append(extract_features(data[start : start + window_samples], fs, cfg))
    if not rows:
        raise ValueError(
            f"Recording has {data.shape[0]} samples, shorter than requested window of {window_samples}"
        )
    return np.vstack(rows)


def make_classifier(name: str):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError(f"Unknown classifier {name!r}")


def build_training_set(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, FeatureConfig, list[dict]]:
    positive_classes = set(parse_csv_list(args.positive_classes) or ["hands", "legs", "feet"])
    data_root = Path(args.data_dir)
    csv_paths = sorted(data_root.rglob("*.csv") if args.recursive else data_root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    first_channels: list[str] | None = None
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    manifest: list[dict] = []

    placeholder_cfg = FeatureConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        filter_order=args.filter_order,
        notch_hz=args.notch_hz,
        target_sampling_rate_hz=args.target_sampling_rate,
        channels=[],
        positive_classes=sorted(positive_classes),
        classifier=args.classifier,
    )

    for path in csv_paths:
        try:
            display_path = str(path.relative_to(data_root))
        except ValueError:
            display_path = str(path)

        if stream_header_is_demo(path) and not args.include_demo:
            manifest.append({"file": display_path, "status": "skipped", "reason": "demo_eeg stream"})
            continue
        label = infer_label(path, positive_classes)
        if label is None:
            manifest.append({"file": display_path, "status": "skipped", "reason": "no selected label"})
            continue

        try:
            data, fs, channels = load_csv_recording(path, args.channels)
            original_fs = fs
            data, fs = resample_recording(data, fs, args.target_sampling_rate)
        except Exception as exc:
            manifest.append(
                {
                    "file": display_path,
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if first_channels is None:
            first_channels = channels
            placeholder_cfg.channels = channels
        elif channels != first_channels:
            manifest.append(
                {
                    "file": display_path,
                    "status": "skipped",
                    "reason": f"selected channels {channels} differ from {first_channels}",
                }
            )
            continue

        try:
            feats = window_features(data, fs, placeholder_cfg)
        except Exception as exc:
            manifest.append(
                {
                    "file": display_path,
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        X_parts.append(feats)
        y_parts.append(np.full(feats.shape[0], label, dtype=int))
        group_parts.append(np.full(feats.shape[0], display_path, dtype=object))
        manifest.append(
            {
                "file": display_path,
                "status": "used",
                "label": int(label),
                "original_fs_hz": round(float(original_fs), 3),
                "fs_hz": round(float(fs), 3),
                "samples": int(data.shape[0]),
                "windows": int(feats.shape[0]),
                "channels": channels,
            }
        )

    if not X_parts:
        raise ValueError("No usable training files after filtering labels/demo recordings")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    if len(np.unique(y)) != 2:
        raise ValueError(f"Training needs two classes; got labels {sorted(np.unique(y).tolist())}")
    return X, y, groups, placeholder_cfg, manifest


def cross_validate(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return {"status": "skipped", "reason": "need at least two recording groups"}

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=splitter)
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=splitter)

    return {
        "status": "ok",
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred)), 4),
        "f1_flap": round(float(f1_score(y, y_pred, pos_label=1)), 4),
        "report": classification_report(y, y_pred, target_names=["idle/rest", "flap/imagery"], zero_division=0),
    }


def train_command(args) -> int:
    load_science_dependencies()
    X, y, groups, cfg, manifest = build_training_set(args)
    model = make_classifier(args.classifier)
    cv = cross_validate(model, X, y, groups)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_config": asdict(cfg),
        "labels": {"0": "idle/rest", "1": "flap/imagery"},
        "training_files": manifest,
        "cv": cv,
    }
    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)

    print(f"Saved model: {output}")
    print(f"Training windows: {len(y)} ({int(np.sum(y == 0))} idle/rest, {int(np.sum(y == 1))} flap/imagery)")
    print(f"Channels: {', '.join(cfg.channels)}")
    print(f"Filter: Butterworth bandpass {cfg.bandpass_low_hz:g}-{cfg.bandpass_high_hz:g} Hz, order {cfg.filter_order}")
    if cfg.notch_hz:
        print(f"Notch: {cfg.notch_hz:g} Hz")
    if cv["status"] == "ok":
        print(
            "Cross-validation: "
            f"accuracy={cv['accuracy']}, balanced_accuracy={cv['balanced_accuracy']}, f1_flap={cv['f1_flap']}"
        )
        print(cv["report"])
    else:
        print(f"Cross-validation skipped: {cv['reason']}")
    print("File manifest:")
    for item in manifest:
        print("  " + json.dumps(item, ensure_ascii=False))
    return 0


def udp_sender(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(message: str) -> None:
        sock.sendto(message.encode("utf-8"), (host, port))

    return sock, send


def test_udp_command(args) -> int:
    sock, send = udp_sender(args.host, args.port)
    try:
        print(f"Sending UDP flap messages to {args.host}:{args.port}; message={args.message!r}")
        if args.count is not None:
            for index in range(args.count):
                send(args.message)
                print(f"sent {index + 1}/{args.count}")
                if index + 1 < args.count:
                    time.sleep(args.interval)
            return 0

        print("Press Enter to flap. Type q then Enter to quit.")
        while True:
            try:
                text = input("> ")
            except EOFError:
                print("No interactive input available; send one message and exit.")
                send(args.message)
                break
            if text.strip().lower() == "q":
                break
            send(args.message)
            print("sent")
    finally:
        sock.close()
    return 0


def resolve_eeg_stream(name: str | None, timeout: float):
    from pylsl import StreamInlet, resolve_streams

    deadline = time.monotonic() + timeout
    target = f"name={name!r}" if name else "type='EEG'"
    print(f"Waiting for LSL EEG stream with {target} ...")
    while time.monotonic() < deadline:
        streams = resolve_streams(wait_time=1.0)
        matches = [s for s in streams if (s.name() == name if name else s.type().lower() == "eeg")]
        if matches:
            stream = matches[0]
            print(
                f"Connected to {stream.name()!r}: "
                f"{stream.channel_count()} channels at {stream.nominal_srate():g} Hz"
            )
            return StreamInlet(stream, max_buflen=30, processing_flags=0)
    raise TimeoutError(f"No LSL EEG stream appeared within {timeout:g} seconds")


def make_lsl_output(stream_name: str):
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(stream_name, "Markers", 1, 0, "float32", "bci-flappy-output")
    outlet = StreamOutlet(info)

    def send(value: float) -> None:
        outlet.push_sample([float(value)])

    return send


def prediction_score(model, features: np.ndarray) -> float:
    row = features.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(row)[0, 1])
    if hasattr(model, "decision_function"):
        decision = float(model.decision_function(row)[0])
        return 1.0 / (1.0 + np.exp(-decision))
    return float(model.predict(row)[0])


def realtime_command(args) -> int:
    load_science_dependencies()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_config = dict(bundle["feature_config"])
    feature_config.setdefault("target_sampling_rate_hz", None)
    cfg = FeatureConfig(**feature_config)

    inlet = resolve_eeg_stream(args.eeg_stream, args.timeout)
    info = inlet.info()
    fs = args.sampling_rate or info.nominal_srate()
    if fs <= 0:
        raise ValueError("The LSL stream has no nominal sampling rate; pass --sampling-rate")

    if args.output == "udp":
        sock, send_udp = udp_sender(args.host, args.port)

        def emit_flap() -> None:
            send_udp(args.message)

        cleanup = sock.close
    elif args.output == "lsl":
        send_lsl = make_lsl_output(args.lsl_output_stream)

        def emit_flap() -> None:
            send_lsl(1.0)
            time.sleep(0.05)
            send_lsl(0.0)

        cleanup = lambda: None
    else:

        def emit_flap() -> None:
            print("FLAP")

        cleanup = lambda: None

    channel_indices = [int(item) for item in parse_csv_list(args.live_channel_indices) or []]
    if not channel_indices:
        channel_indices = list(range(len(cfg.channels)))
    if max(channel_indices) >= info.channel_count():
        raise ValueError(
            f"Requested live channel index {max(channel_indices)} but stream has only {info.channel_count()} channels"
        )

    window_samples = int(round(cfg.window_seconds * fs))
    step_seconds = args.step_seconds or cfg.step_seconds
    buffer: deque[list[float]] = deque(maxlen=window_samples * 3)
    next_prediction = time.monotonic()
    last_flap = 0.0
    print(
        f"Realtime mode: window={cfg.window_seconds:g}s, step={step_seconds:g}s, "
        f"threshold={args.threshold:g}, cooldown={args.cooldown:g}s, output={args.output}"
    )

    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.05, max_samples=max(1, int(fs * 0.2)))
            if chunk:
                buffer.extend(chunk)

            now = time.monotonic()
            if len(buffer) < window_samples or now < next_prediction:
                continue
            next_prediction = now + step_seconds

            data = np.asarray(list(buffer)[-window_samples:], dtype=float)
            data = data[:, channel_indices]
            feature_data, feature_fs = resample_recording(data, fs, cfg.target_sampling_rate_hz)
            score = prediction_score(model, extract_features(feature_data, feature_fs, cfg))
            print(f"score={score:.3f}")
            if score >= args.threshold and now - last_flap >= args.cooldown:
                emit_flap()
                last_flap = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cleanup()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a classifier from recorded EEG CSV files")
    train.add_argument("--data-dir", required=True, help="folder containing Hands/LegImagery/Rest CSV files")
    train.add_argument("--model-out", default="model.joblib", help="where to save the trained model bundle")
    train.add_argument("--positive-classes", default="hands,legs,feet", help="classes mapped to flap: hands, legs, feet, or a comma-separated subset")
    train.add_argument("--channels", help="CSV channel names or zero-based signal-column indices, e.g. Ch1,Ch2 or 0,1,2")
    train.add_argument("--recursive", action="store_true", help="search for CSV files below data-dir recursively")
    train.add_argument("--include-demo", action="store_true", help="include demo_eeg CSV files such as Hands4.csv")
    train.add_argument("--window-seconds", type=float, default=1.5)
    train.add_argument("--step-seconds", type=float, default=0.25)
    train.add_argument("--bandpass-low", type=float, default=8.0)
    train.add_argument("--bandpass-high", type=float, default=30.0)
    train.add_argument("--filter-order", type=int, default=4)
    train.add_argument("--notch-hz", type=float, default=None, help="optional power-line notch, e.g. 50 or 60")
    train.add_argument("--target-sampling-rate", type=float, default=None, help="optional resampling rate before feature extraction, e.g. 500")
    train.add_argument("--classifier", choices=["logistic", "lda", "svm", "rf"], default="logistic")
    train.set_defaults(func=train_command)

    test_udp = sub.add_parser("test-udp", help="manually send UDP flap commands to the game")
    test_udp.add_argument("--host", default="127.0.0.1")
    test_udp.add_argument("--port", type=int, default=5005)
    test_udp.add_argument("--message", default="1")
    test_udp.add_argument("--count", type=int, default=None, help="send this many messages without interactive input")
    test_udp.add_argument("--interval", type=float, default=0.2, help="seconds between messages when --count is used")
    test_udp.set_defaults(func=test_udp_command)

    realtime = sub.add_parser("realtime", help="read LSL EEG, classify, and send flap commands")
    realtime.add_argument("--model", required=True, help="model bundle produced by train")
    realtime.add_argument("--eeg-stream", help="exact LSL EEG stream name; omit to use first stream of type EEG")
    realtime.add_argument("--timeout", type=float, default=30.0)
    realtime.add_argument("--sampling-rate", type=float, help="override stream nominal sampling rate")
    realtime.add_argument("--live-channel-indices", help="zero-based live LSL channel indices; default matches trained count")
    realtime.add_argument("--threshold", type=float, default=0.65)
    realtime.add_argument("--cooldown", type=float, default=0.45)
    realtime.add_argument("--step-seconds", type=float, help="override prediction step")
    realtime.add_argument("--output", choices=["udp", "lsl", "print"], default="udp")
    realtime.add_argument("--host", default="127.0.0.1")
    realtime.add_argument("--port", type=int, default=5005)
    realtime.add_argument("--message", default="1")
    realtime.add_argument("--lsl-output-stream", default="MotorImagery")
    realtime.set_defaults(func=realtime_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
