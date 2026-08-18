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
from pylsl import StreamInfo, StreamOutlet

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    
    
# ------------------------- starting point of  our codes ------------------
try:
    import scipy.signal as signal
except ImportError:
    signal = None
    print("Warning: scipy is missing")
    
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
except ImportError:
    LinearDiscriminantAnalysis = None
    print("Warning: scikit-learn is missing")


def apply_bandpass_filter(data: np.ndarray,
                          fs: float,
                          lowcut: float = 1.0,
                          highcut: float = 50.0) -> np.ndarray:
    """Step 1: Preprocess. Apply the Butterworth bandpass filter to remove noise."""
    if signal is None:
        return data
    
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(4, [low, high], btype = "band")
    
    # zero-phase filter along sample dimension (axis=1) using filtfilt (forward-backward filter)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data
    
    
def extract_features(data: np.ndarray, fs: float) -> np.ndarray:
     """Step 2: Extract features. Compress raw data into bands (Theta, Alpha, Beta)."""
    if signal is None: 
        return np.zeros(data.shape[0] * 3)
    
    nperseg    = min(int(fs), data.shape[1]) # number of points per segment (~1 second of data)
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    bands      = {"theta": (4, 8), "alpha": (8, 12), "beta": (13, 30)}
    features   = []
    
    for ch_idx in range(data.shape[0]):
        for band_name, (low, high) in bands.items():
            band_idx = np.logical_and(freqs >= low, freqs <= high)
            power = np.mean(psd[ch_idx, band_idx])
            features.append(power)
            
    return np.asarray(features, dtype = float)

def load_eeg_csv(filepath: str) -> np.ndarray:
    """Load EEG data from a CSV file. Returns (n_channels, n_samples)."""
    data = np.genfromtxt(filepath, delimiter=",")
    if np.isnan(data[0]).any():
        data = np.genfromtxt(filepath, delimiter=",", skip_header=1)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] > data.shape[1]:
        data = data.T

    return data
    

def get_trained_classifier(fs: float, window_sec: float, channels: np.ndarray, n_features: int):
    """Step 3: Train/Load Classifier (Left Hand, Right Hand, Leg, Rest) from calibration CSVs."""
    if LinearDiscriminantAnalysis is None: 
        return None
    
    calibration_files = {        
        "Hand": "Hand1Training.csv",
        "Leg": "FeetTraining.csv",
        "Rest": "RestTraining.csv"
    }
    
    
    x_train = []
    y_train = []
    
    window_samples = int(window_sec * fs)
    step_samples   = int(window_samples * 0.5)
    
    
    for label, filename in calibration_files.items():
        print(f" Learning {label.upper()} from {filename}...")
        
        raw_data = load_eeg_csv(filename)
        raw_data = raw_data[channels, :]
        clean_data = apply_bandpass_dilter(raw_data, fs, lowcut = 1.0, highcut = 50.0)
        
        total_samples = clean_data.shape[1]
        for start_idx in range(0, total_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            epoch = clean_data[:, start_idx:end_idx]
            
            features = extract_features(epoch, fs)
            x_train.append(features)
            y_train.append(label)
            
        
    
    if len(x_train) == 0:
        print("  [Error] No training data found, check for CSV files")
        return None
    
    print(f" Training LDA model on {len(x_train)} real brainwave epochs...")
    
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(x_train, y_train)
    return classifier

# ------------- until HERE it is only bandpass, feature extraction, and classifier definitions ---------
# ------------- other added part is in grab(), after this point till the same --- thing its Toby's code ---------

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


# --------------- grab() starts here actual implementation is in here ---------------

def grab(args):
    """Connect to an EEG stream and pull one window of samples, printing its shape."""
    
    inlet = resolve_eeg(args.eeg_stream, args.timeout)
    info  = inlet.info()
    fs = args.sampling_rate or info.nominal_srate()
    if fs <= 0:
        raise ValueError("The stream has no nominal rate; supply --sampling-rate")
    channels = parse_channels(args.channels, info.channel_count())
    
    n_channels  = len(channels)
    windows_sec = args.seconds
    
    # train classifier 
    n_features = n_channels * 3 # theta, alpha, beta per channel
    classifier = get_trained_classifier(fs, window_sec, channels, n_features)
    
    STREAM_NAME = "MotorImagery"
    STREAM_TYPE = "Markers"
    CHANNEL_COUNT = 3
    SAMPLE_RATE = 0 # irregular rate - we push samples manually, not on a clock
    info = StreamInfo(STREAM_NAME, STREAM_TYPE, CHANNEL_COUNT, SAMPLE_RATE, "float32", "eeg-classifier-001")
    outlet = StreamOutlet(info)
    print(f"Publishing LSL stream '{STREAM_NAME}' (type '{STREAM_TYPE}')")
    
    print(f"\nStarting live classification stream ({window_sec}s windows)...")
    print("=" * 60)
    print(f"{'Status / Window':<18} | {'Feature Count':<14} | {'Prediction':<15}")
    print("-" * 60)
    
    try:
        while True:
            # pull live windows from LSL
            t_start = time.time()
            epoch = collect_window(inlet, window_sec, fs, channels)
            
            # Step 1: preprocess
            clean_epoch = apply_bandpass_filter(epoch, fs, lowcut = 1.0, highcut=50.0)
            
            # Step 2: extract features (theta, alpha, beta)
            features = extract_features(clean_epoch, fs)
            
            # Step 3: classify the live window
            prediction = classifier.predict([features])[0] if classifier else "N/A"
            
            # this is a try to assign the hand and leg and rest
            pred_lower = prediction.lower()
            if "hand" in pred_lower:
                unity_value = 1.0 # fly
                signal_label = "FLY (1.0)"
            else:
                unity_value = 0.0 # stop
                signal_label = "STOP (0.0)"
                
            outlet.push_sample([unit_value])
            
            elapsed = time.time() - t_start

            print(f"Live Window      | {len(features):<14} | {prediction.upper()}")
            
    except KeyboardInterrupt:
        print("\nStopping live classification stream.")
    
# ------------------------ our part is finished ----------------
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
    grab_cmd.add_argument("--seconds", type=float, default=1.0)
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