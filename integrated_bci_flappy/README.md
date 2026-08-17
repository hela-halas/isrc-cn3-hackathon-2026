# Integrated BCI Flappy Pipeline

This folder is a clean integration layer for the ISRC CN3 hackathon files. It does not modify the original starter code or notebooks.

Goal:

```text
FlexEEG LSL stream or recorded CSV
  -> preprocessing/filtering
  -> sliding-window features
  -> machine-learning classifier
  -> flap decision
  -> UDP or LSL command to the Flappy Bird game
```

## Files

- `bci_flappy_pipeline.py`: one-file pipeline for training, UDP testing, and real-time control.
- `requirements.txt`: Python packages needed by the pipeline.

## What Is Implemented

### Input

Offline training reads CSV files such as:

- `Hands1.csv` ... `Hands5.csv`
- `LegImagery1.csv` ... `LegImagery5.csv`
- `Rest1.csv`, `Rest1 (1).csv` ...

The CSV reader skips comment lines starting with `#`, then uses numeric EEG columns.

Default channel behavior:

- ignores `timestamp` and `relative_time`;
- uses the first 7 remaining numeric signal columns;
- for FlexEEG files this usually means `Ch1` to `Ch7`;
- `Ch9` is not used by default, because the starter MATLAB comment says FlexEEG may have `8 EEG + 1 aux`.

If the real headset montage says different channels should be used, pass them explicitly:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --channels Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7
```

or by zero-based signal-column index:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --channels 0,1,2,3,4,5,6
```

### Filter

The pipeline uses:

- Butterworth band-pass filter;
- default range: `8-30 Hz`;
- default order: `4`;
- optional notch filter: `--notch-hz 50` or `--notch-hz 60`.

Why this default:

- motor imagery is often evaluated through mu rhythm and beta rhythm changes;
- `8-13 Hz` and `13-30 Hz` are used downstream as feature bands;
- this is a hackathon baseline, not a final neuroscience-grade preprocessing pipeline.

### Feature Extraction

Each sliding window is converted into features per channel:

- log variance after band-pass filtering;
- log mu-band power: `8-13 Hz`;
- log beta-band power: `13-30 Hz`.

With 7 channels this gives:

```text
7 channels * 3 features = 21 features per window
```

Default windowing:

- window length: `1.5 s`;
- step size: `0.25 s`.

These can be changed:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --window-seconds 2.0 --step-seconds 0.25
```

### Machine Learning

Implemented classifiers:

- `logistic`: standard scaler + balanced logistic regression, default;
- `lda`: standard scaler + linear discriminant analysis;
- `svm`: standard scaler + RBF SVM with probabilities;
- `rf`: balanced random forest.

Choose one with:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --classifier lda
```

Default label mapping:

- `Rest*.csv` -> `0`, idle/no flap;
- `Hands*.csv` -> `1`, flap;
- `LegImagery*.csv` -> `1`, flap.

If you only want hands as the positive class:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --positive-classes hands
```

If you only want legs:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --positive-classes legs
```

By default `demo_eeg` recordings are skipped. In the current folder, `Hands4.csv` appears to be a demo stream rather than the same FlexEEG source as the other files.

## Install

From this folder:

```powershell
python -m pip install -r requirements.txt
```

If the hackathon machine uses Anaconda, run this inside the same Anaconda environment used for LSL.

## Train a Baseline Model

From this integration folder:

```powershell
python bci_flappy_pipeline.py train --data-dir "..\EEG Hackathon data bios and one" --model-out model.joblib
```

The script prints:

- which files were used or skipped;
- inferred sampling rate per file;
- selected channels;
- number of training windows;
- cross-validation accuracy, balanced accuracy, and F1 for flap.

The output model bundle is:

```text
model.joblib
```

## Test the Game Interface First

Before using EEG, confirm the game responds to UDP.

Start the Flappy Bird / Flappy Brain game, then run:

```powershell
python bci_flappy_pipeline.py test-udp --host 127.0.0.1 --port 5005 --message 1
```

Press Enter. If the bird flaps, the game interface is working.

If the game runs on another PC, replace `127.0.0.1` with that PC's IP address.

## Real-Time EEG Control

Start the EEG stream first, then run:

```powershell
python bci_flappy_pipeline.py realtime --model model.joblib --output udp --host 127.0.0.1 --port 5005
```

Useful realtime options:

```powershell
python bci_flappy_pipeline.py realtime --model model.joblib --threshold 0.7 --cooldown 0.5
```

- `--threshold`: classifier score needed to trigger a flap.
- `--cooldown`: minimum seconds between two flap commands.
- `--live-channel-indices`: zero-based channel indices in the live LSL stream.

Example if the live stream should use channels 0-6:

```powershell
python bci_flappy_pipeline.py realtime --model model.joblib --live-channel-indices 0,1,2,3,4,5,6
```

## Alternative LSL Output to the Game

If the game is configured to receive LSL markers instead of UDP:

```powershell
python bci_flappy_pipeline.py realtime --model model.joblib --output lsl --lsl-output-stream MotorImagery
```

The LSL output sends:

- `1.0` for flap;
- then `0.0` shortly after, returning to idle.

This matches the original `lsl_dummy_input.py` behavior.

## How to Import Into the Game

Usually no source-code import is needed.

Expected setup:

1. Start the provided Flappy Bird / Flappy Brain game executable.
2. Confirm whether the game listens through UDP or LSL.
3. If UDP, use host `127.0.0.1`, port `5005`, message `"1"` unless the game settings say otherwise.
4. Run `test-udp` to verify that a manual command makes the bird flap.
5. Train `model.joblib`.
6. Run `realtime` so EEG predictions emit the same flap command.

If the provided game is a Unity project rather than an executable, look for scripts named like:

- `UDPInputReceiver.cs`
- `LSLInputReceiver.cs`
- `LSL4UnityBirdInlet.cs`

The values to match are usually:

- UDP port;
- flap message string;
- LSL stream name;
- LSL stream type;
- flap value.

## Current Integration Assumptions

- The game accepts a single flap command, not continuous vertical control.
- `Rest` is idle/no flap.
- `Hands` and/or `LegImagery` are possible motor-imagery flap classes.
- The first 7 numeric EEG columns are the intended competition channels unless changed with `--channels`.
- `Hands4.csv` is skipped by default because it is from `demo_eeg`, not the same FlexEEG stream.

These assumptions are intentionally exposed as command-line options, because the team still needs to confirm the actual electrode mapping and game receiver settings.

## Practical Debug Order

1. `test-udp`: prove the game can flap from Python.
2. `train`: prove the offline data can produce a model.
3. `realtime --output print`: prove live EEG can produce classifier scores without touching the game.
4. `realtime --output udp`: connect the classifier to the game.
5. Tune `--threshold`, `--cooldown`, and `--window-seconds`.

## Known Limitations

- This is a baseline integration, not a final BCI algorithm.
- There is no calibration UI.
- Offline labels are inferred from filenames.
- Without event markers, each whole CSV is treated as one condition.
- Window-level cross-validation can still be optimistic when recordings are short; the script uses recording groups to reduce leakage.
- If real-time channel order differs from CSV channel order, you must set `--live-channel-indices`.
