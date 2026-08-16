# Motor-imagery classifier

Band-pass → CSP → log-variance → LDA, plus a real-time loop that drives the
game. Built on top of the recordings in `EEG Hackathon data bios and one/`.

```bash
pip install numpy scipy scikit-learn      # pylsl too, for live use
python train_classifier.py audit          # what is actually in the data
python train_classifier.py train          # fit + cross-validate, writes model.pkl
python realtime_bci.py simulate --csv "EEG Hackathon data bios and one/Hands2.csv"
python realtime_bci.py run --output udp   # live, with the headset
python test_bci.py                        # 22 tests
```

## What the data turned out to be

`train_classifier.py audit` reports this; it is worth knowing before trusting
any accuracy number.

| Finding | Consequence |
|---|---|
| Real recordings run at **~125.5 Hz**, not the 250 Hz assumed in `bandpass.ipynb` | Every cutoff in that notebook lands at half its intended frequency |
| **`Hands4.csv` is synthetic** — stream `demo_eeg from localhost`, 8 channels named Fp1…O2, 250 Hz, clock from 0, all channels ~15 µV | Excluded automatically (wrong channel layout) |
| **`Rest1.csv` and `Rest1 (1).csv` are the same recording** | Excluded as a duplicate; keeping both would leak between CV folds |
| **`Ch9` is identically zero**, `Ch3` runs 10–40× the other channels | Both dropped — CSP maximises variance ratios, so a railing electrode dominates every component |
| **`rest` was recorded in a different session** (LSL clock ~238 600 vs ~95 300–96 240 for the imagery classes) | `rest` is excluded by default: a classifier separating it may be reading session, not intent |

That leaves **9 usable trials** — 4 `hands`, 5 `legimagery` — recorded
interleaved in one session. Small, but honest.

## Results

Leave-one-trial-out over those 9 recordings, 2 s epochs, 8–30 Hz:

```
  epoch accuracy    71.3%   (chance 56.6%)
  trial accuracy   100.0%   (majority vote within each held-out recording)
  p-value           0.000
```

`python train_classifier.py sweep` compares bands, and confirms the band choice
empirically rather than by assertion:

| band | 2 s epochs | 3 s epochs |
|---|---|---|
| 4–8 Hz (theta) | 46.5% | 55.9% |
| 8–13 Hz (mu) | 67.4% | 60.4% |
| 13–30 Hz (beta) | 67.4% | 68.5% |
| **8–30 Hz (mu+beta)** | **71.3%** | **82.0%** |
| 1–50 Hz (the notebook's band) | 59.7% | 59.5% |

Mu+beta beats the notebook's 1–50 Hz by ~12–22 points, and theta sits at chance
exactly as it should. 3 s epochs score higher but cost 3 s of latency; 2 s is
the playable compromise.

## Design decisions worth keeping

**Causal filtering everywhere, including training.** `filter_offline` uses
`sosfilt`, not `sosfiltfilt`. The live loop cannot see the future, so training
on zero-phase data fits the classifier to features that never occur at run
time. Phase distortion does not matter here because log-variance discards
phase.

**Streaming and offline filtering are bit-identical.** `StreamingBandpass`
carries `zi` across chunks and both paths initialise state the same way;
`test_streaming_matches_offline_exactly` asserts agreement to 1e-12 across
ragged chunk sizes. Without carried state every window gets a startup
transient — large, label-independent, and very learnable.

**Cross-validation holds out whole recordings.** Epochs overlap by 75%, so
splitting them randomly puts near-identical epochs in train and test and
reports high-nineties accuracy for a classifier that is at chance on a new
trial. `test_cross_validation_never_shares_a_trial` pins this down.

**The first second of every trial is discarded** (`discard_seconds`), because
the causal filter is still settling and that transient is identical across
classes.

**Smoothing goes on the output, not the EEG.** `Smoother` is an EMA over the
classifier's probability. Lower `--alpha` gives a calmer bird and more lag.

## Before the hackathon

The provided recordings are someone else's brain, and 9 trials is very few.
Record calibration data with your own headset on the day and retrain — the CV
number above will not transfer. `--target-class` picks which class flaps, and
the output protocol already matches `lsl_dummy_input.py` and
`udp_dummy_input.py`, so Unity needs no changes.
