# Classification pipeline

Working code for the motor-imagery classifier that drives the bird, plus an
assessment of what the supplied recordings can and cannot support.

## TL;DR

**The supplied CSVs cannot train a usable classifier.** Every `Rest` recording
came off a different machine than every `Hands`/`LegImagery` recording, so the
class label and the recording session are the same variable. The 95.7%
"rest vs imagery" score is a laptop detector. The one comparison that shares a
session — hands vs legs — runs at **58.0% against a 56.0% baseline**, i.e.
nothing.

Plan for the day: record your own calibration data with `record_calibration.py`,
train on that, run `realtime_classifier.py`. The pipeline is ready and tested;
only the data is missing.

## Layout

| File | What it does |
| --- | --- |
| `bci/data.py` | Loads the FlexEEG CSV exports, resamples to a uniform grid, drops dead/artifact channels |
| `bci/features.py` | Notch + band-pass + common-average reference, epoching, log band power |
| `bci/model.py` | Trains, saves and loads the live classifier |
| `train_offline.py` | Compares classifiers under two validation schemes; `--save` writes `model.pkl` |
| `diagnose_confound.py` | Demonstrates the session confound in the supplied data |
| `record_calibration.py` | Cued recorder — **run this first on the day** |
| `realtime_classifier.py` | LSL in → flap out over LSL or UDP |

## Which classifier

**Shrinkage LDA**, matching the recommendation in `pipeline-layout.ipynb`.
`train_offline.py` scores all four candidates from that notebook, so this is
measured rather than assumed:

| classifier | random 5-fold | leave-one-recording-out |
| --- | --- | --- |
| LDA (shrinkage) | 65.3% | **58.0%** |
| LogisticRegression | 65.3% | 56.4% |
| SVM (RBF) | 68.0% | 46.5% |
| RandomForest | 61.3% | 47.8% |
| *chance* | *56.0%* | *56.0%* |

LDA wins for the reason `pipeline-layout.ipynb` gives: with ~30 features and
~100 calibration windows, anything with more capacity overfits. Note that the
higher-capacity models (SVM, RF) score **below chance** on held-out recordings
while looking fine under random k-fold — a textbook overfitting signature.

### Why two columns

Random k-fold is what `class_exampl.ipynb` does via `train_test_split`. Windows
overlap by 50%, so near-identical windows land on both sides of the split and
the score is inflated by ~7-20 points. Leave-one-recording-out is the number
that predicts live performance. **Always read the right-hand column.**

## Problems found in the supplied data

1. **Session confound (fatal).** `Rest*` is from host `Umbara`; `Hands*` and
   `LegImagery*` are from `DESKTOP-0N5AHDD`. Predicting the *host* from the EEG
   scores 95.7% — identical to the rest-vs-imagery score, because they are the
   same discrimination. Run `diagnose_confound.py`.
2. **Sampling rate is ~125 Hz, not 250 Hz.** Samples arrive in bursts, so the
   median inter-sample gap reads ~700 Hz while the effective rate is 125.6 Hz.
   `bandpass.ipynb` assumes `fps = 250`, which puts every cutoff off by 2x.
   `bci/data.py` resamples onto a uniform grid before filtering.
3. **Dead and artifact channels.** Ch9 is identically zero. Ch3 carries an
   artifact ~40x the amplitude of a clean channel; Ch2 does the same in several
   Rest files. All three are dropped by default.
4. **`Rest1.csv` is two recordings concatenated**, with a second `# Stream:`
   header partway through — and its first half duplicates `Rest1 (1).csv`.
   The loader splits on header blocks so the two are never filtered as one.
5. **`Hands4.csv` is a different device**: 250 Hz, named channels (Fp1..O2),
   amplitudes ~20x smaller. Excluded by default.
6. **Tiny.** 141 seconds of usable EEG, 116 windows. Typical motor-imagery
   training sets are 20-40 minutes per subject.

## On the day

```bash
pip install numpy scipy scikit-learn pylsl

# 1. Record calibration data — interleaved classes, one subject, one sitting
python3 record_calibration.py --subject alice --classes rest,hands --trials 40

# 2. Train and check the leave-one-recording-out score
python3 train_offline.py --data-dir calibration/alice --save

# 3. Drive the game
python3 realtime_classifier.py --output lsl --flap-class hands
```

Sanity-check step 3 with `--dry-run` first: it prints decisions without sending
anything, so you can confirm the classifier responds to the subject before
involving Unity.

### Recording rules that matter more than the code

- **Interleave the classes.** Recording all the rest trials then all the hands
  trials rebuilds exactly the confound that ruins the supplied data — drift,
  impedance and fatigue all track time.
- **One subject, one sitting, headset stays on.** Removing and replacing it
  changes the montage and invalidates everything recorded before.
- **Aim for 40+ trials per class.** 20 is workable; 5 is not.
- **Prefer rest vs one imagery class.** Hands vs legs is a genuinely hard
  discrimination, and the game only needs flap / don't-flap.

## Output format

Matches the dummy senders exactly, so Unity cannot tell the difference:

- **LSL** — 1-channel `MotorImagery` stream, type `Markers`, pushes `1.0` then
  `0.0` (as `lsl_dummy_input.py`).
- **UDP** — the string `"1"` to `127.0.0.1:5005` (as `udp_dummy_input.py`).

Two knobs govern the feel. `--vote N` majority-votes over the last N decisions:
raw per-window output flips label ~17 times in a 9-second recording, and 5-vote
smoothing cuts that to 6, at the cost of ~600 ms of latency. `--refractory S`
sets the minimum gap between flaps so one sustained thought is one flap.
