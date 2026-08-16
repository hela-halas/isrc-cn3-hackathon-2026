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
| `bci/riemann.py` | Covariance → Riemannian tangent space features, + SVM-RBF pipeline |
| `bci/model.py` | Trains, saves and loads the live classifier |
| `train_offline.py` | Compares classifiers under two validation schemes; `--save` writes `model.pkl` |
| `compare_riemann.py` | Tangent-space dimensionality, and whether reduction helps |
| `diagnose_confound.py` | Demonstrates the session confound in the supplied data |
| `record_calibration.py` | Cued recorder — **run this first on the day** |
| `realtime_classifier.py` | LSL in → flap out over LSL or UDP |

## Which classifier (band-power baseline)

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

## Riemannian tangent space + SVM-RBF

`bci/riemann.py`, compared by `compare_riemann.py`.

### What goes into the SVM

Not raw EEG, and not band power — **tangent-space vectors derived from spatial
covariance matrices**:

```
epoch                    6 channels x 250 samples  = 1500 numbers
  -> covariance          6 x 6 SPD matrix
  -> tangent space       21 features   ( = n(n+1)/2 )
  -> StandardScaler -> SVC(kernel="rbf")
```

The covariance matrix *is* the feature. Motor imagery modulates mu/beta power
in sensorimotor cortex, which changes both the variance of individual channels
and the correlation between them; covariance captures both, and unlike
per-channel band power it keeps the cross-channel terms where C3/C4
lateralisation lives.

Covariance matrices are symmetric positive definite, so they lie on a curved
manifold. Feeding them to an SVM directly is a category error — the
straight-line distance between two SPD matrices isn't the meaningful distance.
The tangent space projection flattens the manifold locally around the
Riemannian mean of the *training* covariances, after which Euclidean
classifiers apply. **That mean is a fitted parameter**: fit it on the training
fold only, or you leak across the split.

### Does it need dimensionality reduction?

**No.** The covariance → tangent space step already *is* the dimensionality
reduction: 1500 numbers down to 21. Reducing 21 further is not the problem you
have. Measured on hands vs legs:

| pipeline | random 5-fold | leave-one-recording-out |
| --- | --- | --- |
| Riemann TS + SVM-RBF (21f, no reduction) | 70.7% | 42.8% |
| + PCA(10) | 73.3% | 46.4% |
| + PCA(5) | 70.7% | 49.4% |
| + SelectKBest(10) | 70.7% | 43.1% |
| Riemann TS + LDA (no SVM) | 70.7% | 45.4% |
| **Filter bank** TS + SVM-RBF (84f) | **84.0%** | **41.1%** |
| Filter bank + PCA(15) | 82.7% | 40.3% |
| *chance* | *56.0%* | *56.0%* |

PCA moves the honest column by a few points and everything stays below chance.
It costs 21 → 12 components to keep 90% of the variance, so there isn't much
redundancy to squeeze out.

Where reduction *does* matter is the **filter bank**: 4 sub-bands × 21 = 84
features against 116 windows is 1.4 samples per feature, and it shows the
widest 5-fold/LORO gap in the whole table — 84.0% down to 41.1%. That 43-point
spread is what overfitting looks like. If you use a filter bank, you need the
reduction; the better answer is to not need the filter bank.

The number to watch is **samples per feature**: 5.5 single-band, 1.4 filter
bank. Both are low, and the fix is more calibration data, not fewer features.

### Riemannian does not fix the confound — it sharpens it

| | 6 channels | 8 channels (Ch2/Ch3 kept) |
| --- | --- | --- |
| predict recording host | **100.0%** | 99.1% |
| rest vs imagery | **100.0%** | 99.1% |

Riemannian features separate the two laptops *perfectly*, better than band
power's 95.7%. That makes sense — covariance structure is precisely what
changes when electrode impedance and montage change between sessions. It is
more session-sensitive, not less.

### The 8-channel result is a clock, not a brain

Keeping the artifact channels lifts hands vs legs to 73.6%, which looks like a
win until you break it down by recording:

| recording | label | t_start | LORO accuracy |
| --- | --- | --- | --- |
| Hands3 | hands | 195s | 100% |
| Hands2 | hands | 305s | 88% |
| Hands1 | hands | 415s | 86% |
| LegImagery5 | legs | 487s | **0%** |
| LegImagery4 | legs | 655s | 100% |
| LegImagery3 | legs | 930s | 100% |
| Legimagery2 | legs | 1018s | 89% |
| LegImagery1 | legs | 1082s | 100% |
| Hands5 | hands | 1161s | **0%** |

Hands was recorded early in the session, legs late. The two recordings that
break that ordering — the earliest legs trial and the one late hands trial —
are *exactly* the two that score 0%. The classifier learned recording time via
drift in the artifact channels. Use the 6-channel configuration.

### Using it

```bash
python3 compare_riemann.py                              # the table above
python3 train_offline.py --riemann --save model.pkl     # train and persist
python3 realtime_classifier.py --model model.pkl        # live, unchanged
```

The saved model records `feature_mode="riemann"`, and the live loop hands the
pipeline raw epochs instead of band-power vectors. Both paths are tested
end-to-end.

### Verdict

Keep Riemannian TS + SVM-RBF as the pipeline — it's the right choice for real
data and costs nothing to keep. Skip PCA at 21 features. Revisit reduction only
if you go to a filter bank or add channels. On *this* dataset every variant is
at or below chance, which is a fact about the data, not the method.

## Why two columns

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
pip install numpy scipy scikit-learn pyriemann pylsl

# 1. Record calibration data — interleaved classes, one subject, one sitting
python3 record_calibration.py --subject alice --classes rest,hands --trials 40

# 2. Train and check the leave-one-recording-out score
python3 train_offline.py --data-dir calibration/alice --riemann --save

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
