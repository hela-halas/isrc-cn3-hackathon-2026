#!/usr/bin/env python3
"""Tests for the BCI pipeline.

    python test_bci.py

Plain unittest so it runs with no extra dependencies on the game machine. The
tests that matter most are the ones asserting that the offline and streaming
paths produce identical output, and that cross-validation cannot leak epochs
between folds -- those are the two failure modes that silently inflate accuracy.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bci.csp import CSP
from bci.data import flag_bad_channels, label_from_filename, load_csv, load_dataset
from bci.filters import design_bandpass, filter_offline, StreamingBandpass
from bci.pipeline import epoch_dataset, leave_one_trial_out, make_pipeline

FS = 125.0
DATA_DIR = Path(__file__).parent / "EEG Hackathon data bios and one"


def synthetic_csv(
    path: Path, label_hz: float, source_channel: int = 0, n: int = 1300, seed: int = 0
) -> None:
    """Write a FlexEEG-shaped CSV with a known oscillation on one channel.

    ``source_channel`` is what the two classes must differ in. CSP separates
    classes by *spatial* variance ratios, so two classes whose oscillations sit
    on the same electrode at different frequencies are -- correctly -- not
    separable by it, however obvious they look on a spectrogram.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    data = rng.standard_normal((9, n)) * 20
    # Kept modest on purpose: a source loud enough to trip the bad-channel
    # screen in load_dataset would be dropped before it reached CSP.
    data[source_channel] += 60 * np.sin(2 * np.pi * label_hz * t)
    data[8] = 0.0  # Ch9 is dead in the real recordings
    with path.open("w") as fh:
        fh.write("# Stream: FlexEEG (EEG) from test\n# Time range\n# Buffer\n")
        fh.write("timestamp,relative_time,Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8,Ch9\n")
        for i in range(n):
            row = ",".join(f"{v:.4f}" for v in data[:, i])
            fh.write(f"{1000 + i / FS:.6f},{i / FS:.6f},{row}\n")


class TestFilters(unittest.TestCase):
    def test_streaming_matches_offline_exactly(self):
        """The whole design rests on this: chunked live filtering must equal
        one-shot offline filtering, or training and inference see different
        features."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((7, 1000)) + 50.0  # offset, to exercise state init
        sos = design_bandpass(FS)

        bandpass = StreamingBandpass(FS, 7)
        pieces, i = [], 0
        for size in [13, 7, 100, 1, 250, 29, 600]:
            if i >= x.shape[1]:
                break
            pieces.append(bandpass(x[:, i : i + size]))
            i += size
        pieces.append(bandpass(x[:, i:]))

        np.testing.assert_allclose(
            np.concatenate(pieces, axis=1), filter_offline(x, sos), atol=1e-12
        )

    def test_reset_clears_state(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 300))
        bandpass = StreamingBandpass(FS, 4)
        first = bandpass(x)
        bandpass.reset()
        np.testing.assert_allclose(first, bandpass(x), atol=1e-12)

    def test_passband_keeps_signal_and_stopband_rejects(self):
        t = np.arange(1250) / FS
        sos = design_bandpass(FS, (8.0, 30.0))
        for freq, keep in [(2.0, False), (12.0, True), (20.0, True), (55.0, False)]:
            wave = np.sin(2 * np.pi * freq * t)[None, :]
            retained = filter_offline(wave, sos)[0, 300:].std() / wave[0, 300:].std()
            if keep:
                self.assertGreater(retained, 0.9, f"{freq} Hz should pass")
            else:
                self.assertLess(retained, 0.1, f"{freq} Hz should be rejected")

    def test_rejects_band_above_nyquist(self):
        with self.assertRaises(ValueError):
            design_bandpass(125.0, (8.0, 70.0))

    def test_rejects_wrong_channel_count(self):
        bandpass = StreamingBandpass(FS, 4)
        with self.assertRaises(ValueError):
            bandpass(np.zeros((5, 10)))


class TestCSP(unittest.TestCase):
    def _planted(self, n_epochs=40, n_channels=6, n_samples=250, seed=0):
        """Two classes differing only in which channel mixture is loud."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(n_epochs):
            epoch = rng.standard_normal((n_channels, n_samples))
            if i % 2:
                epoch[0] *= 4.0
                y.append("a")
            else:
                epoch[1] *= 4.0
                y.append("b")
            X.append(epoch)
        return np.asarray(X), np.asarray(y)

    def test_recovers_planted_source(self):
        X, y = self._planted()
        csp = CSP(n_components=2).fit(X, y)
        # The leading filter should load on channel 0 or 1, the planted ones.
        dominant = int(np.argmax(np.abs(csp.filters_[0])))
        self.assertIn(dominant, (0, 1))

    def test_features_separate_classes(self):
        X, y = self._planted()
        features = CSP(n_components=2).fit(X, y).transform(X)
        a = features[y == "a", 0]
        b = features[y == "b", 0]
        self.assertGreater(abs(a.mean() - b.mean()), 2 * (a.std() + b.std()) / 2)

    def test_output_shape_and_finiteness(self):
        X, y = self._planted()
        features = CSP(n_components=4).fit(X, y).transform(X)
        self.assertEqual(features.shape, (len(X), 4))
        self.assertTrue(np.isfinite(features).all())

    def test_rejects_three_classes(self):
        X, y = self._planted()
        y = y.copy()
        y[:5] = "c"
        with self.assertRaises(ValueError):
            CSP().fit(X, y)

    def test_rejects_odd_components(self):
        X, y = self._planted()
        with self.assertRaises(ValueError):
            CSP(n_components=3).fit(X, y)

    def test_rejects_more_components_than_channels(self):
        X, y = self._planted(n_channels=4)
        with self.assertRaises(ValueError):
            CSP(n_components=6).fit(X, y)


class TestData(unittest.TestCase):
    def test_label_parsing_folds_case(self):
        self.assertEqual(label_from_filename(Path("LegImagery3.csv")), "legimagery")
        self.assertEqual(label_from_filename(Path("Legimagery2.csv")), "legimagery")
        self.assertEqual(label_from_filename(Path("Rest1 (4).csv")), "rest")

    def test_flags_dead_and_railing_channels(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 500))
        data[2] *= 50.0   # railing
        data[4] = 0.0     # dead
        self.assertEqual(flag_bad_channels(data), [2, 4])

    def test_excludes_duplicates_and_wrong_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_csv(tmp / "Hands1.csv", 10.0, seed=1)
            synthetic_csv(tmp / "Hands2.csv", 10.0, seed=2)
            synthetic_csv(tmp / "Hands3.csv", 10.0, seed=2)  # duplicate of Hands2
            synthetic_csv(tmp / "LegImagery1.csv", 10.0, source_channel=1, seed=3)
            (tmp / "Fake1.csv").write_text(
                "timestamp,relative_time,Fp1,Fp2\n" + "".join(
                    f"{i / FS},{i / FS},1.0,2.0\n" for i in range(200)
                )
            )
            trials = load_dataset(tmp, labels=["hands", "legimagery"])
        names = sorted(t.name for t in trials)
        self.assertEqual(names, ["Hands1", "Hands2", "LegImagery1"])
        self.assertTrue(all("Ch9" not in t.channels for t in trials))

    def test_measures_rate_from_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Hands1.csv"
            synthetic_csv(path, 10.0)
            trial = load_csv(path)
        self.assertAlmostEqual(trial.fs, FS, delta=0.5)


class TestPipeline(unittest.TestCase):
    def _dataset(self, tmp: Path):
        """Four trials per class, differing in which channel carries the source."""
        for i in range(4):
            synthetic_csv(tmp / f"Hands{i + 1}.csv", 10.0, source_channel=0, seed=10 + i)
        for i in range(4):
            synthetic_csv(tmp / f"LegImagery{i + 1}.csv", 10.0, source_channel=1, seed=20 + i)
        return load_dataset(tmp, labels=["hands", "legimagery"])

    def test_epochs_carry_their_trial_as_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._dataset(Path(tmp))
            X, y, groups = epoch_dataset(trials, epoch_seconds=2.0, step_seconds=0.5)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(groups))
        self.assertEqual(set(groups.tolist()), {t.name for t in trials})
        # Every epoch of a trial must carry that trial's single label.
        for group in set(groups.tolist()):
            self.assertEqual(len(set(y[groups == group].tolist())), 1)

    def test_cross_validation_never_shares_a_trial(self):
        """The leak that inflates accuracy: overlapping epochs from one
        recording landing in both train and test."""
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._dataset(Path(tmp))
            X, y, groups = epoch_dataset(trials)

        seen = []
        original = make_pipeline

        def spy():
            model = original()
            fit = model.fit

            def wrapped(Xtr, ytr, **kw):
                seen.append(Xtr.shape[0])
                return fit(Xtr, ytr, **kw)

            model.fit = wrapped
            return model

        result = leave_one_trial_out(X, y, groups, spy)
        self.assertEqual(result.n_trials, len(trials))
        self.assertEqual(len(seen), len(trials))
        for held_out, n_train in zip(sorted(set(groups.tolist())), seen):
            self.assertEqual(n_train, int(np.sum(groups != held_out)))

    def test_separable_signal_is_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._dataset(Path(tmp))
            X, y, groups = epoch_dataset(trials)
            result = leave_one_trial_out(X, y, groups)
        self.assertGreater(result.accuracy, 0.8)
        self.assertLess(result.p_value, 0.05)

    def test_refuses_too_few_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_csv(tmp / "Hands1.csv", 10.0, source_channel=0, seed=1)
            synthetic_csv(tmp / "LegImagery1.csv", 10.0, source_channel=1, seed=2)
            trials = load_dataset(tmp, labels=["hands", "legimagery"])
            X, y, groups = epoch_dataset(trials)
            with self.assertRaises(ValueError):
                leave_one_trial_out(X, y, groups)

    def test_epoch_longer_than_recording_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._dataset(Path(tmp))
            with self.assertRaises(ValueError):
                epoch_dataset(trials, epoch_seconds=60.0)


@unittest.skipUnless(DATA_DIR.is_dir(), "hackathon recordings not present")
class TestRealRecordings(unittest.TestCase):
    def test_loads_the_nine_usable_motor_imagery_trials(self):
        trials = load_dataset(DATA_DIR, labels=["hands", "legimagery"])
        self.assertEqual(len(trials), 9)
        self.assertNotIn("Hands4", [t.name for t in trials])  # synthetic demo_eeg dump
        for trial in trials:
            self.assertAlmostEqual(trial.fs, 125.5, delta=1.0)

    def test_beats_chance_out_of_sample(self):
        trials = load_dataset(DATA_DIR, labels=["hands", "legimagery"])
        X, y, groups = epoch_dataset(trials)
        result = leave_one_trial_out(X, y, groups)
        self.assertGreater(result.accuracy, result.chance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
