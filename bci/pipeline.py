"""End-to-end pipeline and honest cross-validation.

The pipeline is band-pass -> CSP -> log-variance -> LDA, the standard
motor-imagery chain. The care in this module is mostly about *evaluation*,
because with a handful of trials it is very easy to produce a number that looks
excellent and means nothing.

Epochs are cut with overlap, so two epochs from the same 10-second recording
share most of their samples and are nearly identical. Splitting those across
train and test leaks the answer and reports accuracies in the high nineties for
a classifier that is at chance on a new trial. :func:`leave_one_trial_out`
therefore holds out whole recordings, which is the only split that estimates
what the game will actually do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from bci.csp import CSP
from bci.data import Trial
from bci.filters import DEFAULT_BAND, DEFAULT_ORDER, design_bandpass, filter_offline

DEFAULT_EPOCH_SECONDS = 2.0
DEFAULT_STEP_SECONDS = 0.5


def make_pipeline(n_components: int = 4, shrinkage: float = 0.1) -> Pipeline:
    """CSP + LDA.

    LDA with Ledoit-Wolf shrinkage rather than an SVM or a forest: with four
    features and a few dozen epochs, a low-variance linear model is the right
    complexity, and it produces calibrated ``predict_proba`` output that the
    game loop can smooth into a continuous control signal.
    """
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, shrinkage=shrinkage)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def epoch_dataset(
    trials: list[Trial],
    band: tuple[float, float] = DEFAULT_BAND,
    order: int = DEFAULT_ORDER,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    discard_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter each trial, then cut it into overlapping epochs.

    Returns ``(X, y, groups)`` where ``X`` is (n_epochs, n_channels, n_samples),
    ``y`` holds the class labels and ``groups`` the originating trial name --
    which is what keeps the cross-validation honest.

    ``discard_seconds`` drops the start of each recording, where the causal
    filter is still settling. That transient is identical in shape for every
    trial and carries no class information, but it is large, and leaving it in
    lets CSP fit it.
    """
    if not trials:
        raise ValueError("no trials given")

    fs = float(np.mean([t.fs for t in trials]))
    sos = design_bandpass(fs, band, order)
    width = int(round(epoch_seconds * fs))
    step = int(round(step_seconds * fs))
    skip = int(round(discard_seconds * fs))
    if width <= 0 or step <= 0:
        raise ValueError("epoch_seconds and step_seconds must be positive")

    epochs: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []

    for trial in trials:
        filtered = filter_offline(trial.data, sos)[:, skip:]
        if filtered.shape[1] < width:
            raise ValueError(
                f"{trial.name}: {filtered.shape[1] / fs:.1f}s left after discarding "
                f"the filter transient, need {epoch_seconds:g}s"
            )
        for start in range(0, filtered.shape[1] - width + 1, step):
            epochs.append(filtered[:, start : start + width])
            labels.append(trial.label)
            groups.append(trial.name)

    return np.asarray(epochs), np.asarray(labels), np.asarray(groups)


@dataclass
class CVResult:
    """Cross-validation outcome, with the caveats needed to read it."""

    accuracy: float
    chance: float
    per_fold: dict[str, float]
    n_epochs: int
    n_trials: int
    trial_accuracy: float
    p_value: float
    confusion: np.ndarray
    classes: list[str]

    def report(self) -> str:
        lines = [
            f"Leave-one-trial-out CV over {self.n_trials} recordings "
            f"({self.n_epochs} epochs, {len(self.classes)} classes)",
            "",
            f"  epoch accuracy   {self.accuracy:6.1%}   (chance {self.chance:.1%})",
            f"  trial accuracy   {self.trial_accuracy:6.1%}   "
            f"(majority vote within each held-out recording)",
            f"  p-value          {self.p_value:6.3f}   "
            f"(binomial, epochs treated as independent -- optimistic)",
            "",
            "  per held-out recording:",
        ]
        for name, acc in sorted(self.per_fold.items()):
            flag = "" if acc > self.chance else "   <- at or below chance"
            lines.append(f"    {name:<18} {acc:6.1%}{flag}")
        lines += ["", "  confusion (rows = true, cols = predicted):", ""]
        header = " " * 18 + "".join(f"{c:>14}" for c in self.classes)
        lines.append(header)
        for i, cls in enumerate(self.classes):
            row = "".join(f"{v:>14d}" for v in self.confusion[i])
            lines.append(f"    {cls:<14}{row}")
        return "\n".join(lines)


def leave_one_trial_out(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, pipeline_factory=make_pipeline
) -> CVResult:
    """Hold out one whole recording at a time and score the rest.

    With a handful of trials this is high-variance, but it is unbiased. A
    k-fold over epochs would be far smoother and completely wrong.
    """
    classes = sorted(set(y.tolist()))
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 3:
        raise ValueError(
            f"only {len(unique_groups)} recordings; leave-one-out needs at least 3"
        )

    index = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    per_fold: dict[str, float] = {}
    correct = 0
    total = 0
    trial_correct = 0

    for held_out in unique_groups:
        test_mask = groups == held_out
        train_mask = ~test_mask
        if len(set(y[train_mask].tolist())) < 2:
            raise ValueError(
                f"holding out {held_out} leaves fewer than two classes to train on"
            )

        model = pipeline_factory()
        model.fit(X[train_mask], y[train_mask])
        predicted = model.predict(X[test_mask])
        truth = y[test_mask]

        per_fold[held_out] = float(np.mean(predicted == truth))
        correct += int(np.sum(predicted == truth))
        total += len(truth)
        for t, p in zip(truth, predicted):
            confusion[index[t], index[p]] += 1

        # How the game would actually decide, given a whole recording to vote on.
        values, counts = np.unique(predicted, return_counts=True)
        if values[np.argmax(counts)] == truth[0]:
            trial_correct += 1

    chance = max(np.mean(y == c) for c in classes)
    p_value = float(stats.binomtest(correct, total, chance, alternative="greater").pvalue)

    return CVResult(
        accuracy=correct / total,
        chance=float(chance),
        per_fold=per_fold,
        n_epochs=len(y),
        n_trials=len(unique_groups),
        trial_accuracy=trial_correct / len(unique_groups),
        p_value=p_value,
        confusion=confusion,
        classes=classes,
    )
