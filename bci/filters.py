"""Butterworth band-pass filtering, offline and streaming.

Two rules are enforced here because breaking either is the usual reason a
motor-imagery BCI cross-validates well and then fails live:

1. **Same filter in training and inference.** The streaming filter and the
   offline filter are built from the same ``sos`` coefficients and both run
   causally, so the features a classifier is trained on match the ones it sees
   in the game.
2. **Second-order sections, not transfer-function coefficients.** ``b, a`` form
   goes numerically unstable above roughly order 5; ``sos`` does not.

Phase distortion is not corrected for, and does not need to be: the features
downstream are band-power (log-variance), which discards phase entirely.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Mu (8-13 Hz) and beta (13-30 Hz) together: the sensorimotor rhythms whose
# event-related desynchronisation carries motor imagery. Wider bands mostly add
# ocular drift below and EMG above.
DEFAULT_BAND = (8.0, 30.0)
DEFAULT_ORDER = 4


def design_bandpass(
    fs: float, band: tuple[float, float] = DEFAULT_BAND, order: int = DEFAULT_ORDER
) -> np.ndarray:
    """Butterworth band-pass as second-order sections.

    Butterworth is maximally flat in the passband, so it does not ripple the
    band-power that the classifier reads as its feature.
    """
    low, high = band
    nyquist = fs / 2.0
    if not 0 < low < high < nyquist:
        raise ValueError(
            f"band {band} is not valid for fs={fs:g} Hz (needs 0 < low < high < {nyquist:g})"
        )
    return signal.butter(order, [low, high], btype="band", fs=fs, output="sos")


def initial_state(sos: np.ndarray, first_sample: np.ndarray) -> np.ndarray:
    """Filter state matching a signal that has been sitting at ``first_sample``.

    Starting from a zero state instead makes the filter ring its way up from
    zero to the signal's operating point over the first second or so. That
    transient is large, identical across recordings, and completely unrelated
    to class -- exactly the kind of structure CSP will latch onto if it is left
    in the training data but absent at run time.
    """
    zi = signal.sosfilt_zi(sos)                       # (n_sections, 2)
    return zi[:, None, :] * first_sample[None, :, None]  # (n_sections, n_channels, 2)


def filter_offline(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter ``(n_channels, n_samples)`` causally along time.

    Deliberately ``sosfilt`` and not ``sosfiltfilt``: the live pipeline cannot
    see the future, so training on zero-phase data would fit a classifier to
    features that never occur at run time.

    Uses the same state initialisation as :class:`StreamingBandpass`, so an
    offline call and a chunked live call on the same samples agree to numerical
    precision. :func:`bci.tests` asserts exactly that.
    """
    if data.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got {data.shape}")
    if data.shape[1] == 0:
        return data
    zi = initial_state(sos, data[:, 0])
    filtered, _ = signal.sosfilt(sos, data, axis=-1, zi=zi)
    return filtered


class StreamingBandpass:
    """Stateful band-pass for chunk-by-chunk filtering of a live stream.

    Carrying the filter state across chunks is what keeps the output identical
    to filtering the whole recording at once. Filtering each window from a zero
    state instead injects a startup transient into every window -- a large,
    perfectly label-independent artifact that a classifier will happily fit.

    >>> bp = StreamingBandpass(fs=125.0, n_channels=8)
    >>> chunk = np.random.randn(8, 32)
    >>> out = bp(chunk)
    >>> out.shape
    (8, 32)
    """

    def __init__(
        self,
        fs: float,
        n_channels: int,
        band: tuple[float, float] = DEFAULT_BAND,
        order: int = DEFAULT_ORDER,
    ) -> None:
        self.fs = fs
        self.n_channels = n_channels
        self.band = band
        self.sos = design_bandpass(fs, band, order)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the filter state; the next chunk starts a fresh recording."""
        self._zi = None

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Filter ``(n_channels, n_samples)`` and advance the internal state."""
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"expected ({self.n_channels}, n_samples), got {chunk.shape}"
            )
        if chunk.shape[1] == 0:
            return chunk

        if self._zi is None:
            self._zi = initial_state(self.sos, chunk[:, 0])

        filtered, self._zi = signal.sosfilt(self.sos, chunk, axis=-1, zi=self._zi)
        return filtered
