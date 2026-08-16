"""Motor-imagery BCI pipeline for the ISRC CN3 Hackathon 2026.

Modules
-------
data        loading and auditing the FlexEEG CSV recordings
filters     Butterworth band-pass filtering (offline and streaming)
csp         Common Spatial Patterns, as an sklearn transformer
pipeline    end-to-end feature extraction + classifier, with honest CV
"""

from bci.data import Trial, load_csv, load_dataset, audit
from bci.filters import design_bandpass, filter_offline, initial_state, StreamingBandpass
from bci.csp import CSP
from bci.pipeline import make_pipeline, epoch_dataset, leave_one_trial_out

__all__ = [
    "Trial", "load_csv", "load_dataset", "audit",
    "design_bandpass", "filter_offline", "initial_state", "StreamingBandpass",
    "CSP", "make_pipeline", "epoch_dataset", "leave_one_trial_out",
]
