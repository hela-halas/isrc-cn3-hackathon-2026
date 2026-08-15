"""
Dummy input sender for testing LSLInputReceiver.cs (or LSL4UnityBirdInlet.cs).

Publishes a single-channel LSL stream and pushes a "flap" sample every time
you press Enter, and an "idle" sample right after — mimicking a real
classifier that briefly reports "flap" then returns to idle.

Requires pylsl:
    pip install pylsl

Usage:
    python lsl_dummy_input.py
"""

import time
from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = "MotorImagery"   # must match streamName in LSLInputReceiver.cs
STREAM_TYPE = "Markers"
CHANNEL_COUNT = 1
SAMPLE_RATE = 0                # irregular rate — we push samples manually, not on a clock

FLAP_VALUE = 1.0                # must match flapValue in LSLInputReceiver.cs
IDLE_VALUE = 0.0

info = StreamInfo(STREAM_NAME, STREAM_TYPE, CHANNEL_COUNT, SAMPLE_RATE, "float32", "dummy-source-001")
outlet = StreamOutlet(info)

print(f"Publishing LSL stream '{STREAM_NAME}' (type '{STREAM_TYPE}')")
print("Unity should detect this automatically within a couple of seconds.")
print("Press Enter to push a flap sample. Type 'q' then Enter to quit.\n")

while True:
    user_input = input("> ")
    if user_input.strip().lower() == "q":
        break

    outlet.push_sample([FLAP_VALUE])
    print(f"Pushed flap ({FLAP_VALUE})")

    # Briefly return to idle, similar to a real classifier's output between detections
    time.sleep(0.05)
    outlet.push_sample([IDLE_VALUE])

print("Closed.")
