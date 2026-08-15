"""
Dummy input sender for testing UDPInputReceiver.cs.

Sends a "flap" UDP packet to Unity every time you press Enter.
No dependencies beyond the Python standard library.

Usage:
    python udp_dummy_input.py

Make sure UDPInputReceiver's listenPort in Unity matches PORT below,
and that the game is in Play mode with the bird already flying
(BeginFlying() must have been called, i.e. Start button pressed).
"""

import socket

HOST = "127.0.0.1"   # localhost — change if Unity is running on a different machine
PORT = 5005           # must match UDPInputReceiver.listenPort in Unity
FLAP_MESSAGE = "1"    # must match UDPInputReceiver.flapMessage in Unity

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending flap commands to {HOST}:{PORT}")
print("Press Enter to send a flap. Type 'q' then Enter to quit.\n")

while True:
    user_input = input("> ")
    if user_input.strip().lower() == "q":
        break

    sock.sendto(FLAP_MESSAGE.encode("utf-8"), (HOST, PORT))
    print(f"Sent flap (\"{FLAP_MESSAGE}\")")

sock.close()
print("Closed.")
