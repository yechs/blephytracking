#!/usr/bin/env python3
"""
BLE Physical-Layer Fingerprinting – Example 2

Reads 20 BLE signals captured over-the-air:
  - Signals  1-10 : device A
  - Signals 11-20 : device B

Extracts a 25-dimensional hardware-imperfection fingerprint for each signal
and prints the result matrix.

Usage
-----
    python ble_fingerprinting_example2.py

Dependencies
------------
    numpy, scipy
    (install with:  pip install numpy scipy)
"""
import os
import struct
import time
import numpy as np

# Resolve path to Example_Data relative to this script
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_HERE, "BLE_Fingerprinting", "Example_Data")

# Make sure the package is importable when running from the repo root
import sys
sys.path.insert(0, _HERE)

from BLE_Fingerprinting.ble_fingerprint import ble_fingerprint


# ── Parameters (match BLE_Fingerprinting_example2.m exactly) ─────────────── #
Fs               = 3.125e6   # baseband sampling rate (Hz)
snr              = 40        # SNR (dB)
preamble_detect  = True
interp_fac       = 32
n_partition      = 250
fingerprint_size = 1
n_signals        = 20
# ─────────────────────────────────────────────────────────────────────────── #


def load_signal(filepath: str) -> np.ndarray:
    """
    Load a binary IQ file stored as interleaved float32 pairs (I, Q, I, Q, …).
    Returns a complex 1-D ndarray.
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()
    floats = np.frombuffer(raw, dtype=np.float32)
    samples = floats.reshape(-1, 2)
    return samples[:, 0] + 1j * samples[:, 1]


def main():
    fingerprint_all = np.zeros((n_signals, fingerprint_size))

    t_start = time.perf_counter()

    for i in range(1, n_signals + 1):
        filepath = os.path.join(_DATA_DIR, str(i))
        signal   = load_signal(filepath)

        # Remove trailing padding (MATLAB: signal = signal(1:end-12))
        signal = signal[:-12]

        fp, bits = ble_fingerprint(
            signal,
            snr,
            Fs,
            preamble_detect=preamble_detect,
            interp_fac=interp_fac,
            n_partition=n_partition,
        )

        if fp.size == fingerprint_size:
            fingerprint_all[i - 1, :] = fp
            print(f"  Signal {i:2d}: OK   (error={fp[0]:.4f})")
        else:
            print(f"  Signal {i:2d}: FAILED (quality check not met)")

    elapsed = time.perf_counter() - t_start
    print(f"\nElapsed time: {elapsed:.2f} s")

    print("\nFingerprint matrix (20 × 25):")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(fingerprint_all)

    return fingerprint_all


if __name__ == "__main__":
    main()
