import numpy as np

from ble_fingerprint import ble_fingerprint
from gfsk_modulate import gfsk_modulate


def test_ble_fingerprint_returns_vector_or_none():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=32)
    fs = 1e6
    signal = gfsk_modulate(bits, 500e3, fs)

    fingerprint, decoded_bits = ble_fingerprint(
        signal,
        snr=40,
        fs=fs,
        preamble_detect=0,
        interp_fac=1,
        n_partition=1,
    )

    assert decoded_bits.shape[0] == len(bits)
    if fingerprint is not None:
        assert fingerprint.shape == (25,)
