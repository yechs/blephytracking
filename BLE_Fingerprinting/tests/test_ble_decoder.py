import numpy as np

from ble_decoder import ble_decoder
from gfsk_modulate import gfsk_modulate


def test_ble_decoder_shapes():
    bits = np.array([0, 1, 0, 1, 1, 0, 1])
    fs = 2e6
    signal = gfsk_modulate(bits, 500e3, fs)

    ble_signal, signal_freq, decoded = ble_decoder(signal, fs, preamble_detect=0)

    assert ble_signal.shape == signal_freq.shape
    assert decoded.dtype == bool
    assert len(decoded) == len(bits)
