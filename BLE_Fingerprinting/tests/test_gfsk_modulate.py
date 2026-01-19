import numpy as np

from gfsk_modulate import gfsk_modulate


def test_gfsk_modulate_length_and_magnitude():
    bits = np.array([0, 1, 1, 0, 1])
    fs = 1e6
    signal = gfsk_modulate(bits, 500e3, fs)

    assert len(signal) == len(bits) * int(fs / 1e6)
    np.testing.assert_allclose(np.abs(signal), np.ones_like(signal), atol=1e-6)
