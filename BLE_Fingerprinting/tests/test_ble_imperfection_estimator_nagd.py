import numpy as np

from ble_imperfection_estimator_nagd import ble_imperfection_estimator_nagd
from gfsk_modulate import gfsk_modulate


def test_ble_imperfection_estimator_outputs_finite():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=16)
    fs = 1e6
    signal = gfsk_modulate(bits, 500e3, fs)

    result = ble_imperfection_estimator_nagd(
        signal,
        bits,
        fs,
        init_f0=0,
        init_e=0,
        init_phi=0,
        init_I=0,
        init_Q=0,
        init_amp=1,
        snr=40,
        n_partition=1,
    )

    amp, e, phi, I, Q, IQO, IQI, f0, phi_off, error, _ = result
    values = np.array([amp, e, phi, I, Q, IQO, IQI, f0, phi_off, error])
    assert np.all(np.isfinite(values))
