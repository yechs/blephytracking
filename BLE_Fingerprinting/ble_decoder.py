import math

import numpy as np

from gfsk_modulate import gfsk_modulate


def _instantaneous_frequency(signal: np.ndarray, fs: float) -> np.ndarray:
    signal_angle = np.unwrap(np.angle(signal))
    slope = signal_angle[2:] - signal_angle[1:-1]
    freq = slope / (2 * math.pi) * fs
    pad = len(signal) - len(freq)
    if pad > 0:
        freq = np.concatenate([freq, np.zeros(pad)])
    return freq


def ble_decoder(signal: np.ndarray, fs: float, preamble_detect: int):
    pream = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    preamble_signal = gfsk_modulate(pream, 500e3, fs)
    sps = int(fs / 1e6)
    preamble_signal = preamble_signal[int(2.5 * sps) : -int(0.5 * sps)]

    preamble_freq = _instantaneous_frequency(preamble_signal, fs)

    signal_freq = _instantaneous_frequency(signal, fs)

    if preamble_detect == 0:
        start_ind = 0
    else:
        sig_len = len(signal_freq)
        z = np.correlate(signal_freq, preamble_freq, mode="full")
        z = z[sig_len:]
        if len(z) > int(20e-6 * fs):
            start_range = int(2e-6 * fs)
            end_range = int(20e-6 * fs)
            start_ind = start_range + np.argmax(np.abs(z[start_range:end_range]))
        else:
            start_ind = 0

    signal = signal[start_ind:]
    signal_freq = signal_freq[start_ind:]

    sample_count = (len(signal_freq) // sps) * sps
    signal_freq = signal_freq[:sample_count]
    ble_signal = signal[:sample_count]

    bits_freq = signal_freq.reshape(-1, sps)
    mid = sps // 2
    bits = np.mean(bits_freq[:, mid : mid + 2], axis=1) > 0
    return ble_signal, signal_freq, bits
