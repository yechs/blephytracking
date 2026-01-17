import math

import numpy as np


def _gaussdesign(bt: float, span: int, samples_per_symbol: int) -> np.ndarray:
    """Approximate MATLAB gaussdesign for GFSK pulse shaping."""
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    t = np.linspace(-span / 2, span / 2, span * samples_per_symbol + 1)
    alpha = math.sqrt(math.log(2.0)) / bt
    h = math.sqrt(math.pi) / alpha * np.exp(-((math.pi * t) / alpha) ** 2)
    h = h / np.sum(h)
    return h


def _cumulative_trapezoid(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    trapezoids = (x[1:] + x[:-1]) / 2
    return np.concatenate([[0.0], np.cumsum(trapezoids)])


def gfsk_modulate(bits: np.ndarray, freqsep: float, fs: float) -> np.ndarray:
    """Generate a BLE GFSK signal for the provided bit sequence."""
    samples_per_symbol = int(fs / 1e6)
    t = np.arange(samples_per_symbol * len(bits)) / fs
    gamma_fsk = np.empty_like(t, dtype=float)
    for i, bit in enumerate(bits):
        gamma_fsk[i * samples_per_symbol : (i + 1) * samples_per_symbol] = (bit * 2) - 1

    gauss_filter = _gaussdesign(0.3, 3, samples_per_symbol)
    gamma_gfsk = np.convolve(gamma_fsk, gauss_filter, mode="full")[: len(gamma_fsk)]
    gfsk_phase = (freqsep / fs) * math.pi * _cumulative_trapezoid(gamma_gfsk)
    return np.exp(1j * gfsk_phase)
