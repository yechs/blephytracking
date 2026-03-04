import numpy as np
from scipy.signal import lfilter


def _gaussdesign(BT: float, span: int, sps: int) -> np.ndarray:
    """
    Gaussian pulse-shaping FIR filter equivalent to MATLAB's gaussdesign(BT, span, sps).
    BT:   3-dB bandwidth-symbol period product (e.g. 0.3 for BLE)
    span: filter length in symbol periods
    sps:  samples per symbol
    """
    # sigma in symbol periods such that 3-dB bandwidth = BT
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
    sigma_samp = sigma * sps  # convert to samples
    t = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float)
    h = np.exp(-t**2 / (2.0 * sigma_samp**2))
    h /= h.sum()
    return h


def gfsk_modulate(bits, freqsep: float, Fs: float) -> np.ndarray:
    """
    Generate a complex GFSK-modulated BLE signal.

    Parameters
    ----------
    bits    : array-like of 0/1 bit values
    freqsep : frequency separation between '0' and '1' bits (Hz), typically 500e3
    Fs      : sampling frequency (Hz)

    Returns
    -------
    y : complex column ndarray of shape (N,)
    """
    bits = np.asarray(bits, dtype=float)
    nsample = int(Fs / 1e6)          # samples per bit (BLE = 1 Mbit/s → 1 µs/bit)

    # Map bits → ±1 frequency deviation waveform
    gamma_fsk = np.repeat(2.0 * bits - 1.0, nsample)

    # Gaussian pulse shaping (BT=0.3, span=3 symbols)
    gauss_filter = _gaussdesign(0.3, 3, nsample)
    gamma_gfsk = lfilter(gauss_filter, 1.0, gamma_fsk)

    # Phase = integral of frequency deviation  (cumtrapz on uniform grid, initial=0)
    # scipy.integrate.cumulative_trapezoid(..., initial=0) matches MATLAB cumtrapz
    from scipy.integrate import cumulative_trapezoid
    gfsk_phase = (freqsep / Fs) * np.pi * cumulative_trapezoid(gamma_gfsk, initial=0.0)

    y = np.exp(1j * gfsk_phase)
    return y  # column vector (1-D complex array)
