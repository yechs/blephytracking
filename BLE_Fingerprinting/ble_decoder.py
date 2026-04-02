"""
Simple BLE GFSK demodulator.

Port of BLE_Decoder.m – extracts the aligned complex signal, instantaneous
frequency waveform, and decoded bit sequence from a received BLE IQ recording.
"""
import numpy as np
from scipy.signal import correlate

from .gfsk_modulate import gfsk_modulate


def ble_decode(signal: np.ndarray, fs: float, preamble_detect: bool = True):
    """
    Decode a received BLE IQ signal.

    Parameters
    ----------
    signal          : complex 1-D ndarray (IQ samples at rate fs)
    fs              : sampling frequency (Hz)
    preamble_detect : if True, use cross-correlation to find and align the
                      preamble; if False, assume the signal starts at sample 0.

    Returns
    -------
    ble_signal   : complex 1-D ndarray – aligned and truncated IQ signal
    signal_freq  : 1-D ndarray – instantaneous frequency (Hz) of ble_signal
    bits         : 1-D bool ndarray – decoded bit sequence
    """
    nsample = int(fs / 1e6)  # samples per bit (BLE 1 Mbit/s → 1 µs/bit)

    # ------------------------------------------------------------------ #
    #  Reference preamble in frequency domain
    # ------------------------------------------------------------------ #
    pream_bits = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    preamble_signal = gfsk_modulate(pream_bits, 500e3, fs)

    # Trim: MATLAB preamble_signal(fs/1e6*2.5 : end-fs/1e6*0.5)
    #       1-indexed start = int(2.5*nsample)  →  0-indexed = int(2.5*nsample)-1
    pream_start = int(2.5 * nsample) - 1
    pream_end = len(preamble_signal) - int(0.5 * nsample)
    preamble_signal = preamble_signal[pream_start:pream_end]

    # Instantaneous frequency of preamble: forward difference of unwrapped phase
    pream_angle = np.unwrap(np.angle(preamble_signal))
    preamble_freq = (pream_angle[1:] - pream_angle[:-1]) / (2.0 * np.pi) * fs
 
    # ------------------------------------------------------------------ #
    #  Instantaneous frequency of received signal
    #  MATLAB uses a 1-sample-offset forward difference (samples 3:end minus 2:end-1)
    # ------------------------------------------------------------------ #
    sig_angle = np.unwrap(np.angle(signal))
    # Python 0-indexed equivalent: sig_angle[2:] - sig_angle[1:-1]
    sig_slope = sig_angle[2:] - sig_angle[1:-1]
    signal_freq = sig_slope / (2.0 * np.pi) * fs
    signal_freq = np.append(signal_freq, 0.0)   # pad to keep length = len(signal)-1

    # ------------------------------------------------------------------ #
    #  Preamble detection via cross-correlation
    # ------------------------------------------------------------------ #
    if not preamble_detect:
        start_ind = 0
    else:
        l = len(signal_freq)

        # scipy correlate(a, b) zero-lag is at index len(b)-1  (0-indexed).
        # MATLAB xcorr(signal_freq, preamble_freq) zero-lag at index l-1 (0-indexed).
        # MATLAB z = z(l+1:end)  →  positive lags starting at lag 1
        # In scipy that corresponds to output[len(preamble_freq):]
        p_len = len(preamble_freq)
        z_full = correlate(signal_freq, preamble_freq, mode="full")
        z_causal = z_full[p_len:]   # positive lags, length = l - 1

        if len(z_causal) > int(20e-6 * fs):
            search_start = int(np.floor(2e-6 * fs)) - 1   # 0-indexed in z_causal
            search_end   = int(np.floor(20e-6 * fs))      # exclusive
            z_slice = z_causal[search_start:search_end]
            # argmax gives 0-indexed position within slice
            # MATLAB max() on same slice gives 1-indexed position = argmax + 1
            # MATLAB then uses that as signal(start_ind:end) (1-indexed)
            # → Python equivalent: signal[argmax:]
            start_ind = int(np.argmax(np.abs(z_slice)))
        else:
            start_ind = 0
            

    signal = signal[start_ind:]
    signal_freq = signal_freq[start_ind:]

    # Truncate to integer number of bit periods
    n_bits = int(len(signal_freq) // nsample)
    signal_freq = signal_freq[: n_bits * nsample]
    ble_signal = signal[: n_bits * nsample]

    # ------------------------------------------------------------------ #
    #  Bit recovery: slice each µs block, sample the middle
    # ------------------------------------------------------------------ #
    bits_freq = signal_freq.reshape(n_bits, nsample)
    mid = nsample // 2
    # MATLAB: mean(bits_freq(:, nsample/2 : nsample/2+1), 2)  (1-indexed cols)
    #         Python 0-indexed: cols mid-1 and mid  (two columns)
    bits = (np.mean(bits_freq[:, mid - 1: mid + 1], axis=1) > 0)

    return ble_signal, signal_freq, bits