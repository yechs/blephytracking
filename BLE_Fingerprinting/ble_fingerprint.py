"""
BLE physical-layer fingerprinting – main processing function.

Port of BLE_Fingerprint.m
Source: "Evaluating Physical-Layer BLE Location Tracking Attacks on Mobile Devices"
        IEEE Symposium on Security and Privacy 2022
"""
import warnings
import numpy as np
from scipy.signal import resample_poly

from .ble_decoder import ble_decode
from .ble_imperfection_estimator_nagd import ble_imperfection_estimator_nagd
from .fit_ellipse import fit_ellipse


def ble_fingerprint(
    signal: np.ndarray,
    snr: float,
    fs: float,
    preamble_detect: bool = True,
    interp_fac: int = 32,
    n_partition: int = 250,
):
    """
    Compute the 25-dimensional hardware-imperfection fingerprint for one BLE signal.

    Parameters
    ----------
    signal          : complex IQ samples at rate `fs`
    snr             : SNR in dB (passed to the NAGD estimator)
    fs              : baseband sampling frequency before interpolation (Hz)
    preamble_detect : enable preamble-correlation alignment in the decoder
    interp_fac      : upsample factor applied before processing
    n_partition     : random partitions for the NAGD estimator

    Returns
    -------
    fingerprint : 1-D ndarray of length 25, or empty array if quality fails
    bits        : decoded bit sequence (1-D bool ndarray)
    """
    # Receiver / channel centre (both 2.48 GHz → no shift)
    fc = 2.48e9
    ch = 2.48e9

    fingerprint = np.array([])

    # ------------------------------------------------------------------ #
    # 1.  Normalise
    # ------------------------------------------------------------------ #
    signal = signal / np.mean(np.abs(signal))

    # ------------------------------------------------------------------ #
    # 2.  Upsample  (MATLAB interp → resample_poly with lowpass filter)
    # ------------------------------------------------------------------ #
    fs_interp = fs * interp_fac
    if interp_fac != 1:
        signal = resample_poly(signal, interp_fac, 1)

    # ------------------------------------------------------------------ #
    # 3.  Channel centering – extract BLE band and shift to DC
    # ------------------------------------------------------------------ #
    if interp_fac != 1:
        ls          = len(signal)
        sig_fft     = np.fft.fftshift(np.fft.fft(signal))
        lcenter     = ls // 2                                     # 0-indexed centre bin
        # MATLAB: lchannel = floor((ch-fc)/fs*ls + ls/2)  (1-indexed)
        # Python 0-indexed: same formula gives the 0-indexed bin when ch==fc
        lchannel    = int(np.floor((ch - fc) / fs_interp * ls + ls / 2))
        lbandwidth  = int(np.floor(2.0 / (fs_interp / 1e6) * (ls - 1) / 2))

        sig_fft_c   = np.zeros(ls, dtype=complex)
        # MATLAB (1-indexed): signal_fft_centered(lcenter-lbw : lcenter+lbw) = ...
        # Python (0-indexed):
        sig_fft_c[lcenter - lbandwidth: lcenter + lbandwidth + 1] = \
            sig_fft[lchannel - lbandwidth: lchannel + lbandwidth + 1]

        signal = np.fft.ifft(np.fft.ifftshift(sig_fft_c))

    # ------------------------------------------------------------------ #
    # 4.  BLE decoding – align preamble, extract frequency & bits
    # ------------------------------------------------------------------ #
    signal, signal_freq, bits = ble_decode(signal, fs_interp, preamble_detect)

    # ------------------------------------------------------------------ #
    # 5.  Initial CFO estimate from preamble averaging
    # ------------------------------------------------------------------ #
    pream_len  = int(fs_interp / 1e6 * 8) - 1
    pream_freq = signal_freq[:pream_len]
    est_cfo    = float(np.mean(pream_freq))
    est_cfo2   = est_cfo if abs(est_cfo) <= 100e3 else 0.0

    # ------------------------------------------------------------------ #
    # 6.  NAGD imperfection estimation
    # ------------------------------------------------------------------ #
    amp, epsilon, phi, I, Q, IQO, IQI, f0, phi_off, error = \
        ble_imperfection_estimator_nagd(
            signal, bits, fs_interp, est_cfo2,
            0.0, 0.0, 0.0, 0.0, 1.0, snr, n_partition,
        )

    # ------------------------------------------------------------------ #
    # 7.  Frequency / phase correction
    # ------------------------------------------------------------------ #
    tt     = np.arange(len(signal)) / fs_interp
    signal = signal * np.exp(-1j * (2.0 * np.pi * f0 * tt
                                    + phi_off / (360.0 / (2.0 * np.pi))))

    # ------------------------------------------------------------------ #
    # 8.  Ellipse fit on shuffled I/Q constellation
    # ------------------------------------------------------------------ #
    flag = True
    try:
        sig_shuffled = signal[np.random.permutation(len(signal))]
        ell = fit_ellipse(
            -np.real(sig_shuffled) / amp * 5.0,
             3.0 * np.imag(sig_shuffled) / amp,
        )
    except (ValueError, ZeroDivisionError):
        warnings.warn("Ill ellipse")
        flag = False
        return fingerprint, bits

    # ------------------------------------------------------------------ #
    # 9.  Quadrant signal statistics (8 angular segments)
    # ------------------------------------------------------------------ #
    angsig = np.angle(signal)   # in (-π, π]
    spl    = 8
    quar   = np.zeros(spl, dtype=complex)

    for sp in range(1, spl // 2 + 1):
        lo = (sp - 1) * 2.0 * np.pi / spl
        hi =  sp      * 2.0 * np.pi / spl
        mask = (angsig > lo) & (angsig < hi)
        quar[sp - 1] = np.mean(signal[mask]) if np.any(mask) else 0.0

        lo2 = -np.pi + (sp - 1) * 2.0 * np.pi / spl
        hi2 = -np.pi +  sp      * 2.0 * np.pi / spl
        mask2 = (angsig > lo2) & (angsig < hi2)
        quar[sp - 1 + spl // 2] = np.mean(signal[mask2]) if np.any(mask2) else 0.0

    # ------------------------------------------------------------------ #
    # 10. Assemble 25-D fingerprint vector
    # ------------------------------------------------------------------ #
    quar_mean   = np.mean(quar)
    sig_re_mean = np.mean(np.real(signal))
    sig_im_mean = np.mean(np.imag(signal))

    fingerprint_vec = np.array([
        error,                                                  # 0
        amp,                                                    # 1
        f0,                                                     # 2
        est_cfo,                                                # 3
        IQO,                                                    # 4
        I,                                                      # 5
        Q,                                                      # 6
        np.sqrt(I**2 + Q**2),                                   # 7
        IQI,                                                    # 8
        epsilon,                                                # 9
        phi,                                                    # 10
        ell.X0   / ell.a,                                       # 11
        ell.Y0   / ell.b,                                       # 12
        ell.X0_in / ell.a,                                      # 13
        ell.Y0_in / ell.b,                                      # 14
        np.sqrt((ell.X0   / ell.a)**2 + (ell.Y0   / ell.b)**2),# 15
        np.sqrt((ell.X0_in / ell.a)**2 + (ell.Y0_in / ell.b)**2),# 16
        ell.a * 3.0 / ell.b / 5.0,                             # 17
        ell.phi,                                                # 18
        np.real(quar_mean),                                     # 19
        np.imag(quar_mean),                                     # 20
        np.abs(quar_mean),                                      # 21
        sig_re_mean,                                            # 22
        sig_im_mean,                                            # 23
        np.abs(sig_re_mean + 1j * sig_im_mean),                 # 24
    ], dtype=float)

    if len(fingerprint_vec) != 25:
        flag = False
        
    print(fingerprint_vec)

    if error < 0.45 and flag:
        fingerprint = fingerprint_vec

    return fingerprint, bits