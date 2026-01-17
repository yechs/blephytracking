import math

import numpy as np

from ble_decoder import ble_decoder
from ble_imperfection_estimator_nagd import ble_imperfection_estimator_nagd
from fit_ellipse import fit_ellipse


def _interp_signal(signal: np.ndarray, interp_fac: int) -> np.ndarray:
    if interp_fac == 1:
        return signal
    x = np.arange(len(signal))
    x_new = np.linspace(0, len(signal) - 1, len(signal) * interp_fac)
    real_interp = np.interp(x_new, x, np.real(signal))
    imag_interp = np.interp(x_new, x, np.imag(signal))
    return real_interp + 1j * imag_interp


def ble_fingerprint(signal, snr, fs, preamble_detect, interp_fac, n_partition):
    fc = 2.48e9
    ch = 2.48e9

    fs = fs * interp_fac

    signal = signal / np.mean(np.abs(signal))
    signal = _interp_signal(signal, interp_fac)

    if interp_fac != 1:
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        ls = len(signal)

        signal_fft_centered = np.zeros(ls, dtype=complex)
        lcenter = ls // 2
        lchannel = int((ch - fc) / fs * ls + ls / 2)
        lbandwidth = int(2 / (fs / 1e6) * (ls - 1) / 2)
        signal_fft_centered[lcenter - lbandwidth : lcenter + lbandwidth + 1] = signal_fft[
            lchannel - lbandwidth : lchannel + lbandwidth + 1
        ]
        signal = np.fft.ifft(np.fft.ifftshift(signal_fft_centered))

    signal, signal_freq, bits = ble_decoder(signal, fs, preamble_detect)

    pream_samples = int(fs / 1e6 * 8 - 1)
    pream = signal_freq[:pream_samples]
    est_cfo = np.mean(pream)
    est_cfo2 = est_cfo if abs(est_cfo) <= 100e3 else 0

    (
        amp,
        epsilon,
        phi,
        I,
        Q,
        IQO,
        IQI,
        f0,
        phi_off,
        error,
        _,
    ) = ble_imperfection_estimator_nagd(
        signal,
        bits,
        fs,
        est_cfo2,
        0,
        0,
        0,
        0,
        1,
        snr,
        n_partition,
    )

    tt = np.arange(len(signal)) / fs
    signal = signal * np.exp(-1j * (2 * math.pi * f0 * tt + phi_off / (360 / (2 * math.pi))))

    flag = True
    try:
        sig = np.random.permutation(signal)
        ell = fit_ellipse(-np.real(sig) / amp * 5, 3 * np.imag(sig) / amp)
        if math.isnan(ell.a) or math.isnan(ell.b):
            raise ValueError("Ellipse fit failed")
    except Exception:
        flag = False
        ell = None

    angsig = np.angle(signal)
    spl = 8
    quar = np.zeros(spl, dtype=complex)
    for sp in range(1, spl // 2 + 1):
        e1 = (angsig > (sp - 1) * 2 * math.pi / spl) & (angsig < sp * 2 * math.pi / spl)
        quar[sp - 1] = np.mean(signal[e1])
        e2 = (angsig > (-math.pi + (sp - 1) * 2 * math.pi / spl)) & (
            angsig < (-math.pi + sp * 2 * math.pi / spl)
        )
        quar[sp - 1 + spl // 2] = np.mean(signal[e2])

    if not flag:
        return None, bits

    fingerprint_vec = np.array(
        [
            error,
            amp,
            f0,
            est_cfo,
            IQO,
            I,
            Q,
            math.sqrt(I**2 + Q**2),
            IQI,
            epsilon,
            phi,
            ell.X0 / ell.a,
            ell.Y0 / ell.b,
            ell.X0_in / ell.a,
            ell.Y0_in / ell.b,
            math.sqrt((ell.X0 / ell.a) ** 2 + (ell.Y0 / ell.b) ** 2),
            math.sqrt((ell.X0_in / ell.a) ** 2 + (ell.Y0_in / ell.b) ** 2),
            ell.a * 3 / ell.b / 5,
            ell.phi,
            np.real(np.mean(quar)),
            np.imag(np.mean(quar)),
            np.abs(np.mean(quar)),
            np.mean(np.real(signal)),
            np.mean(np.imag(signal)),
            np.abs(np.mean(np.real(signal)) + 1j * np.mean(np.imag(signal))),
        ]
    )

    if len(fingerprint_vec) != 25:
        return None, bits

    if error < 0.45:
        return fingerprint_vec, bits

    return None, bits
