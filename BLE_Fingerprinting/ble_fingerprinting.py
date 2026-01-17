import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EllipseFit:
    a: float
    b: float
    phi: float
    X0: float
    Y0: float
    X0_in: float
    Y0_in: float
    long_axis: float
    short_axis: float
    status: str


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
    gamma_gfsk = np.convolve(gamma_fsk, gauss_filter, mode="same")
    gfsk_phase = (freqsep / fs) * math.pi * _cumulative_trapezoid(gamma_gfsk)
    return np.exp(1j * gfsk_phase)


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> EllipseFit:
    """Port of the MATLAB fit_ellipse utility from MathWorks File Exchange."""
    orientation_tolerance = 1e-7

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    X = np.column_stack((x**2, x * y, y**2, x, y))
    a_vec = np.sum(X, axis=0) @ np.linalg.inv(X.T @ X)

    a, b, c, d, e = a_vec

    if min(abs(b / a), abs(b / c)) > orientation_tolerance:
        orientation_rad = 0.5 * math.atan2(b, (c - a))
        cos_phi = math.cos(orientation_rad)
        sin_phi = math.sin(orientation_rad)
        a, b, c, d, e = (
            a * cos_phi**2 - b * cos_phi * sin_phi + c * sin_phi**2,
            0.0,
            a * sin_phi**2 + b * cos_phi * sin_phi + c * cos_phi**2,
            d * cos_phi - e * sin_phi,
            d * sin_phi + e * cos_phi,
        )
        mean_x, mean_y = (
            cos_phi * mean_x - sin_phi * mean_y,
            sin_phi * mean_x + cos_phi * mean_y,
        )
    else:
        orientation_rad = 0.0
        cos_phi = math.cos(orientation_rad)
        sin_phi = math.sin(orientation_rad)

    test = a * c
    if test <= 0:
        status = "Parabola found" if test == 0 else "Hyperbola found"
        return EllipseFit(
            a=np.nan,
            b=np.nan,
            phi=np.nan,
            X0=np.nan,
            Y0=np.nan,
            X0_in=np.nan,
            Y0_in=np.nan,
            long_axis=np.nan,
            short_axis=np.nan,
            status=status,
        )

    if a < 0:
        a, c, d, e = -a, -c, -d, -e

    X0 = -d / (2 * a)
    Y0 = -e / (2 * c)
    F = 1 + (d**2) / (4 * a) + (e**2) / (4 * c)
    a_axis = math.sqrt(abs(F / a))
    b_axis = math.sqrt(abs(F / c))
    long_axis = 2 * max(a_axis, b_axis)
    short_axis = 2 * min(a_axis, b_axis)

    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    X0_in, Y0_in = (R @ np.array([X0, Y0])).tolist()

    return EllipseFit(
        a=a_axis,
        b=b_axis,
        phi=orientation_rad,
        X0=X0,
        Y0=Y0,
        X0_in=X0_in,
        Y0_in=Y0_in,
        long_axis=long_axis,
        short_axis=short_axis,
        status="",
    )


def ble_decoder(signal: np.ndarray, fs: float, preamble_detect: int):
    pream = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    preamble_signal = gfsk_modulate(pream, 500e3, fs)
    sps = int(fs / 1e6)
    preamble_signal = preamble_signal[int(2.5 * sps) : -int(0.5 * sps)]

    preamble_angle = np.unwrap(np.angle(preamble_signal))
    preamble_freq = np.diff(preamble_angle) / (2 * math.pi) * fs

    signal_angle = np.unwrap(np.angle(signal))
    slope = signal_angle[2:] - signal_angle[1:-1]
    signal_freq = slope / (2 * math.pi) * fs
    signal_freq = np.concatenate([signal_freq, [0.0]])

    if preamble_detect == 0:
        start_ind = 0
    else:
        l = len(signal_freq)
        z = np.correlate(signal_freq, preamble_freq, mode="full")
        z = z[l:]
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


def ble_imperfection_estimator_nagd(
    x,
    seq,
    fs,
    init_f0,
    init_e,
    init_phi,
    init_I,
    init_Q,
    init_amp,
    snr,
    n_partition,
):
    snr_lin = 10 ** (snr / 20)
    err_thresh = max(0.4, 1 / (snr_lin + 1))

    sps = int(fs / 1e6)
    x2 = x[: -2 * sps]
    t2 = np.arange(len(x)) / fs

    seq = np.concatenate([[1, 0], np.asarray(seq).astype(int)])
    est_signal_perfect2 = gfsk_modulate(seq, 500e3, fs)

    if n_partition != 1:
        start = int(3.5 * sps)
        end = len(est_signal_perfect2) - int(0.5 * sps)
        est_signal_perfect2 = est_signal_perfect2[start:end]
    else:
        start = int(3.0 * sps)
        end = len(est_signal_perfect2) - int(1 * sps)
        est_signal_perfect2 = est_signal_perfect2[start:end]

    fr = []
    I2 = Q2 = IQO2 = IQI2 = e2 = phi2 = phi_off2 = f02 = amp2 = 0.0
    error2 = 0.0
    err = 0.0

    n = min(10, n_partition)
    l = int(len(x2) / n_partition)

    for inter in range(n):
        r = np.random.permutation(len(est_signal_perfect2))[:l]
        est_signal_perfect = est_signal_perfect2[r]
        t = t2[r]
        x_part = x2[r]

        e_new = init_e
        phi_new = math.radians(init_phi)
        I_new = init_I
        Q_new = init_Q

        if inter == 0:
            f0_new = init_f0
        else:
            f0_new = f0
            init_f0 = f0
        w0_new = 2 * math.pi * f0_new
        phi_off_new = 36 / 360 * 2 * math.pi
        amp_new = init_amp

        e = e_new
        phi = phi_new
        I = I_new
        Q = Q_new
        w0 = w0_new
        phi_off = phi_off_new
        amp = amp_new

        est_signal = (
            (amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi))
            + 1j
            * (amp + e)
            * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
            + I
            + 1j * Q
        ) * np.exp(1j * (w0 * t + phi_off))

        e_t = phi_t = I_t = Q_t = w0_t = phi_off_t = amp_t = 0.0
        error_diff = 1.0
        count = 0
        round_count = 1
        error = [1.0]

        while error_diff > 1e-7 and count < 10_000:
            count += 1

            if error_diff > 1e-7:
                lr = 1e-3
                mom = 0.9
            else:
                lr = 1e-4
                mom = 0.9

            phi = phi_new - mom * phi_t
            I = I_new - mom * I_t
            Q = Q_new - mom * Q_t
            w0 = w0_new - mom * w0_t
            phi_off = phi_off_new - mom * phi_off_t

            if count > round_count * 2e2 and error[-1] > err_thresh:
                if math.floor(round_count / 2) * 2 == round_count - 1:
                    round_count += 1
                    f0 = init_f0 - math.floor(round_count / 2) * 1.5e3
                    w0_new = 2 * math.pi * f0
                else:
                    round_count += 1
                    f0 = init_f0 + (round_count / 2) * 1.5e3
                    w0_new = 2 * math.pi * f0

            imag_part = (
                ((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                * np.sin(w0 * t + phi_off)
                + (
                    (amp + e) * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                    + Q
                )
                * np.cos(w0 * t + phi_off)
            )

            real_part = (
                ((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                * np.cos(w0 * t + phi_off)
                - (
                    (amp + e) * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                    + Q
                )
                * np.sin(w0 * t + phi_off)
            )

            imag_residual = np.imag(x_part) - imag_part
            real_residual = np.real(x_part) - real_part

            e_d = -np.mean(
                imag_residual
                * (
                    (-(np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)))
                    * np.sin(w0 * t + phi_off)
                    + (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                    * np.cos(w0 * t + phi_off)
                )
            )
            phi_d = -np.mean(
                imag_residual
                * (
                    (amp - e)
                    * (-np.real(est_signal_perfect) * np.sin(phi) + np.imag(est_signal_perfect) * np.cos(phi))
                    * np.sin(w0 * t + phi_off)
                    + (amp + e)
                    * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi))
                    * np.cos(w0 * t + phi_off)
                )
            )
            I_d = -np.mean(imag_residual * np.sin(w0 * t + phi_off))
            Q_d = -np.mean(imag_residual * np.cos(w0 * t + phi_off))
            w0_d = -np.mean(
                imag_residual
                * (
                    t
                    * ((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                    * np.cos(w0 * t + phi_off)
                    - t
                    * ((amp + e) * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi)) + Q)
                    * np.sin(w0 * t + phi_off)
                )
            )
            phi_off_d = -np.mean(
                imag_residual
                * (
                    ((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                    * np.cos(w0 * t + phi_off)
                    - (
                        (amp + e)
                        * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                        + Q
                    )
                    * np.sin(w0 * t + phi_off)
                )
            )
            amp_d = -np.mean(
                imag_residual
                * (
                    (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi) + I)
                    * np.sin(w0 * t + phi_off)
                    + (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi) + Q)
                    * np.cos(w0 * t + phi_off)
                )
            )

            e_d = e_d - np.mean(
                real_residual
                * (
                    (-(np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)))
                    * np.cos(w0 * t + phi_off)
                    - (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                    * np.sin(w0 * t + phi_off)
                )
            )
            phi_d = phi_d - np.mean(
                real_residual
                * (
                    (amp - e)
                    * (-np.real(est_signal_perfect) * np.sin(phi) + np.imag(est_signal_perfect) * np.cos(phi))
                    * np.cos(w0 * t + phi_off)
                    - (amp + e)
                    * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi))
                    * np.sin(w0 * t + phi_off)
                )
            )
            I_d = I_d - np.mean(real_residual * np.cos(w0 * t + phi_off))
            Q_d = Q_d + np.mean(real_residual * np.sin(w0 * t + phi_off))
            w0_d = w0_d - np.mean(
                real_residual
                * (
                    -t
                    * ((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                    * np.sin(w0 * t + phi_off)
                    - t
                    * ((amp + e) * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi)) + Q)
                    * np.cos(w0 * t + phi_off)
                )
            )
            phi_off_d = phi_off_d - np.mean(
                real_residual
                * (
                    -((amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi)) + I)
                    * np.sin(w0 * t + phi_off)
                    - (
                        (amp + e)
                        * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                        + Q
                    )
                    * np.cos(w0 * t + phi_off)
                )
            )
            amp_d = amp_d - np.mean(
                real_residual
                * (
                    (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi) + I)
                    * np.cos(w0 * t + phi_off)
                    - (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi) + Q)
                    * np.sin(w0 * t + phi_off)
                )
            )

            if error_diff > 1e-7:
                e_t = mom * e_t + lr * e_d
                phi_t = mom * phi_t + lr * phi_d
                I_t = mom * I_t + lr * I_d
                Q_t = mom * Q_t + lr * Q_d
                w0_t = mom * w0_t + 1e8 * lr * w0_d
                phi_off_t = mom * phi_off_t + 10 * lr * phi_off_d
                amp_t = mom * amp_t + lr * amp_d
            else:
                e_t = mom * e_t + lr * e_d
                phi_t = mom * phi_t + lr * phi_d
                I_t = mom * I_t + lr * I_d
                Q_t = mom * Q_t + lr * Q_d
                phi_off_t = mom * phi_off_t + 10 * lr * phi_off_d
                amp_t = mom * amp_t + lr * amp_d

            e_new = e_new - e_t
            phi_new = phi_new - phi_t
            I_new = I_new - I_t
            Q_new = Q_new - Q_t
            w0_new = w0_new - w0_t
            phi_off_new = phi_off_new - phi_off_t
            amp_new = amp_new - amp_t

            e = e_new
            phi = phi_new
            I = I_new
            Q = Q_new
            w0 = w0_new
            phi_off = phi_off_new
            amp = amp_new

            est_signal = (
                (amp - e) * (np.real(est_signal_perfect) * np.cos(phi) - np.imag(est_signal_perfect) * np.sin(phi))
                + 1j
                * (amp + e)
                * (np.imag(est_signal_perfect) * np.cos(phi) + np.real(est_signal_perfect) * np.sin(phi))
                + I
                + 1j * Q
            ) * np.exp(1j * (w0 * t + phi_off))

            error_value = 0.5 * (np.linalg.norm(est_signal - x_part) ** 2) / (np.linalg.norm(x_part) ** 2)
            error.append(error_value)

            if len(error) > 1 and error[-1] < err_thresh:
                error_diff = abs(error[-1] - error[-2])

        err = max(err, error[-1])

        signal = x_part * np.exp(-1j * (w0 * t + phi_off))

        try:
            ell = fit_ellipse(np.real(signal), 3 * np.imag(signal))
            if math.isnan(ell.a) or math.isnan(ell.b):
                raise ValueError("Ellipse fit failed")
            IQO = math.sqrt((ell.X0 / ell.a) ** 2 + (ell.Y0 / ell.b) ** 2)
            IQI = ell.a / ell.b * 3
            flag = True
        except Exception:
            flag = False

        if flag:
            f0 = w0 / (2 * math.pi)
            phi_off_deg = phi_off * (360 / (2 * math.pi))
            phi_deg = phi * (360 / (2 * math.pi))
            fr.append(f0)
            I2 += I
            Q2 += Q
            IQO2 += IQO
            IQI2 += IQI
            e2 += e
            phi_off2 += phi_off_deg
            phi2 += phi_deg
            f02 += f0
            amp2 += amp
            error2 += err

    I = I2 / n
    I = -I / amp
    Q = Q2 / n
    Q = Q / amp
    IQO = IQO2 / n
    IQI = IQI2 / n
    phi = phi2 / n
    e = e2 / n
    phi_off = phi_off2 / n
    f0 = f02 / n
    amp = amp2 / n
    error = err
    return amp, e, phi, I, Q, IQO, IQI, f0, phi_off, error, fr


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


def run_example(
    data_dir: str,
    fs: float = 3.125e6,
    snr: float = 40,
    preamble_detect: int = 1,
    interp_fac: int = 32,
    n_partition: int = 250,
    fingerprint_size: int = 25,
):
    fingerprint_all = np.zeros((20, fingerprint_size))
    for i in range(1, 21):
        sample_path = f"{data_dir}/{i}"
        signal = np.fromfile(sample_path, dtype=np.float32)
        signal = signal.reshape(-1, 2)
        signal = signal[:, 0] + 1j * signal[:, 1]
        signal = signal[:-12]

        fingerprint, _ = ble_fingerprint(
            signal,
            snr,
            fs,
            preamble_detect,
            interp_fac,
            n_partition,
        )
        if fingerprint is not None:
            fingerprint_all[i - 1, :] = fingerprint

    return fingerprint_all
