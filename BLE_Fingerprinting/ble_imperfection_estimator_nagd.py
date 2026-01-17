import math

import numpy as np

from fit_ellipse import fit_ellipse
from gfsk_modulate import gfsk_modulate


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

    min_len = min(len(est_signal_perfect2), len(x2), len(t2))
    est_signal_perfect2 = est_signal_perfect2[:min_len]
    x2 = x2[:min_len]
    t2 = t2[:min_len]

    fr = []
    I2 = Q2 = IQO2 = IQI2 = e2 = phi2 = phi_off2 = f02 = amp2 = 0.0
    error2 = 0.0
    err = 0.0

    n = min(10, n_partition)
    l = int(len(x2) / n_partition) if n_partition else len(x2)

    for inter in range(n):
        r = np.random.permutation(min_len)[:l]
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
