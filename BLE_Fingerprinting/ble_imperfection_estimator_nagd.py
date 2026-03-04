"""
Nesterov Accelerated Gradient Descent estimator for BLE hardware imperfections.

Port of BLE_Imperfection_Estimator_NAGD.m
Source: "Evaluating Physical-Layer BLE Location Tracking Attacks on Mobile Devices"
        IEEE Symposium on Security and Privacy 2022
"""
import numpy as np

from .gfsk_modulate import gfsk_modulate
from .fit_ellipse import fit_ellipse


def ble_imperfection_estimator_nagd(
    x: np.ndarray,
    seq: np.ndarray,
    Fs: float,
    init_f0: float,
    init_e: float = 0.0,
    init_phi: float = 0.0,
    init_I: float = 0.0,
    init_Q: float = 0.0,
    init_amp: float = 1.0,
    snr_db: float = 40.0,
    n_partition: int = 250,
):
    """
    Estimate hardware imperfection parameters using NAGD.

    Parameters
    ----------
    x           : complex received signal (1-D ndarray)
    seq         : decoded bit sequence (1-D int/bool ndarray)
    Fs          : sampling frequency (Hz)
    init_f0     : initial CFO estimate (Hz)
    init_e      : initial I/Q gain imbalance
    init_phi    : initial I/Q phase imbalance (degrees)
    init_I      : initial I DC offset
    init_Q      : initial Q DC offset
    init_amp    : initial amplitude estimate
    snr_db      : signal-to-noise ratio (dB)
    n_partition : number of random partitions for robust estimation

    Returns
    -------
    amp, e, phi, I, Q, IQO, IQI, f0, phi_off, error
        amp     – estimated amplitude (average over partitions)
        e       – I/Q gain imbalance (average)
        phi     – I/Q phase imbalance in degrees (average)
        I, Q    – normalised DC offsets (average I2/n, average Q2/n, normalised
                  by the last partition's amplitude, matching original MATLAB)
        IQO     – IQ offset magnitude (average)
        IQI     – IQ imbalance ratio (average)
        f0      – residual frequency offset Hz (average)
        phi_off – phase offset in degrees (average)
        error   – worst-case NMSE across all partitions (scalar float)
    """
    snr_lin = 10.0 ** (snr_db / 20.0)
    err_thresh = max(0.4, 1.0 / (snr_lin + 1.0))

    nsample = int(Fs / 1e6)

    # Remove last 2 µs from received signal
    x2 = x[: len(x) - 2 * nsample]

    # Time vector – MATLAB: t2 = (1:length(x)) * (1/Fs)  (1-indexed)
    t2 = np.arange(1, len(x) + 1, dtype=np.float64) / Fs

    # Generate ideal reference GFSK signal prepended with [1, 0]
    seq_full = np.concatenate([[1, 0], np.asarray(seq, dtype=float)])
    ref_full = gfsk_modulate(seq_full, 500e3, Fs)

    if n_partition != 1:
        # MATLAB: est_signal_perfect2(3.5*Fs/1e6+1 : end-0.5*Fs/1e6)
        # 1-indexed start 3.5*nsample+1 → 0-indexed start = int(3.5*nsample)
        ref_start = int(3.5 * nsample)
        ref_end   = len(ref_full) - int(0.5 * nsample)
    else:
        ref_start = int(3.0 * nsample)
        ref_end   = len(ref_full) - int(1.0 * nsample)

    est_signal_perfect2 = ref_full[ref_start:ref_end]

    # Partition length
    n       = min(10, n_partition)
    min_len = min(len(est_signal_perfect2), len(x2))
    l       = max(1, int(min_len / n_partition))

    # Accumulators across partitions
    I2 = Q2 = IQO2 = IQI2 = e2 = phi2 = phi_off2 = f02 = amp2 = 0.0
    err = 0.0

    # `f0` carries the latest frequency estimate between partitions
    f0  = init_f0
    amp = init_amp     # will be overwritten; kept here to mirror MATLAB variable scope

    for inter in range(n):
        # Random subset of indices shared by ref signal, time vector, and data
        r = np.random.permutation(min_len)[:l]

        esp    = est_signal_perfect2[r]
        esp_re = np.real(esp)
        esp_im = np.imag(esp)
        t      = t2[r]
        xr     = x2[r]
        xr_re  = np.real(xr)
        xr_im  = np.imag(xr)

        # ---------- Initialisation ------------------------------------ #
        e_new       = float(init_e)
        phi_new     = float(init_phi) / 180.0 * np.pi
        I_new       = float(init_I)
        Q_new       = float(init_Q)
        if inter == 0:
            f0_new  = float(init_f0)
        else:
            f0_new  = f0
            init_f0 = f0
        w0_new      = 2.0 * np.pi * f0_new
        phi_off_new = 36.0 / 360.0 * 2.0 * np.pi   # initial phase offset (radians)
        amp_new     = float(init_amp)

        # Working copies (Nesterov lookahead will modify these)
        e       = e_new
        phi     = phi_new
        I       = I_new
        Q       = Q_new
        w0      = w0_new
        phi_off = phi_off_new
        amp     = amp_new

        # Momentum buffers
        e_t = phi_t = I_t = Q_t = w0_t = phi_off_t = amp_t = 0.0

        error_diff  = 1.0
        count       = 0
        rnd         = 1
        error_list  = [1.0]

        lr  = 1e-3
        mom = 0.9

        # ---------- Gradient-descent inner loop ----------------------- #
        while error_diff > 1e-7 and count < 10000:
            count += 1

            # Nesterov lookahead (e and amp skip this step, matching MATLAB)
            phi     = phi_new     - mom * phi_t
            I       = I_new       - mom * I_t
            Q       = Q_new       - mom * Q_t
            w0      = w0_new      - mom * w0_t
            phi_off = phi_off_new - mom * phi_off_t
            # e   = e_new   (commented out in original)
            # amp = amp_new (commented out in original)

            # Re-initialise CFO if convergence is stalled
            if count > rnd * 200 and error_list[-1] > err_thresh:
                if (rnd // 2) * 2 == rnd - 1:
                    rnd    += 1
                    f0      = init_f0 - (rnd // 2) * 1.5e3
                    w0_new  = 2.0 * np.pi * f0
                else:
                    rnd    += 1
                    f0      = init_f0 + (rnd / 2.0) * 1.5e3
                    w0_new  = 2.0 * np.pi * f0

            # Pre-compute shared trig / rotation terms
            phase     = w0 * t + phi_off
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            cos_phi   = np.cos(phi)
            sin_phi   = np.sin(phi)

            rot_re = esp_re * cos_phi - esp_im * sin_phi   # IQ-rotated real part
            rot_im = esp_im * cos_phi + esp_re * sin_phi   # IQ-rotated imag part

            A_re = (amp - e) * rot_re + I
            B_im = (amp + e) * rot_im + Q

            Real_part = A_re * cos_phase - B_im * sin_phase
            Imag_part = A_re * sin_phase + B_im * cos_phase

            err_re = xr_re - Real_part
            err_im = xr_im - Imag_part

            # ---- Gradients from imaginary error ---- #
            e_d = -np.mean(err_im * (-rot_re * sin_phase + rot_im * cos_phase))

            phi_d = -np.mean(err_im * (
                (amp - e) * (-esp_re * sin_phi + esp_im * cos_phi) * sin_phase
                + (amp + e) * ( esp_re * cos_phi - esp_im * sin_phi) * cos_phase
            ))

            I_d       = -np.mean(err_im * sin_phase)
            Q_d       = -np.mean(err_im * cos_phase)

            w0_d = -np.mean(err_im * (
                t * A_re * cos_phase - t * B_im * sin_phase
            ))

            phi_off_d = -np.mean(err_im * (
                A_re * cos_phase - B_im * sin_phase
            ))

            amp_d = -np.mean(err_im * (
                (rot_re + I) * sin_phase + (rot_im + Q) * cos_phase
            ))

            # ---- Add gradients from real error ---- #
            e_d -= np.mean(err_re * (-rot_re * cos_phase - rot_im * sin_phase))

            phi_d -= np.mean(err_re * (
                (amp - e) * (-esp_re * sin_phi + esp_im * cos_phi) * cos_phase
                - (amp + e) * ( esp_re * cos_phi - esp_im * sin_phi) * sin_phase
            ))

            I_d       -= np.mean(err_re *  cos_phase)
            Q_d       += np.mean(err_re *  sin_phase)

            w0_d -= np.mean(err_re * (
                -t * A_re * sin_phase - t * B_im * cos_phase
            ))

            phi_off_d -= np.mean(err_re * (
                -A_re * sin_phase - B_im * cos_phase
            ))

            amp_d -= np.mean(err_re * (
                (rot_re + I) * cos_phase - (rot_im + Q) * sin_phase
            ))

            # ---- Momentum update ------------------------------------ #
            e_t       = mom * e_t       + lr        * e_d
            phi_t     = mom * phi_t     + lr        * phi_d
            I_t       = mom * I_t       + lr        * I_d
            Q_t       = mom * Q_t       + lr        * Q_d
            w0_t      = mom * w0_t      + 1e8 * lr  * w0_d
            phi_off_t = mom * phi_off_t + 10.0 * lr * phi_off_d
            amp_t     = mom * amp_t     + lr        * amp_d

            e_new       -= e_t
            phi_new     -= phi_t
            I_new       -= I_t
            Q_new       -= Q_t
            w0_new      -= w0_t
            phi_off_new -= phi_off_t
            amp_new     -= amp_t

            # Sync working copies from updated _new values
            e       = e_new
            phi     = phi_new
            I       = I_new
            Q       = Q_new
            w0      = w0_new
            phi_off = phi_off_new
            amp     = amp_new

            # --- NMSE error metric ----------------------------------- #
            phase2     = w0 * t + phi_off
            cos_phase2 = np.cos(phase2)
            sin_phase2 = np.sin(phase2)
            cos_phi2   = np.cos(phi)
            sin_phi2   = np.sin(phi)
            rot_re2    = esp_re * cos_phi2 - esp_im * sin_phi2
            rot_im2    = esp_im * cos_phi2 + esp_re * sin_phi2
            A2 = (amp - e) * rot_re2 + I
            B2 = (amp + e) * rot_im2 + Q
            est_re2 = A2 * cos_phase2 - B2 * sin_phase2
            est_im2 = A2 * sin_phase2 + B2 * cos_phase2

            diff_sq = (est_re2 - xr_re)**2 + (est_im2 - xr_im)**2
            denom   = xr_re**2 + xr_im**2
            nmse    = np.sum(diff_sq) / np.sum(denom) / 2.0
            error_list.append(nmse)

            if len(error_list) > 1 and error_list[-1] < err_thresh:
                error_diff = abs(error_list[-1] - error_list[-2])

        # Worst-case error
        err = max(err, error_list[-1])

        # Apply correction to full x2 and fit ellipse
        tt        = t2[: len(x2)]
        corrected = x2 * np.exp(-1j * (w0 * tt + phi_off))

        try:
            ell  = fit_ellipse(np.real(corrected), 3.0 * np.imag(corrected))
            IQO  = np.sqrt((ell.X0 / ell.a)**2 + (ell.Y0 / ell.b)**2)
            IQI  = ell.a / ell.b * 3.0
            flag = True
        except (ValueError, ZeroDivisionError):
            flag = False

        if flag:
            # Convert frequency / phase to output units
            f0_part      = w0 / (2.0 * np.pi)
            phi_off_part = phi_off * (360.0 / (2.0 * np.pi))   # radians → degrees
            phi_part     = phi     * (360.0 / (2.0 * np.pi))   # radians → degrees

            # Note: `amp` here is still the gradient-descent final value (in radians
            # domain), matching the MATLAB variable at this point in the loop.
            f02      += f0_part
            phi_off2 += phi_off_part
            phi2     += phi_part
            I2       += I
            Q2       += Q
            IQO2     += IQO
            IQI2     += IQI
            e2       += e
            amp2     += amp

    # `amp` here is the last partition's amplitude (pre-average), matching MATLAB
    # which normalises I and Q by the post-loop value of `amp` before computing
    # `amp = amp2/n`.
    last_amp = amp   # = amp_new from last gradient-descent iteration

    I_avg = I2 / n
    Q_avg = Q2 / n
    I_out = -I_avg / last_amp if last_amp != 0.0 else 0.0
    Q_out =  Q_avg / last_amp if last_amp != 0.0 else 0.0

    IQO     = IQO2     / n
    IQI     = IQI2     / n
    phi_out = phi2     / n
    e_out   = e2       / n
    phi_off = phi_off2 / n
    f0      = f02      / n
    amp_out = amp2     / n
    error   = err

    return amp_out, e_out, phi_out, I_out, Q_out, IQO, IQI, f0, phi_off, error
