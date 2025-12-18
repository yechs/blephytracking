import numpy as np
import warnings
from typing import Tuple, List, Optional, Any, Union

# Import dependent modules (assuming they are in the same directory)
from gfsk_modulate import gfsk_modulate
from fit_ellipse import fit_ellipse


def BLE_Imperfection_Estimator_NAGD(
    x: np.ndarray,
    seq: np.ndarray,
    Fs: float,
    init_f0: float,
    init_e: float,
    init_phi: float,
    init_I: float,
    init_Q: float,
    init_amp: float,
    snr: float,
    n_partition: int,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    List[float],
    List[Any],
]:
    """
    Estimates hardware imperfection fingerprints using Nesterov Accelerated Gradient Descent.

    Parameters:
    - x: Received signal (1D complex array)
    - seq: Decoded bit sequence (0s and 1s)
    - Fs: Sampling rate (Hz)
    - init_*: Initial guesses for parameters
    - snr: Signal-to-Noise ratio (dB)
    - n_partition: Number of partitions for averaging

    Returns:
    - Tuple containing estimated parameters:
        (amp, e, phi, I, Q, IQO, IQI, f0, phi_off, error, est_signal, fr, e_o)
    """

    # SNR conversion
    snr_lin: float = 10 ** (snr / 20.0)
    err_thresh: float = max(0.4, 1.0 / (snr_lin + 1))

    # Preprocess input signal
    end_cut: int = int(2 * Fs / 1e6)
    x2: np.ndarray = x[:-end_cut] if end_cut > 0 else x

    # Time vector
    t2: np.ndarray = (np.arange(1, len(x) + 1)) * (1.0 / Fs)

    # =========================================================================
    # Generate Perfect Reference Signal
    # =========================================================================
    seq_extended: np.ndarray = np.concatenate(([1, 0], seq))
    est_signal_perfect2: np.ndarray = gfsk_modulate(seq_extended, 500e3, Fs)

    # Trim perfect signal
    nsample: int = int(Fs / 1e6)
    start_idx: int
    end_idx: int

    if n_partition != 1:
        start_idx = int(3.5 * nsample)
        end_idx = int(0.5 * nsample)
        est_signal_perfect2 = est_signal_perfect2[start_idx:-end_idx]
    else:
        start_idx = int(3.0 * nsample)
        end_idx = int(1.0 * nsample)
        est_signal_perfect2 = est_signal_perfect2[start_idx:-end_idx]

    # Accumulators
    fr_list: List[float] = []
    I2: float = 0.0
    Q2: float = 0.0
    IQO2: float = 0.0
    IQI2: float = 0.0
    e2: float = 0.0
    phi2: float = 0.0
    phi_off2: float = 0.0
    f02: float = 0.0
    amp2: float = 0.0
    error2: float = 0.0
    est_signal: np.ndarray = np.array([], dtype=complex)  # Placeholder for last signal

    # Loop parameters
    n_iters: int = min(10, n_partition)
    l_part: int = len(x2) // n_partition

    # Main Partition Loop
    for inter in range(n_iters):
        # Random partition
        valid_len: int = min(len(est_signal_perfect2), len(x2))
        r_indices: np.ndarray = np.random.permutation(valid_len)
        r_indices = r_indices[:l_part]

        est_signal_perfect: np.ndarray = est_signal_perfect2[r_indices]
        t: np.ndarray = t2[r_indices]
        x_curr: np.ndarray = x2[r_indices]

        # Initialization
        e_new: float = init_e
        phi_new: float = init_phi / 180.0 * np.pi
        I_new: float = init_I
        Q_new: float = init_Q

        f0_val: float
        if inter == 0:
            f0_new = init_f0
            f0_val = init_f0
        else:
            # f0_val comes from the previous iteration's success or accumulator average
            # Ideally we keep track of the last successful f0.
            # The MATLAB code uses f0 from previous loop scope.
            f0_new = f0_val
            init_f0 = f0_val

        w0_new: float = 2 * np.pi * f0_new
        phi_off_new: float = 36.0 / 360.0 * 2 * np.pi
        amp_new: float = init_amp

        # Current params
        e, phi, I_val, Q_val = e_new, phi_new, I_new, Q_new
        w0, phi_off, amp = w0_new, phi_off_new, amp_new

        # Momentum variables
        e_t: float = 0.0
        phi_t: float = 0.0
        I_t: float = 0.0
        Q_t: float = 0.0
        w0_t: float = 0.0
        phi_off_t: float = 0.0
        amp_t: float = 0.0

        # Pre-calculate complex components
        esp_real: np.ndarray = np.real(est_signal_perfect)
        esp_imag: np.ndarray = np.imag(est_signal_perfect)

        error_hist: List[float] = []
        error_diff: float = 1.0
        count: int = 0
        round_counter: int = 1

        # =====================================================================
        # Gradient Descent Loop
        # =====================================================================
        while error_diff > 1e-7 and count < 10000:
            count += 1

            lr: float
            mom: float
            if error_diff > 1e-7:
                lr, mom = 1e-3, 0.9
            elif error_diff < 1e-7:
                lr, mom = 1e-4, 0.9
            else:
                lr, mom = 1e-3, 0.9

            # Momentum Application
            phi = phi_new - mom * phi_t
            I_val = I_new - mom * I_t
            Q_val = Q_new - mom * Q_t
            w0 = w0_new - mom * w0_t
            phi_off = phi_off_new - mom * phi_off_t

            # Helper Terms
            cos_phi: float = np.cos(phi)
            sin_phi: float = np.sin(phi)

            term1: np.ndarray = esp_real * cos_phi - esp_imag * sin_phi
            term2: np.ndarray = esp_imag * cos_phi + esp_real * sin_phi

            arg: np.ndarray = w0 * t + phi_off
            sin_arg: np.ndarray = np.sin(arg)
            cos_arg: np.ndarray = np.cos(arg)

            # Re-initialization logic
            current_err: float = error_hist[-1] if len(error_hist) > 0 else 1.0
            if count > round_counter * 200 and current_err > err_thresh:
                if (round_counter // 2) * 2 == round_counter - 1:
                    round_counter += 1
                    f0_val = init_f0 - (round_counter // 2) * 1.5e3
                else:
                    round_counter += 1
                    f0_val = init_f0 + (round_counter / 2) * 1.5e3
                w0_new = 2 * np.pi * f0_val
                w0 = w0_new

            # Construct Model Signal
            part_a: np.ndarray = (amp - e) * term1 + I_val
            part_b: np.ndarray = (amp + e) * term2 + Q_val

            Imag_part: np.ndarray = part_a * sin_arg + part_b * cos_arg
            Real_part: np.ndarray = part_a * cos_arg - part_b * sin_arg

            diff_imag: np.ndarray = np.imag(x_curr) - Imag_part
            diff_real: np.ndarray = np.real(x_curr) - Real_part

            # Gradients
            d_Imag_de: np.ndarray = -term1 * sin_arg + term2 * cos_arg
            d_Real_de: np.ndarray = -term1 * cos_arg - term2 * sin_arg
            e_d: float = -np.mean(diff_imag * d_Imag_de) - np.mean(
                diff_real * d_Real_de
            )

            d_term1_phi: np.ndarray = -term2
            d_term2_phi: np.ndarray = term1
            d_Imag_phi: np.ndarray = ((amp - e) * d_term1_phi) * sin_arg + (
                (amp + e) * d_term2_phi
            ) * cos_arg
            d_Real_phi: np.ndarray = ((amp - e) * d_term1_phi) * cos_arg - (
                (amp + e) * d_term2_phi
            ) * sin_arg
            phi_d: float = -np.mean(diff_imag * d_Imag_phi) - np.mean(
                diff_real * d_Real_phi
            )

            d_Imag_I: np.ndarray = sin_arg
            d_Real_I: np.ndarray = cos_arg
            I_d: float = -np.mean(diff_imag * d_Imag_I) - np.mean(diff_real * d_Real_I)

            d_Imag_Q: np.ndarray = cos_arg
            d_Real_Q: np.ndarray = -sin_arg
            Q_d: float = -np.mean(diff_imag * d_Imag_Q) - np.mean(diff_real * d_Real_Q)

            d_Imag_w0: np.ndarray = part_a * (t * cos_arg) - part_b * (t * sin_arg)
            d_Real_w0: np.ndarray = -part_a * (t * sin_arg) - part_b * (t * cos_arg)
            w0_d: float = -np.mean(diff_imag * d_Imag_w0) - np.mean(
                diff_real * d_Real_w0
            )

            d_Imag_phioff: np.ndarray = part_a * cos_arg - part_b * sin_arg
            d_Real_phioff: np.ndarray = -part_a * sin_arg - part_b * cos_arg
            phi_off_d: float = -np.mean(diff_imag * d_Imag_phioff) - np.mean(
                diff_real * d_Real_phioff
            )

            d_Imag_amp: np.ndarray = term1 * sin_arg + term2 * cos_arg
            d_Real_amp: np.ndarray = term1 * cos_arg - term2 * sin_arg
            amp_d: float = -np.mean(diff_imag * d_Imag_amp) - np.mean(
                diff_real * d_Real_amp
            )

            # Update Steps
            e_t = mom * e_t + lr * e_d
            phi_t = mom * phi_t + lr * phi_d
            I_t = mom * I_t + lr * I_d
            Q_t = mom * Q_t + lr * Q_d
            if error_diff > 1e-7:
                w0_t = mom * w0_t + 1e8 * lr * w0_d

            phi_off_t = mom * phi_off_t + 10 * lr * phi_off_d
            amp_t = mom * amp_t + lr * amp_d

            e_new = e_new - e_t
            phi_new = phi_new - phi_t
            I_new = I_new - I_t
            Q_new = Q_new - Q_t
            w0_new = w0_new - w0_t
            phi_off_new = phi_off_new - phi_off_t
            amp_new = amp_new - amp_t

            e, phi, I_val, Q_val = e_new, phi_new, I_new, Q_new
            w0, phi_off, amp = w0_new, phi_off_new, amp_new

            # Calculate Error
            est_signal = (
                (amp - e) * term1 + 1j * (amp + e) * term2 + I_val + 1j * Q_val
            ) * np.exp(1j * (w0 * t + phi_off))

            mse_num: float = np.mean(np.abs(est_signal - x_curr) ** 2)
            mse_den: float = np.mean(np.abs(x_curr) ** 2)
            curr_error: float = mse_num / mse_den / 2.0

            error_hist.append(curr_error)

            if len(error_hist) > 1 and error_hist[-1] < err_thresh:
                error_diff = abs(error_hist[-1] - error_hist[-2])
            else:
                error_diff = 1.0

        # =====================================================================
        # Post-Loop Processing
        # =====================================================================
        err_val: float = max(0.0, error_hist[-1] if error_hist else 1.0)

        corrected_signal: np.ndarray = x_curr * np.exp(-1j * (w0 * t + phi_off))

        ell_res: Optional[Dict[str, Any]] = fit_ellipse(
            np.real(corrected_signal), 3 * np.imag(corrected_signal)
        )

        flag: int = 0
        IQO_val: float = 0.0
        IQI_val: float = 0.0

        if ell_res is not None and ell_res.get("a") is not None:
            flag = 1
            ell_a_val: float = ell_res["a"]
            ell_b_val: float = ell_res["b"]
            ell_X0_val: float = ell_res["X0"]
            ell_Y0_val: float = ell_res["Y0"]

            IQO_val = np.sqrt(
                (ell_X0_val / ell_a_val) ** 2 + (ell_Y0_val / ell_b_val) ** 2
            )
            IQI_val = (ell_a_val / ell_b_val) * 3.0

        if flag == 1:
            f0_val = w0 / (2 * np.pi)
            phi_deg: float = phi * (360.0 / (2 * np.pi))
            phi_off_deg: float = phi_off * (360.0 / (2 * np.pi))

            fr_list.append(f0_val)
            I2 += I_val
            Q2 += Q_val
            IQO2 += IQO_val
            IQI2 += IQI_val
            e2 += e
            phi2 += phi_deg
            phi_off2 += phi_off_deg
            f02 += f0_val
            amp2 += amp
            error2 += err_val
        else:
            print("Ellipse was not found.")

    # Averaging results
    n_valid: int = n_iters if n_iters > 0 else 1

    I_final: float = I2 / n_valid
    Q_final: float = Q2 / n_valid
    amp_final: float = amp2 / n_valid

    if amp_final != 0:
        I_final = -I_final / amp_final
        Q_final = Q_final / amp_final

    IQO_final: float = IQO2 / n_valid
    IQI_final: float = IQI2 / n_valid
    phi_final: float = phi2 / n_valid
    e_final: float = e2 / n_valid
    phi_off_final: float = phi_off2 / n_valid
    f0_final: float = f02 / n_valid

    return (
        amp_final,
        e_final,
        phi_final,
        I_final,
        Q_final,
        IQO_final,
        IQI_final,
        f0_final,
        phi_off_final,
        error2,
        est_signal,
        fr_list,
        [],
    )
