import numpy as np
from scipy import interpolate
import warnings
from typing import Tuple, List, Optional, Any, Dict, Union

# Import our previously defined modules
from BLE_Decoder import BLE_Decoder
from BLE_Imperfection_Estimator_NAGD import BLE_Imperfection_Estimator_NAGD
from fit_ellipse import fit_ellipse


def BLE_Fingerprint(
    signal_in: np.ndarray,
    snr: float,
    fs: float,
    preamble_detect: int,
    interp_fac: int,
    n_partition: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes BLE fingerprints including CFO, I/Q offset, and I/Q imbalance.

    Parameters:
    - signal_in: Input complex signal array.
    - snr: Signal-to-Noise Ratio (dB).
    - fs: Sampling frequency (Hz).
    - preamble_detect: Flag to enable preamble detection (1 or 0).
    - interp_fac: Upsampling/Interpolation factor.
    - n_partition: Number of partitions for the estimator.

    Returns:
    - fingerprint: A 25-element feature vector (or None if failed).
    - bits: Decoded binary bits (or None).
    """

    # Constants
    fc: float = 2.48e9
    ch: float = 2.48e9

    # Update sampling rate
    fs_new: float = fs * interp_fac

    # Initialize output
    fingerprint: Optional[np.ndarray] = None

    # Normalize signal
    # MATLAB: signal = signal/mean(abs(signal));
    signal_norm: np.ndarray = signal_in / np.mean(np.abs(signal_in))

    # Interpolate
    # MATLAB: signal = interp(signal,interp_fac);
    # scipy.signal.resample_poly is the efficient way to do integer upsampling
    if interp_fac != 1:
        # Note: MATLAB's interp adds a low-pass filter gain of interp_fac.
        # resample_poly handles the filtering but check gain.
        # We usually just use resample_poly(x, up, down).
        from scipy.signal import resample_poly

        signal_interp = resample_poly(signal_norm, interp_fac, 1)
    else:
        signal_interp = signal_norm

    # Remove channel offset and center signal
    if interp_fac != 1:
        # FFT shift logic
        signal_fft = np.fft.fftshift(np.fft.fft(signal_interp))
        ls = len(signal_interp)

        signal_fft_centered = np.zeros(ls, dtype=complex)
        lcenter = ls // 2

        # Calculate indices (carefully matching MATLAB 1-based logic to 0-based)
        # MATLAB: lchannel = floor((ch-fc)/fs*ls+ls/2);
        lchannel = int(np.floor((ch - fc) / fs_new * ls + ls / 2))

        # MATLAB: lbandwidth = floor(2/(fs/1e6)*(ls-1)/2);
        # fs is currently fs_new.
        # Wait, MATLAB code uses 'fs' variable which was updated to fs*interp_fac.
        lbandwidth = int(np.floor(2 / (fs_new / 1e6) * (ls - 1) / 2))

        # Slicing
        # Python ranges are [start, end), MATLAB is [start, end].
        # We need width = 2*lbandwidth + 1 ideally.

        # Indices in source (channel)
        src_start = lchannel - lbandwidth
        src_end = lchannel + lbandwidth + 1  # +1 for Python slice inclusion

        # Indices in dest (center)
        dst_start = lcenter - lbandwidth
        dst_end = lcenter + lbandwidth + 1

        # Boundary checks
        if src_start >= 0 and src_end <= ls and dst_start >= 0 and dst_end <= ls:
            signal_fft_centered[dst_start:dst_end] = signal_fft[src_start:src_end]

        signal_final = np.fft.ifft(np.fft.ifftshift(signal_fft_centered))
    else:
        signal_final = signal_interp

    # Decode
    decoded_signal, signal_freq, bits = BLE_Decoder(
        signal_final, fs_new, preamble_detect
    )

    # Estimate CFO from Preamble
    # MATLAB: pream = signal_freq(1:fs/1e6*8-1);
    pream_end_idx = int(fs_new / 1e6 * 8 - 1)
    pream = signal_freq[0:pream_end_idx]

    est_cfo: float = np.mean(pream)
    est_cfo2: float = est_cfo

    if abs(est_cfo2) > 100e3:
        est_cfo2 = 0.0

    # Run Imperfection Estimator
    # Note: Using 0.0 for initial guesses as per MATLAB code
    (amp, epsilon, phi, I, Q, IQO, IQI, f0, phi_off, error_metric, _, _, _) = (
        BLE_Imperfection_Estimator_NAGD(
            decoded_signal,
            bits,
            fs_new,
            est_cfo2,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            snr,
            n_partition,
        )
    )

    # Post-Estimator Signal Correction
    tt = np.arange(len(decoded_signal)) * (1.0 / fs_new)
    # MATLAB: signal = signal.*exp(-1j*(2*pi*f0*tt'+phi_off/(360/(2*pi))));
    # phi_off from estimator is likely in degrees or radians?
    # In Estimator.py we returned it, let's assume consistent units.
    # The MATLAB caller divides phi_off by (360/2pi) -> converting degrees back to radians?
    # Let's assume phi_off is in degrees if it's being divided by ~57.3
    correction_phase = 2 * np.pi * f0 * tt + phi_off / (360.0 / (2 * np.pi))
    corrected_sig = decoded_signal * np.exp(-1j * correction_phase)

    # Ellipse Fit on corrected signal
    flag: int = 1
    ell: Dict[str, Any] = {}

    try:
        # MATLAB: sig = signal(randperm(length(signal)));
        # MATLAB: ell = fit_ellipse(-real(sig)/amp*5, 3*imag(sig)/amp);
        sig_perm = np.random.permutation(corrected_sig)

        x_ell = -np.real(sig_perm) / amp * 5
        y_ell = 3 * np.imag(sig_perm) / amp

        ell_res = fit_ellipse(x_ell, y_ell)
        if ell_res is None or ell_res["a"] is None:
            flag = 0
        else:
            ell = ell_res
    except Exception:
        warnings.warn("Ill ellipse in final fingerprinting")
        flag = 0

    # Quadrant Mean Calculation
    angsig = np.angle(corrected_sig)
    spl = 8
    quar = np.zeros(spl, dtype=complex)

    # MATLAB loop for sp = 1:spl/2
    for sp in range(1, spl // 2 + 1):
        # Range 1: (sp-1)*2pi/spl to sp*2pi/spl
        lower1 = (sp - 1) * 2 * np.pi / spl
        upper1 = sp * 2 * np.pi / spl

        # Boolean mask
        mask1 = (angsig > lower1) & (angsig < upper1)
        if np.any(mask1):
            quar[sp - 1] = np.mean(corrected_sig[mask1])

        # Range 2: -pi + ...
        lower2 = -np.pi + (sp - 1) * 2 * np.pi / spl
        upper2 = -np.pi + sp * 2 * np.pi / spl

        mask2 = (angsig > lower2) & (angsig < upper2)
        if np.any(mask2):
            quar[sp - 1 + spl // 2] = np.mean(corrected_sig[mask2])

    # Construct Feature Vector
    # We must handle cases where ell is empty to avoid crashes
    if flag == 1:
        # Extract scalar values from ell dict
        ell_a = ell["a"]
        ell_b = ell["b"]
        ell_X0 = ell["X0"]
        ell_Y0 = ell["Y0"]
        ell_X0_in = ell["X0_in"]
        ell_Y0_in = ell["Y0_in"]
        ell_phi = ell["phi"]

        fingerprint_vec = np.array(
            [
                error_metric,
                amp,
                f0,
                est_cfo,
                IQO,
                I,
                Q,
                np.sqrt(I**2 + Q**2),
                IQI,
                epsilon,
                phi,
                ell_X0 / ell_a,
                ell_Y0 / ell_b,
                ell_X0_in / ell_a,
                ell_Y0_in / ell_b,
                np.sqrt((ell_X0 / ell_a) ** 2 + (ell_Y0 / ell_b) ** 2),
                np.sqrt((ell_X0_in / ell_a) ** 2 + (ell_Y0_in / ell_b) ** 2),
                ell_a * 3 / ell_b / 5,
                ell_phi,
                np.real(np.mean(quar)),
                np.imag(np.mean(quar)),
                np.abs(np.mean(quar)),
                np.mean(np.real(corrected_sig)),
                np.mean(np.imag(corrected_sig)),
                np.abs(
                    np.mean(np.real(corrected_sig))
                    + 1j * np.mean(np.imag(corrected_sig))
                ),
            ]
        )

        # Verification
        if len(fingerprint_vec) != 25:
            flag = 0

        # MATLAB threshold check: if error(end) < 0.45 && flag == 1
        # error_metric passed from estimator is a scalar (the last error)
        if error_metric < 0.45 and flag == 1:
            fingerprint = fingerprint_vec

    return fingerprint, bits
