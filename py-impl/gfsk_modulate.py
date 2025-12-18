import numpy as np
from scipy import signal, integrate

def design_gaussian_filter(bt, span, sps):
    """
    Design a Gaussian FIR pulse-shaping filter.
    Equivalent to MATLAB's gaussdesign(BT, span, sps).

    Parameters:
    - bt: Bandwidth-Time product
    - span: Filter span in symbols
    - sps: Samples per symbol
    """
    # Total number of taps
    n_taps = int(span * sps)
    if n_taps % 2 == 0:
        n_taps += 1  # Ensure odd number of taps for symmetry

    t = np.linspace(-(span/2), (span/2), n_taps)
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Gaussian impulse response
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t) / alpha)**2)

    # Normalize so the sum of coefficients is 1 (standard for smoothing)
    # Note: In comms, we often normalize energy, but MATLAB's gaussdesign
    # typically normalizes peak or sum depending on config.
    # For GFSK frequency pulse, we usually want unit area to maintain phase steps.
    h = h / np.sum(h)

    return h

def gfsk_modulate(x, freqsep, fs):
    """
    Generates a BLE GFSK signal.

    Parameters:
    - x: Input binary sequence (array-like of 0s and 1s)
    - freqsep: Separation frequency (Hz)
    - fs: Sampling rate (Hz)
    """
    x = np.array(x)
    nsample = int(fs / 1e6)  # BLE symbol rate is 1 Msps

    # Create time vector
    # MATLAB: t = (1:(nsample*length(x)))*(1/Fs);
    total_samples = nsample * len(x)

    # Convert bits to NRZ (Non-Return-to-Zero) pulses: 0 -> -1, 1 -> 1
    # MATLAB logic: gamma_fsk((((i-1)*nsample)+1):(i*nsample)) = ((x(i)*2)-1);
    # In Python we can use repeat for efficiency
    gamma_fsk = np.repeat(x * 2 - 1, nsample)

    # Create Gaussian Filter
    # MATLAB: gaussFilter = gaussdesign(0.3, 3, nsample);
    gauss_filter = design_gaussian_filter(0.3, 3, nsample)

    # Apply Filter
    # MATLAB: gamma_gfsk = filter(gaussFilter, 1, gamma_fsk);
    gamma_gfsk = signal.lfilter(gauss_filter, 1.0, gamma_fsk)

    # Integrate frequency to get phase
    # MATLAB: gfsk_phase = (freqsep/Fs)*pi*cumtrapz(gamma_gfsk);
    # scipy.integrate.cumulative_trapezoid returns array of len N-1 by default,
    # unless initial=0 is set (available in newer scipy) or we insert 0.
    # MATLAB's cumtrapz output is same length as input.
    integrated_gamma = integrate.cumulative_trapezoid(gamma_gfsk, initial=0)
    gfsk_phase = (freqsep / fs) * np.pi * integrated_gamma

    # IQ Modulation
    # MATLAB: y = exp(1i*gfsk_phase);
    y = np.exp(1j * gfsk_phase)

    return y
