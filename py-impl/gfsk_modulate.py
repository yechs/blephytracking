import numpy as np
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid


def design_gaussian_filter(bt, span, sps):
    """
    Generates a Gaussian FIR filter pulse to match MATLAB's gaussdesign.

    Parameters:
        bt   : Bandwidth-Time product (usually 0.3 for BLE)
        span : Filter span in symbols (usually 3)
        sps  : Samples per symbol
    """
    # Create a time vector centered at 0, covering the span of symbols
    # We produce (span * sps) + 1 coefficients to match typical FIR designs
    t = np.linspace(-span / 2, span / 2, int(span * sps) + 1)

    # Calculate the Gaussian coefficients
    # alpha relates to the bandwidth B
    alpha = np.sqrt(np.log(2)) / (np.sqrt(2) * bt)
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    # Normalize so the sum of coefficients is 1 (0 dB gain)
    # This ensures the frequency deviation limit is preserved after filtering
    h = h / np.sum(h)
    return h


def gfsk_modulate(bits, freq_sep, fs, bt=0.3, span=3):
    """
    Modulates bits using GFSK (Gaussian Frequency Shift Keying).

    Parameters:
        bits     : Input binary data (list or numpy array of 0s and 1s)
        freq_sep : Peak-to-peak frequency separation (Hz)
        fs       : Sampling rate (Hz)
        bt       : Bandwidth-Time product (default 0.3 for BLE)
        span     : Symbol span for the filter (default 3)

    Returns:
        y : Complex baseband IQ signal
    """
    bits = np.array(bits)

    # 1. Calculate Samples Per Symbol (SPS)
    # Assuming symbol rate is 1 Msps (standard for BLE 1M PHY)
    symbol_rate = 1e6
    sps = int(fs / symbol_rate)

    # 2. NRZ Mapping (0 -> -1, 1 -> 1) and Upsampling
    # We repeat each bit 'sps' times to create the "square wave"
    x_nrz = (bits * 2) - 1
    gamma_fsk = np.repeat(x_nrz, sps)

    # 3. Gaussian Pulse Shaping
    # Generate the filter coefficients
    gauss_coeffs = design_gaussian_filter(bt, span, sps)

    # Apply the filter. lfilter is equivalent to MATLAB's filter (IIR/FIR)
    # scaling by 1.0 for the denominator (FIR filter)
    gamma_gfsk = signal.lfilter(gauss_coeffs, 1.0, gamma_fsk)

    # 4. Frequency to Phase Conversion
    # We integrate the frequency signal to get phase.
    # cumulative_trapezoid is the equivalent of cumtrapz.
    # We set initial=0 to ensure the output array length matches the input.
    theta = (freq_sep / fs) * np.pi * cumulative_trapezoid(gamma_gfsk, initial=0)

    # 5. IQ Signal Generation
    y = np.exp(1j * theta)

    return y


# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters
    fs = 4e6  # 4 MHz Sampling Rate
    freq_sep = 500e3  # 500 kHz separation (BLE standard)
    data_bits = [0, 1, 0, 1, 1, 0, 0, 1]  # Sample bits

    # Modulate
    iq_signal = gfsk_modulate(data_bits, freq_sep, fs)

    print(f"Generated IQ signal with {len(iq_signal)} samples.")
    print(iq_signal)

    # Visualization
    t_axis = np.arange(len(iq_signal)) / fs

    plt.figure(figsize=(10, 6))

    # Plot Real (I) and Imag (Q) components
    plt.subplot(2, 1, 1)
    plt.plot(t_axis * 1e6, np.real(iq_signal), label="I (In-Phase)")
    plt.plot(t_axis * 1e6, np.imag(iq_signal), label="Q (Quadrature)", alpha=0.7)
    plt.title("GFSK Baseband Signal (I/Q)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Plot Instantaneous Frequency (approx) to verify Gaussian shaping
    # Calculate freq from phase difference
    phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs

    plt.subplot(2, 1, 2)
    plt.plot(t_axis[:-1] * 1e6, inst_freq / 1e3, color="green")
    plt.title("Instantaneous Frequency")
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Frequency (kHz)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
