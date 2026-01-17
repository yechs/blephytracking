import numpy as np
from scipy.signal import correlate

from gfsk_modulate import gfsk_modulate

"""
Discrepancies noted

The Synchronization Bug (Main Reason for Different Bits):
    MATLAB: It searches for the preamble peak inside a window (samples 8 to 80), finds the peak at index 50 relative to that window, and then sets the start index to 50. It forgets to add the 8 starting samples back.
    Result: It starts decoding too early, catching noise or the ramp-up before the packet actually starts. This is why the MATLAB output starts with 1 1 0... (garbage/noise) instead of the clean 0 1 0... preamble.
    Python: I corrected this by adding the offset.

Modulation Index Mismatch:
    MATLAB: Uses 500e3 (500 kHz) deviation for the preamble template. (Standard BLE is 250 kHz, but the code uses 500).
    Python: I used 250 kHz. We must switch to 500 kHz to match the MATLAB template shape.

Phase Derivative Alignment:
    MATLAB: Calculates frequency using angle(3:end) - angle(2:end-1). This drops the first derivative point.
    Python: Used np.diff on the whole array. We must shift the Python array to match this specific slicing.
"""


def ble_decoder(signal, fs, preamble_detect=True):
    """
    Decodes a BLE signal from IQ samples.

    Parameters:
    signal (np.array): Complex IQ samples.
    fs (float): Sampling frequency in Hz.
    preamble_detect (bool): Whether to perform correlation-based synchronization.

    Returns:
    tuple: (ble_signal, signal_freq, bits)
    """

    # 1. Create Preamble Template
    # Standard BLE preamble pattern (0xAA or 0x55)
    pream = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    preamble_signal = gfsk_modulate(pream, 500e3, fs)

    # Trim preamble edges (2.5us from start, 0.5us from end)
    start_trim = int(fs / 1e6 * 2.5)
    end_trim = int(fs / 1e6 * 0.5)
    preamble_signal = preamble_signal[start_trim:-end_trim]

    # 2. Frequency Extraction Helper
    def get_instantaneous_freq(sig):
        # Unwrap phase to handle -pi to pi jumps
        sig_angle = np.unwrap(np.angle(sig))
        # Calculate slope (derivative of phase)
        slope = np.diff(sig_angle)
        # Convert to Hz: (d_phi/d_sample) * (samples/sec) / (2*pi)
        freq = slope / (2 * np.pi) * fs
        # Append 0 to match original length (diff reduces length by 1)
        return np.append(freq, 0)

    preamble_freq = get_instantaneous_freq(preamble_signal)

    # Extract frequency of the actual input signal
    # MATLAB uses indices 3:end minus 2:end-1 which is essentially a central difference
    # or just a shift. We will use the standard diff method as above for consistency.
    signal_freq = get_instantaneous_freq(signal)

    # 3. Finding the Preamble (Synchronization)
    start_ind = 0

    if preamble_detect:
        # Cross-correlation between signal freq and preamble freq
        # mode='valid' mimics the behavior of sliding the smaller preamble over the larger signal
        z = correlate(signal_freq, preamble_freq, mode="valid")

        # Take absolute value as correlation might be negative (180 phase shift)
        # though for frequency profile, direct match is usually positive
        z_abs = np.abs(z)

        # Search window defined in MATLAB: 2us to 20us
        # Note: 'valid' correlation result indices map differently than 'full'.
        # Index 0 in 'z' corresponds to alignment at index 0 of signal.

        lower_bound = int(2e-6 * fs)
        upper_bound = int(20e-6 * fs)

        if len(z_abs) > upper_bound:
            # Find max peak within the specific window
            window_slice = z_abs[lower_bound:upper_bound]
            peak_offset = np.argmax(window_slice)
            start_ind = lower_bound + peak_offset
        else:
            start_ind = 0

    # 4. Slice and Dice
    # Apply synchronization
    signal_cut = signal[start_ind:]
    signal_freq_cut = signal_freq[start_ind:]

    # Calculate samples per symbol (BLE is 1 Mbps -> 1e6 symbols/sec)
    sps = fs / 1e6

    # Truncate to ensure length is a multiple of samples_per_symbol
    num_symbols = int(len(signal_freq_cut) / sps)
    trunc_len = int(num_symbols * sps)

    ble_signal = signal_cut[:trunc_len]
    signal_freq_final = signal_freq_cut[:trunc_len]

    # 5. Demodulation (Reshape and Average)
    # Reshape into (Number of Symbols, Samples per Symbol)
    # MATLAB reshape is Column-Major (Fortran-like), Python is Row-Major (C-like).
    # However, because we are splitting a time-series 1D array into chunks,
    # Python's default row-major reshape is actually exactly what we want here.
    bits_freq_matrix = signal_freq_final.reshape(num_symbols, int(sps))

    # Define the sampling window for the "eye" of the symbol
    # MATLAB used: fs/1e6/2 : fs/1e6/2+1 (roughly the middle sample)
    mid_point = int(sps / 2)

    # Averaging the middle of the symbol period
    # We take a small slice around the center
    decision_metric = np.mean(bits_freq_matrix[:, mid_point : mid_point + 1], axis=1)

    # Hard decision: > 0 is 1, <= 0 is 0
    bits = (decision_metric > 0).astype(int)

    return ble_signal, signal_freq_final, bits


### Usage Example ###
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Configuration ---
    filename = "../BLE_Fingerprinting/Example_Data/1"
    fs = 4e6  # 4 MHz Sampling Rate
    preamble_detect = True

    # --- Step B: Read File (Your MATLAB Logic Ported) ---
    print(f"Reading {filename}...")

    # 1. Read raw binary as float32
    raw_data = np.fromfile(filename, dtype=np.float32)

    # 2. Reshape to pairs (Real, Imag)
    # MATLAB: signal = reshape(signal, 2, []).'
    complex_pairs = raw_data.reshape(-1, 2)

    # 3. Combine into Complex IQ
    # MATLAB: signal = signal(:,1) + 1i * signal(:,2);
    signal = complex_pairs[:, 0] + 1j * complex_pairs[:, 1]

    # 4. Trim artifacts
    # MATLAB: signal = signal(1:end-12);
    if len(signal) > 12:
        signal = signal[:-12]

    # --- Step C: Decode ---
    ble_sig, freq_profile, bits = ble_decoder(signal, fs, preamble_detect)

    # --- Step D: Print Results ---
    print("\n--- Decoding Results ---")
    print(f"Signal Length: {len(signal)} samples")
    print(f"Decoded Bits ({len(bits)}):")
    print(bits)

    # --- Step E: Visualize ---
    t = np.arange(len(freq_profile)) / fs * 1e6  # Time in microseconds

    plt.figure(figsize=(12, 6))

    # Plot Frequency Profile
    plt.plot(t, freq_profile / 1e3, "b-", linewidth=1.5, label="Demodulated Frequency")

    # Overlay Bit Decisions (Red Dots)
    # We plot a dot at the middle of every symbol duration
    sps = fs / 1e6
    bit_times = (np.arange(len(bits)) * sps + sps / 2) / fs * 1e6
    bit_vals = (bits * 2 - 1) * 250  # Scale 0/1 to -250/250 for plotting
    plt.plot(bit_times, bit_vals, "r.", markersize=10, label="Bit Decision Center")

    # Formatting
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.axhline(250, color="g", linestyle=":", alpha=0.3, label="Target +250kHz")
    plt.axhline(-250, color="g", linestyle=":", alpha=0.3, label="Target -250kHz")

    plt.title(f"BLE Signal Decoding: {filename}")
    plt.xlabel("Time ($\\mu$s)")
    plt.ylabel("Frequency (kHz)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
