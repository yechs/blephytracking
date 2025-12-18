import numpy as np
from scipy import signal

# Import the gfsk_modulate function we created previously
from gfsk_modulate import gfsk_modulate


def BLE_Decoder(ble_signal_in, fs, preamble_detect=1):
    """
    Implements a simple Bluetooth demodulator.

    Parameters:
    - ble_signal_in: Complex baseband signal (1D numpy array)
    - fs: Sampling frequency (Hz)
    - preamble_detect: 1 to detect preamble, 0 to assume start at index 0

    Returns:
    - ble_signal: The aligned signal segment
    - signal_freq: Instantaneous frequency array
    - bits: Decoded binary bits (1D numpy array of booleans/ints)
    """

    # Ensure input is 1D array
    ble_signal_in = np.array(ble_signal_in).flatten()
    nsample = int(fs / 1e6)  # Samples per symbol (sps)

    # =========================================================================
    # 1. Create Reference Preamble
    # =========================================================================
    # MATLAB: pream = [0,1,0,1,0,1,0,1,0,1,0];
    pream = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    # Generate modulated preamble
    preamble_signal = gfsk_modulate(pream, 500e3, fs)

    # Trim preamble signal
    # MATLAB: preamble_signal = preamble_signal(fs/1e6*2.5:end-fs/1e6*0.5);
    start_cut = int(nsample * 2.5)
    end_cut = int(nsample * 0.5)
    # Handle negative index for slicing from end
    preamble_signal = preamble_signal[start_cut : -end_cut if end_cut > 0 else None]

    # Calculate preamble frequency profile
    # MATLAB: slope = signal_angle(2:end)-signal_angle(1:end-1);
    preamble_phase = np.unwrap(np.angle(preamble_signal))
    preamble_freq = np.diff(preamble_phase) * fs / (2 * np.pi)

    # =========================================================================
    # 2. Calculate Signal Frequency
    # =========================================================================
    signal_phase = np.unwrap(np.angle(ble_signal_in))
    signal_freq_raw = np.diff(signal_phase) * fs / (2 * np.pi)

    # Pad with 0 to match length of original signal (MATLAB does this at end)
    # MATLAB: signal_freq = [signal_freq;0];
    signal_freq_raw = np.append(signal_freq_raw, 0)

    # =========================================================================
    # 3. Preamble Detection
    # =========================================================================
    start_ind = 0

    if preamble_detect == 0:
        start_ind = 0
    else:
        l = len(signal_freq_raw)

        # Correlate
        # MATLAB: z = xcorr(signal_freq, preamble_freq);
        # In Python, we use correlate. To match xcorr behavior where we want
        # the lag, we correlate signal against preamble.
        z = signal.correlate(signal_freq_raw, preamble_freq, mode="full")

        # Slicing z to match MATLAB's logic: z = z(l+1:end)
        # SciPy correlate 'full' output length is N + M - 1.
        # The center (lag 0) is roughly at index l-1.
        # MATLAB indices are 1-based. z(l+1) in MATLAB is index l in Python.
        # However, we must ensure we are looking at the same lags.
        # We perform the slice exactly as requested:
        z_sliced = z[l:]

        # Search window: 2us to 20us
        # MATLAB: abs(z(floor(2e-6*fs):floor(20e-6*fs)))
        window_start = int(2e-6 * fs)
        window_end = int(20e-6 * fs)

        if len(z_sliced) > window_end:
            # Find max in window
            window_slice = np.abs(z_sliced[window_start:window_end])
            max_idx_in_window = np.argmax(window_slice)

            # MATLAB: [~,start_ind] = max(...)
            # The index returned is relative to the start of the window.
            # We add window_start to get the index relative to z_sliced.
            start_ind = window_start + max_idx_in_window

            # NOTE: MATLAB code mentions commented out line "start_ind = start_ind+fs/1e6/2"
            # We stick to the active code.
        else:
            start_ind = 0

    # =========================================================================
    # 4. Extract Signal and Decode Bits
    # =========================================================================

    # Slice signal from start_ind
    current_signal = ble_signal_in[start_ind:]
    current_freq = signal_freq_raw[start_ind:]

    # Truncate to integer number of symbols
    # MATLAB: floor(length(signal_freq)/(fs/1e6))*(fs/1e6)
    num_symbols = int(len(current_freq) // nsample)
    valid_length = num_symbols * nsample

    final_signal = current_signal[:valid_length]
    final_freq = current_freq[:valid_length]

    # Decode Bits
    # MATLAB: bits_freq = reshape(signal_freq, fs/1e6, length...)'
    # MATLAB reshape is Column-Major. It takes column 1, then column 2...
    # Then the transpose (') turns columns into rows.
    # Effectively, it chunks the 1D array into rows of length `nsample`.
    # Python reshape (-1, nsample) does exactly this (Row-Major fill).
    bits_freq_matrix = final_freq.reshape(-1, nsample)

    # Sampling bits
    # MATLAB: mean(bits_freq(:, fs/1e6/2 : fs/1e6/2+1), 2) > 0
    # MATLAB indices (1-based): sps/2 to sps/2+1
    # Example sps=32 -> indices 16 to 17.
    # Python indices (0-based): 16 to 18 (exclusive)
    sample_start = int(nsample / 2)
    sample_end = sample_start + 2  # Taking 2 samples

    # Take mean across the sampled columns (axis 1)
    bit_decisions = np.mean(bits_freq_matrix[:, sample_start:sample_end], axis=1)

    bits = (bit_decisions > 0).astype(int)

    return final_signal, final_freq, bits
