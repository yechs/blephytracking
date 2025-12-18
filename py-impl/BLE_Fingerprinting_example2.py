import numpy as np
import os
import time
from typing import List, Optional

# Import the main processing function
from BLE_Fingerprint import BLE_Fingerprint


def main() -> None:
    # Parameter setup
    Fs: float = 3.125e6
    snr: float = 40.0
    preamble_detect: int = 1
    interp_fac: int = 32
    n_partition: int = 250
    fingerprint_size: int = 25

    # Initialize storage
    # We will process 20 files as per the example
    num_files: int = 20
    fingerprint_all: np.ndarray = np.zeros((num_files, fingerprint_size))

    print(f"Starting Fingerprinting on {num_files} files...")
    start_time = time.time()

    for i in range(1, num_files + 1):
        # Reading the file
        # Path assumes 'Example_Data' directory exists relative to script
        sample_filepath: str = f"BLE_Fingerprinting/Example_Data/{i}"

        if not os.path.exists(sample_filepath):
            print(f"Warning: File {sample_filepath} not found. Skipping.")
            continue

        # MATLAB: [signal, ~] = fread(fid, 'float');
        # MATLAB stores as [I1, Q1, I2, Q2, ...] usually if it's complex interleaved
        # The reshape logic confirms this: reshape(signal, 2, []).' -> 2 rows, N cols, then transpose
        try:
            raw_data = np.fromfile(sample_filepath, dtype=np.float32)

            # Reshape to (N, 2) where col 0 is Real (I) and col 1 is Imag (Q)
            # MATLAB: reshape(signal, 2, []).'
            # Python reshape(-1, 2) creates N rows, 2 columns.
            iq_data = raw_data.reshape(-1, 2)

            # Create complex signal
            signal_complex = iq_data[:, 0] + 1j * iq_data[:, 1]

            # Slice first 10 for preview (optional)
            # print(signal_complex[:10])

            # Truncate end
            # MATLAB: signal = signal(1:end-12);
            signal_complex = signal_complex[:-12]

            # Physical layer fingerprinting
            fingerprint, bits = BLE_Fingerprint(
                signal_complex, snr, Fs, preamble_detect, interp_fac, n_partition
            )

            if fingerprint is not None:
                fingerprint_all[i - 1, :] = fingerprint
                print(f"Processed File {i}: Success")
            else:
                print(f"Processed File {i}: Failed to extract valid fingerprint")

        except Exception as e:
            print(f"Error processing file {i}: {e}")

    elapsed = time.time() - start_time
    print(f"Done. Total time: {elapsed:.2f} seconds.")

    # Check results (non-zero rows)
    valid_count = np.sum(np.any(fingerprint_all != 0, axis=1))
    print(f"Successfully fingerprinted {valid_count} / {num_files} devices.")

    # print final fingerprints
    print("Fingerprints:")
    for idx in range(num_files):
        print(f"Device {idx + 1}: {fingerprint_all[idx, :]}")


if __name__ == "__main__":
    main()
