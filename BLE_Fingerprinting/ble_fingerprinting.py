import numpy as np

from ble_fingerprint import ble_fingerprint


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
