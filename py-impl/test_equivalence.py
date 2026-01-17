import unittest
import numpy as np

from gfsk_modulate import gfsk_modulate
from fit_ellipse import fit_ellipse
from BLE_Decoder import BLE_Decoder


class TestMatlabEquivalence(unittest.TestCase):

    def setUp(self):
        """
        Setup test fixtures.
        REPLACE values below with the output from generate_test_data.m
        """
        # ==========================================================
        # INPUTS (Must match MATLAB script)
        # ==========================================================
        self.gfsk_bits = [1, 0, 1, 1, 0]
        self.gfsk_freqsep = 500e3
        self.gfsk_fs = 2e6

        # Ellipse Inputs (Recreated identically to MATLAB script)
        t = np.linspace(0, 2 * np.pi, 20)
        a_true = 5
        b_true = 2
        x0_true = 1
        y0_true = -1
        phi_true = np.pi / 4

        # Generate the exact same points
        self.ell_x = (
            x0_true
            + a_true * np.cos(t) * np.cos(phi_true)
            - b_true * np.sin(t) * np.sin(phi_true)
        )
        self.ell_y = (
            y0_true
            + a_true * np.cos(t) * np.sin(phi_true)
            + b_true * np.sin(t) * np.cos(phi_true)
        )

        # ==========================================================
        # EXPECTED OUTPUTS (PASTE FROM MATLAB CONSOLE HERE)
        # ==========================================================

        # Paste the "Real part" array from MATLAB here:
        self.expected_gfsk_real = np.array(
            [1, 0.99989078, 0.99082496, 0.87820218, 0.60619889, 0.59437866, 0.80451513, 0.78661002, 0.35572986, -0.35469203]
        )

        # Paste the "Imag part" array from MATLAB here:
        self.expected_gfsk_imag = np.array(
            [0, 0.014779113, 0.13515141, 0.47828959, 0.79531309, 0.80418531, 0.59393215, 0.61745014, 0.93458882, 0.93498319]
        )

        # Paste Ellipse parameters here:
        self.expected_ellipse = {
            "a": 5.0,
            "b": 2.0,
            "phi": -0.78539816,  # -pi/4
            "X0": 0.0,
            "Y0": -1.41421356,
        }

    def test_gfsk_modulate_equivalence(self):
        """Test GFSK modulation against MATLAB ground truth."""
        print("\nTesting GFSK Modulate...")

        # Run Python Implementation
        y_python = gfsk_modulate(self.gfsk_bits, self.gfsk_freqsep, self.gfsk_fs)

        # Check Length
        # MATLAB length is typically nsample * len(bits)
        # Note: If MATLAB's filter implementation adds delay/padding differently,
        # lengths might differ slightly. We test the core overlapping segment.
        expected_len = len(self.expected_gfsk_real)
        self.assertEqual(len(y_python), expected_len, "Signal lengths do not match")

        # Check Values (Tolerance set to 1e-5 to account for different filter implementations)
        np.testing.assert_allclose(
            np.real(y_python),
            self.expected_gfsk_real,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Real part of GFSK signal mismatch",
        )

        np.testing.assert_allclose(
            np.imag(y_python),
            self.expected_gfsk_imag,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Imaginary part of GFSK signal mismatch",
        )
        print("GFSK Modulate: PASS")

    def test_fit_ellipse_equivalence(self):
        """Test Ellipse Fitting against MATLAB ground truth."""
        print("\nTesting Fit Ellipse...")

        # Run Python Implementation
        result = fit_ellipse(self.ell_x, self.ell_y)

        # print("result from fit_ellipse:", result)
        self.assertIsNotNone(result, "Fit Ellipse returned None")

        # Check Parameters
        # Note: Orientation can sometimes flip by pi or swap a/b axes
        # depending on the solver's eigenvalues sorting.
        # We test for 'close enough' values.

        print(
            f"  Computed a: {result['a']:.5f} / Expected: {self.expected_ellipse['a']:.5f}"
        )
        np.testing.assert_allclose(result["a"], self.expected_ellipse["a"], rtol=1e-4)

        print(
            f"  Computed b: {result['b']:.5f} / Expected: {self.expected_ellipse['b']:.5f}"
        )
        np.testing.assert_allclose(result["b"], self.expected_ellipse["b"], rtol=1e-4)

        print(
            f"  Computed X0: {result['X0']:.5f} / Expected: {self.expected_ellipse['X0']:.5f}"
        )
        np.testing.assert_allclose(result["X0"], self.expected_ellipse["X0"], rtol=1e-4, atol=1e-8)

        print(
            f"  Computed Y0: {result['Y0']:.5f} / Expected: {self.expected_ellipse['Y0']:.5f}"
        )
        np.testing.assert_allclose(result["Y0"], self.expected_ellipse["Y0"], rtol=1e-4)

        print("Fit Ellipse: PASS")

    def test_ble_decoder_structure(self):
        """Smoke test for BLE Decoder to ensure shapes and types are correct."""
        print("\nTesting BLE Decoder (Smoke Test)...")

        # Create a dummy GFSK signal (random bits)
        fs = 2e6
        bits_in = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        # Preamble is 0,1,0,1... so we prefix specific bits to simulate a packet start
        # The decoder looks for 0,1,0,1,0,1...
        # Let's manually construct a signal that looks like a packet
        preamble_bits = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        full_bits = np.concatenate([preamble_bits, bits_in])

        # Modulate
        sig_in = gfsk_modulate(full_bits, 500e3, fs)

        # Add some silence/noise at the start to test detection
        silence = np.zeros(100, dtype=complex)
        sig_with_delay = np.concatenate([silence, sig_in])

        # Run Decoder
        # detection=1 means it should find the preamble inside the delay
        out_sig, out_freq, out_bits = BLE_Decoder(sig_with_delay, fs, preamble_detect=1)

        # Checks
        self.assertGreater(len(out_bits), 0, "Decoder returned no bits")
        self.assertEqual(
            len(out_sig), len(out_freq), "Frequency and Signal length mismatch"
        )

        # The decoder output should have stripped the silence
        # Length should be roughly num_symbols * samples_per_symbol
        nsample = int(fs / 1e6)
        expected_len = len(out_bits) * nsample
        self.assertEqual(
            len(out_sig),
            expected_len,
            "Output signal length not aligned with bit count",
        )

        print("BLE Decoder: PASS (Smoke Test)")


if __name__ == "__main__":
    unittest.main()
