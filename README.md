# blephytracking
This repository includes the code for BLE physical-layer fingerprinting project. 

BLE_Fingerprinting_example2.m reads 20 example BLE signals received by a Software Defined Radio (SDR) over the air, and extracts physical layer fingerprints of those signals. The first 10 signals belong to one transmitter device and the next 10 signals belong to a different device.

The same workflow is also available in Python as `BLE_Fingerprinting/ble_fingerprinting_example2.py`, which uses `BLE_Fingerprinting/ble_fingerprinting.py` for the ported signal-processing routines.

Citation: H. Givehchian, N. Bhaskar, E. R. Herrera, H. R. L. Soto, C. Dameff,D. Bharadia, and A. Schulman, 
          “Evaluating Physical-Layer BLE Location Tracking Attacks on Mobile Devices,” 
          in 2022 IEEE Symposium on Security and Privacy, 2022
