% Manual test for BLE_Decoder
bits = [0 1 0 1 1 0 1];
fs = 2e6;
signal = gfsk_modulate(bits, 500e3, fs);

[ble_signal, signal_freq, decoded] = BLE_Decoder(signal, fs, 0);

assert(numel(ble_signal) == numel(signal_freq));
assert(numel(decoded) == numel(bits));

disp('test_ble_decoder passed');
