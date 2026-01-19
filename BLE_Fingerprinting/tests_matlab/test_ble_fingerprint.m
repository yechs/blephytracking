% Manual test for BLE_Fingerprint
rng(2);
bits = randi([0 1], 1, 32);
fs = 1e6;
signal = gfsk_modulate(bits, 500e3, fs);

[fingerprint, decoded] = BLE_Fingerprint(signal, 40, fs, 0, 1, 1);

assert(numel(decoded) == numel(bits));
if ~isempty(fingerprint)
    assert(size(fingerprint, 2) == 25);
end

disp('test_ble_fingerprint passed');
