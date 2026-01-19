% Manual test for BLE_Imperfection_Estimator_NAGD
rng(1);
bits = randi([0 1], 1, 16);
fs = 1e6;
signal = gfsk_modulate(bits, 500e3, fs);

[amp,e,phi,I,Q,IQO,IQI,f0,phi_off,error,~,~,~] = BLE_Imperfection_Estimator_NAGD(...
    signal, bits', fs, 0, 0, 0, 0, 0, 1, 40, 1);

vals = [amp e phi I Q IQO IQI f0 phi_off error];
assert(all(isfinite(vals)));

disp('test_ble_imperfection_estimator_nagd passed');
