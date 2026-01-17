% Manual test for gfsk_modulate
bits = [0 1 1 0 1];
fs = 1e6;
signal = gfsk_modulate(bits, 500e3, fs);

assert(length(signal) == length(bits) * fs/1e6);
assert(all(abs(abs(signal) - 1) < 1e-6));

disp('test_gfsk_modulate passed');
