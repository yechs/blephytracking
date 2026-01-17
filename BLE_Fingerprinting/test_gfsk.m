% --- Configuration ---
Fs = 4e6;               % 4 MHz Sampling Rate
freqsep = 500e3;        % 500 kHz separation (BLE standard)
bits = [0 1 0 1 1 0 0 1]; % Sample bits

% --- Modulation ---
% (Assumes gfsk_modulate.m is in your path or current folder)
y = gfsk_modulate(bits, freqsep, Fs);

% --- Visualization ---
% Create time vector for plotting
% Length of output is bits * (Fs/1e6)
t_axis = (0:length(y)-1) / Fs;

figure('Name', 'GFSK Verification', 'Color', 'w');

% 1. Plot I/Q Baseband Signals
subplot(2, 1, 1);
plot(t_axis * 1e6, real(y), 'LineWidth', 1.5, 'DisplayName', 'I (In-Phase)');
hold on;
plot(t_axis * 1e6, imag(y), 'LineWidth', 1.5, 'DisplayName', 'Q (Quadrature)');
hold off;
title('GFSK Baseband Signal (I/Q)');
ylabel('Amplitude');
xlabel('Time (\mus)'); % microseconds
legend('show');
grid on;

% 2. Plot Instantaneous Frequency
% We recover frequency from the phase derivative to verify the Gaussian shape
phase_vals = unwrap(angle(y));
inst_freq = diff(phase_vals) * Fs / (2*pi);

% Adjust time axis for frequency plot (diff reduces length by 1)
t_freq = t_axis(1:end-1);

subplot(2, 1, 2);
plot(t_freq * 1e6, inst_freq / 1e3, 'g', 'LineWidth', 1.5);
title('Instantaneous Frequency');
ylabel('Frequency (kHz)');
xlabel('Time (\mus)');
grid on;
ylim([-300 300]); % Set limits to visualize the +/- 250kHz deviation clearly