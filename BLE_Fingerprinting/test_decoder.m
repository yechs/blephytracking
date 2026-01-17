% % === MATLAB Usage & Visualization Script ===
% clear; clc; close all;
% 
% % 1. Setup Parameters
% fs = 4e6;              % Sample rate: 4 MHz
% baud_rate = 1e6;       % BLE standard: 1 Mbps
% sps = fs / baud_rate;  % Samples per symbol (4)
% 
% % 2. Generate Ground Truth Bits
% % Preamble (0,1,0,1...) + Sync + Payload
% preamble = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
% payload  = [1, 1, 1, 0, 0, 0, 1, 0, 1];
% tx_bits  = [preamble, payload];
% 
% % 3. Create Synthetic BLE Signal (Simplified GFSK)
% % Create a frequency profile: +250kHz for '1', -250kHz for '0'
% freq_dev = 250e3;
% tx_freq_profile = zeros(1, length(tx_bits) * sps);
% 
% for i = 1:length(tx_bits)
%     val = (tx_bits(i) * 2) - 1; % Convert 0/1 to -1/+1
%     start_idx = (i-1)*sps + 1;
%     end_idx = i*sps;
%     tx_freq_profile(start_idx:end_idx) = val * freq_dev;
% end
% 
% % Smooth the frequency transitions (Gaussian-like filter effect)
% tx_freq_profile = smoothdata(tx_freq_profile, 'gaussian', sps);
% 
% % Integrate Frequency to get Phase -> Generate IQ Signal
% phase = cumsum(tx_freq_profile) / fs * 2 * pi;
% iq_signal = exp(1j * phase);
% 
% % Add a little noise
% noise = (randn(size(iq_signal)) + 1j*randn(size(iq_signal))) * 0.05;
% rx_signal = iq_signal + noise;
% 
% % Add 5 microseconds of silence (zeros) before the packet
% silence_samples = 5e-6 * fs; 
% rx_signal = [zeros(1,silence_samples), rx_signal];
% 
% % 4. Run the Decoder
% [decoded_sig, decoded_freq, decoded_bits] = BLE_Decoder(rx_signal.', fs, 1);
% 
% % 5. Visualization
% t = (0:length(decoded_freq)-1) / fs * 1e6; % Time in microseconds
% 
% figure('Color','w', 'Position', [100, 100, 800, 400]);
% plot(t, decoded_freq / 1e3, 'LineWidth', 1.5); hold on;
% yline(0, 'k--');
% xlabel('Time (\mus)');
% ylabel('Frequency (kHz)');
% title('MATLAB: Demodulated Frequency Profile');
% grid on;
% 
% % 6. Print Comparison
% disp('--- MATLAB Verification ---');
% disp(['Original Bits: ', num2str(tx_bits(1:20))]);
% disp(['Decoded Bits:  ', num2str(decoded_bits')]);

% ==========================================
% Test_BLE_Usage.m
% ==========================================
clear; clc; close all;

% --- Configuration ---
filename = fullfile('Example_Data', '1');
fs = 4e6;              % 4 MHz
preamble_detect = 1;

% ==========================================
% 1. Mock Data Generation (If file missing)
% ==========================================
if ~exist(filename, 'file')
    fprintf('File %s not found. Generating mock data...\n', filename);
    if ~exist('Example_Data', 'dir'); mkdir('Example_Data'); end
    
    % Create known bit sequence (Same as Python script)
    % Preamble (0,1...) + Payload
    mock_bits = [0,1,0,1,0,1,0,1,0,1,0,  1,1,0,0,1,0,1,0,1,1,1,0]; 
    sps = fs/1e6;
    
    % Modulate (250kHz deviation)
    % Map 0->-1, 1->1
    pulses = repelem(mock_bits * 2 - 1, sps) * 250e3;
    
    % Integrate to get Phase -> IQ
    phase = cumsum(pulses) / fs * 2 * pi;
    iq_sig = exp(1j * phase);
    
    % Add 5us silence at start (20 samples)
    silence = zeros(1, round(5e-6 * fs));
    iq_sig = [silence, iq_sig];
    
    % Interleave Real/Imag for binary saving [Re, Im, Re, Im...]
    interleaved = zeros(1, length(iq_sig)*2);
    interleaved(1:2:end) = real(iq_sig);
    interleaved(2:2:end) = imag(iq_sig);
    
    % Write to file as float32
    fid = fopen(filename, 'w');
    fwrite(fid, interleaved, 'float32');
    fclose(fid);
    fprintf('Mock data created successfully.\n\n');
end

% ==========================================
% 2. File Reading (Your Real Usage Logic)
% ==========================================
fprintf('Reading %s...\n', filename);

fid = fopen(filename, 'r');
[raw_data, ~] = fread(fid, 'float32');
fclose(fid);

% Reshape to (2, N) and transpose to (N, 2)
signal = reshape(raw_data, 2, []).';

% Combine into Complex IQ
signal = signal(:,1) + 1i * signal(:,2);

% Trim artifacts (Logic from your provided chain)
if length(signal) > 12
    signal = signal(1:end-12);
end

% ==========================================
% 3. Decode
% ==========================================
[ble_sig, freq_profile, bits] = BLE_Decoder(signal, fs, preamble_detect);

% ==========================================
% 4. Print Results
% ==========================================
fprintf('\n--- Decoding Results ---\n');
fprintf('Signal Length: %d samples\n', length(signal));
fprintf('Decoded Bits (%d):\n', length(bits));
disp(bits'); % Transpose to print as row

% ==========================================
% 5. Visualization
% ==========================================
t = (0:length(freq_profile)-1) / fs * 1e6; % Time in microseconds

figure('Color', 'w', 'Position', [100, 100, 900, 500]);
hold on; grid on;

% Plot Frequency Profile
plot(t, freq_profile / 1e3, 'b-', 'LineWidth', 1.5);

% Plot Bit Decision Points
sps = fs / 1e6;
bit_times = ((0:length(bits)-1) * sps + sps/2) / fs * 1e6;
bit_vals = (bits * 2 - 1) * 250; % Scale to +/- 250 for visibility
plot(bit_times, bit_vals, 'r.', 'MarkerSize', 15);

% Formatting
yline(0, 'k--');
yline(250, 'g:', 'LineWidth', 1.5);
yline(-250, 'g:', 'LineWidth', 1.5);
xlabel('Time (\mus)');
ylabel('Frequency (kHz)');
title(['BLE Signal Decoding: ' filename]);
legend('Demodulated Freq', 'Bit Decision Center');
ylim([-350 350]);