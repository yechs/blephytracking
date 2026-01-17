% generate_test_data.m
% Run this to get the "Golden" values for Python verification.

clc; clear; close all;
format long e; % Ensure high precision for copying

disp('-------------------------------------------------------');
disp('TEST 1: GFSK MODULATE');
disp('-------------------------------------------------------');

% 1. Setup GFSK Inputs
x_bits = [1; 0; 1; 1; 0];  % Simple bit pattern
freqsep = 500e3;
Fs = 2e6; % 2 samples per symbol

% 2. Run MATLAB function
% Ensure gfsk_modulate.m is in your path
y = gfsk_modulate(x_bits, freqsep, Fs);

% 3. Print Output for Python
disp('Expected "y" (Real part):');
disp(mat2str(real(y).', 8)); % Transpose to row for easy copying
disp('Expected "y" (Imag part):');
disp(mat2str(imag(y).', 8));


disp(' ');
disp('-------------------------------------------------------');
disp('TEST 2: FIT ELLIPSE');
disp('-------------------------------------------------------');

% 1. Setup Ellipse Inputs
% We create a perfect ellipse tilted by 45 degrees
t = linspace(0, 2*pi, 20)';
a_true = 5;
b_true = 2;
x0_true = 1;
y0_true = -1;
phi_true = pi/4; % 45 degrees

% Parametric equations
X = x0_true + a_true*cos(t)*cos(phi_true) - b_true*sin(t)*sin(phi_true);
Y = y0_true + a_true*cos(t)*sin(phi_true) + b_true*sin(t)*cos(phi_true);

% 2. Run MATLAB function
ellipse_t = fit_ellipse(X, Y);

% 3. Print Output for Python
disp('Expected Parameters:');
fprintf('a: %.8f\n', ellipse_t.a);
fprintf('b: %.8f\n', ellipse_t.b);
fprintf('phi: %.8f\n', ellipse_t.phi);
fprintf('X0: %.8f\n', ellipse_t.X0);
fprintf('Y0: %.8f\n', ellipse_t.Y0);
