% Manual test for fit_ellipse
rng(0);
a = 3;
b = 2;
phi = 0.3;
x0 = 1.5;
y0 = -2;

theta = linspace(0, 2*pi, 200);
x = a * cos(theta);
y = b * sin(theta);
R = [cos(phi) -sin(phi); sin(phi) cos(phi)];
pts = R * [x; y];

x_rot = pts(1,:) + x0 + 1e-3 * randn(1, numel(theta));
y_rot = pts(2,:) + y0 + 1e-3 * randn(1, numel(theta));

ell = fit_ellipse(x_rot', y_rot');

assert(strcmp(ell.status, ''));
assert(abs(ell.X0 - x0) < 0.05);
assert(abs(ell.Y0 - y0) < 0.05);

disp('test_fit_ellipse passed');
