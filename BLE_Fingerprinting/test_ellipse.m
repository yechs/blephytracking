% 1. Define the Data (Same numbers as Python)
x = [13.5852; 12.8709; 11.9862; 10.9021; 9.4736; 8.2752; 7.4426; 6.6417; ...
     6.1706; 6.3342; 6.6991; 7.5171; 8.6701; 9.6836; 10.9449; 12.1826; ...
     13.0164; 13.6901; 13.7124; 13.3943];
y = [23.6821; 23.7806; 23.6654; 22.9752; 22.1844; 21.1285; 19.7598; 18.6835; ...
     17.5036; 16.7163; 16.2197; 16.4027; 16.5636; 17.1789; 18.3809; 19.3749; ...
     20.7707; 21.7254; 22.7520; 23.5552];

% 2. Call the function
% Note: We pass an empty axis_handle [] if we don't want to plot immediately, 
% or use gca to plot on the current figure.
% 2. Create Figure and Plot Data Points
figure;
scatter(x, y, 50, 'b', 'filled'); % Plot blue dots for raw data
hold on;                          % Keep the dots!

% 3. Call the function
% Passing 'gca' tells the function to draw the red ellipse on the current plot
result = fit_ellipse(x, y, gca);

% 4. Formatting for better visualization
axis equal;        % CRITICAL: Ensures the ellipse doesn't look distorted
grid on;
title('Least Squares Ellipse Fit');
xlabel('X Coordinate');
ylabel('Y Coordinate');

% Note: fit_ellipse draws 3 lines (major axis, minor axis, and perimeter).
% We add a dummy plot to make the legend look correct (Blue Dot vs Red Line)
h1 = plot(nan, nan, 'bo', 'MarkerFaceColor', 'b'); 
h2 = plot(nan, nan, 'r-');
legend([h1, h2], 'Data Points', 'Fitted Ellipse');

% 5. Print Results
fprintf('\n--- MATLAB Results ---\n');
if isempty(result.status)
    fprintf('Center (X0, Y0): (%.4f, %.4f)\n', result.X0, result.Y0);
    fprintf('Axes (a, b):     (%.4f, %.4f)\n', result.a, result.b);
    fprintf('Angle (phi):     %.4f rad\n', result.phi);
else
    fprintf('Fit failed or not an ellipse.\n');
end