x = [-3, -2.6, -2.35, -2, -1.7, -1.5, -1.3, -1.1, -0.9, -0.5, 0.3, 0.6, 1];
y = [0.55, 1, 2, 4.5, 3.2, 1.6, 1.4, 1.8, 1, 0.4, 0.2, 0.3, 0.5];

% Fit a spline to the data points
pp = spline(x, y);

% Generate a range of x values for a smooth plot
x_fit = linspace(min(x), max(x), 100);

% Evaluate the spline at the points in x_fit
y_fit = ppval(pp, x_fit);

% Plot the original data points and the fitted spline
figure;
plot(x, y, 'ro', 'MarkerSize', 10, 'DisplayName', 'Data Points'); % Original data points
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Spline Fit');
xlabel('x');
ylabel('y');
legend show;
title('Spline Fit to Data Points');
grid on;
hold off;