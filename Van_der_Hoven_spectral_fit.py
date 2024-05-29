# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:07:10 2024

@author: Yihan Liu
"""

import numpy as np
from scipy.interpolate import splrep
import pickle
from scipy.interpolate import splev
import matplotlib.pyplot as plt

# Define your data points
x = np.array([-3, -2.6, -2.35, -2, -1.7, -1.5, -1.3, -1.1, -0.9, -0.5, 0.3, 0.6, 1])#, 1.3, 1.5, 1.7, 2, 2.5, 3])
y = np.array([0.55, 1, 2, 4.5, 3.2, 1.6, 1.4, 1.8, 1, 0.4, 0.2, 0.3, 0.5])#, 0.7, 2.1, 3, 1.9, 1, 0.5])

# Fit a spline to the data points
tck = splrep(x, y)

# Save the tck tuple to a file
with open('Van_der_Hoven_spectral_fit.pkl', 'wb') as f:
    pickle.dump(tck, f)
    
# Generate a range of x values for a smooth plot
x_fit = np.linspace(-3, 1, 100)

# Evaluate the spline at the points in x_fit
y_fit = splev(x_fit, tck)

# Plot the original data points and the fitted spline
plt.plot(x_fit, y_fit, 'b-', linewidth=2, label='Spline Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spline Fit to Data Points')
plt.grid(True)
plt.show()