# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:49:49 2024

@author: ghhh7
"""
import numpy as np
from gen_wind_Van_Der_Hoven import generate_wind
import matplotlib.pyplot as plt

def pierson_moskowitz_spectrum(U19_5, zeta, eta, t, random_phases):
    """
    This function generates the Pierson-Moskowitz spectrum for a given wind speed U10 and frequency f.
    
    parameters
    ----------
    U19_5 : float
        the average wind speed at 19.5m above the sea surface
    zeta : float
        the x component to evaluate
    eta : float
        the y component to evaluate. (Note: the coordinate system here is different
                                      from the Betti model. The downward is negative
                                      in this case)
    t: float
        the time to evaluate.
    random_phase : Numpy Array
        the random phase to generate wave. Should be in [0, 2*pi)

    Returns
    -------
    wave_eta : float
        The wave elevation
    [v_x, v_y, a_x, a_y]: list
        The wave velocity and acceleration in x and y direction
    """
    g = 9.81  # gravitational constant
    alpha = 0.0081  # Phillips' constant

    f_pm = 0.14*(g/U19_5)  # peak frequency
    
    N = 400
    
    cutof_f = 3*f_pm # Cutoff frequency
    
    f = np.linspace(0.1, cutof_f, N) # Array
    omega = 2*np.pi*f # Array
    delta_f = f[1] - f[0] # Array

    S_pm = (alpha*g**2/((2*np.pi)**4*f**5))*np.exp(-(5/4)*(f_pm/f)**4) # Array
    
    a = np.sqrt(2*S_pm*delta_f)
    k = omega**2/g    
    
    
    # Perform the calculations in a vectorized manner
    sin_component = np.sin(omega*t - k*zeta + random_phases)
    cos_component = np.cos(omega*t - k*zeta + random_phases)
    exp_component = np.exp(k*eta)
    
    wave_eta = np.sum(a * sin_component)
    
    v_x = np.sum(omega * a * exp_component * sin_component)
    v_y = np.sum(omega * a * exp_component * cos_component)
    
    a_x = np.sum((omega**2) * a * exp_component * cos_component)
    a_y = -np.sum((omega**2) * a * exp_component * sin_component)
    
    
    
    return wave_eta


# Parameters
T_s1 = 180  # Sampling period (seconds)
T_total = 1500  # Total simulation time (seconds)
v_bar = 11  # Mean wind speed (m/s)

L = 180  # Turbulence length scale in meters
k_sigma_v = 0.13 # Slope parameter
T_s = 1  # Time step in seconds

# Generate white noise for each segment
import random
seeds = [random.randint(0, 9999999) for _ in range(3)]
#seeds = [3348203, 8879543, 6120326]
# Print the list of random numbers
print(seeds)
# generate medium long component noise use the first seed
state_before = np.random.get_state()
np.random.seed(seeds[0])
white_noise_ml = np.random.uniform(-np.pi, np.pi, 31) 
np.random.set_state(state_before)

# generate turbulence noise use the second seed
state_before = np.random.get_state()
np.random.seed(seeds[1])
white_noise_turb = np.random.normal(0, 1, int(np.ceil(T_total / T_s1) * T_s1))  # For turbulence component
np.random.set_state(state_before)

#white_noise_ml = np.random.uniform(-np.pi, np.pi, 31)  # For phase in medium-long term
#white_noise_turb = np.random.normal(0, 1, int(np.ceil(T_total / T_s1) * T_s1))  # For turbulence component

# Generate large array of wind speeds with turbulence
wind_speeds, v_ml = generate_wind(v_bar, L, k_sigma_v, T_s, T_s1, T_total, white_noise_ml, white_noise_turb)

# Plot the simulated wind speeds
plt.plot(np.arange(0, len(wind_speeds), T_s), wind_speeds, 'b-', linewidth=0.5, label='Wind Speed')
plt.xlabel('Time (s)')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed')

plt.grid(True)
plt.show()
