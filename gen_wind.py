# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:26:25 2024

Wind generator

@author: Yihan Liu
"""
import matplotlib.pyplot as plt
import numpy as np

def von_Karman_power_spectrum(sigma, L, v_bar, omega):
    """
    Compute Von Karman Power Spectrum

    Parameters
    ----------
    sigma : float
        the turbulence intensity
    L : float
        the turbulence length scale
    v_bar : float
        Average wind speed
    omega : float
        the discrete angular frequency 

    Returns
    -------
    float
        Computed power spectral density for given omega.
    """
    return 0.475 * sigma**2 * (L/v_bar) / ((1 + (omega*L/v_bar)**2)**(5/6))


    
def generate_frequencies(N = 9, k_range = (-3, 2), base=10):
    """
    Generate discrete frequencies for the simulation.

    Parameters
    ----------
    N : int
        Number of frequencies per decade
    k_range : tuple
        Range of decades (min_k, max_k)

    Returns
    -------
    list
        List of frequencies (in cycles/hour)
    """
    frequencies = []
    for k in range(k_range[0], k_range[1] + 1):
        frequencies.extend([i * (base**k) for i in range(1, N+1)])
    return frequencies

def simulate_wind_speed(T_s1, T_F, v_bar, sigma, L):
    """
    Simulate the medium- and long-term wind speed component.

    Parameters
    ----------
    N : int
        Number of frequencies per decade
    k_range : tuple
        Range of decades (min_k, max_k)
    T_s1 : float
        Sampling period for medium-long term component (seconds)
    T_F : float
        Total simulation time (seconds)
    v_bar : float
        Mean wind speed
    sigma : float
        Turbulence intensity
    L : float
        Turbulence length scale

    Returns
    -------
    np.array
        Simulated wind speeds over time
        
        An array of medium long wind speed at each time step.
        i.e. simulating for 600 seconds with time step 60 seconds, 
        will return an array of length 10.
        
    """
    frequencies = generate_frequencies()[:30]
    omegas = np.array(frequencies) * 2 * np.pi / 3600  # Convert cycles/hour to rad/second
    amplitudes = []
    phases = np.random.uniform(-np.pi, np.pi, size=len(omegas))
    
    # Calculate amplitudes using the power spectrum
    for i in range(len(omegas) - 1):
        Svv_i = von_Karman_power_spectrum(sigma, L, v_bar, omegas[i])
        Svv_ip1 = von_Karman_power_spectrum(sigma, L, v_bar, omegas[i+1])
        amplitude = 2 * np.sqrt((Svv_i + Svv_ip1) * (omegas[i+1] - omegas[i]) / np.pi)
        amplitudes.append(amplitude)
    
    amplitudes = np.array(amplitudes)
    times = np.arange(0, T_F, T_s1)
    wind_speeds = v_bar + sum(a * np.cos(o * times + p) for a, o, p in zip(amplitudes, omegas[:-1], phases))
    
    return wind_speeds



# Parameters
T_s1 = 60  # Sampling every 60 seconds
T_F = 600  # Simulate for 5 hours (18000 seconds)
v_bar = 20  # Mean wind speed 10 m/s
sigma = 14  # Turbulence intensity
L = 200  # Turbulence length scale

# Generate wind speeds
wind_speeds = simulate_wind_speed(T_s1, T_F, v_bar, sigma, L)
print(wind_speeds)
    


times = np.arange(0, T_F, T_s1)
# Plotting the wind speeds
plt.figure(figsize=(10, 6))
plt.plot(times / 3600, wind_speeds, label='Wind Speed', color='blue')  # times/3600 to convert seconds to hours
plt.xlabel('Time (hours)')
plt.ylabel('Wind Speed (m/s)')
plt.title('Simulated Wind Speed Over Time')
plt.grid(True)
plt.legend()
plt.show()









