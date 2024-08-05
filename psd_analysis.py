# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:14:28 2024

@author: ghhh7
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_psd(time_series, time_step=0.5, nperseg=256):
    """
    Plots the Power Spectrum Density (PSD) of a time series with constant time steps.
    
    Parameters:
    time_series (numpy array): The time series data.
    time_step (float): The time step between consecutive samples.
    nperseg (int): Length of each segment for Welch's method. Default is 256.
    """
    # Calculate the sampling frequency
    fs = 1 / time_step
    
    # Compute the Power Spectrum Density using Welch's method
    frequencies, psd = welch(time_series, fs, nperseg=nperseg)
    
    return frequencies, psd
    
    
    
def load_data(seeds):

    
    output_file_name = f'{seeds[0]}_{seeds[1]}_{seeds[2]}.npz'

    data = np.load(f'reproduced_results/data/{output_file_name}', allow_pickle=True)
    
    # Extracting the data
    t = data['t'][:-1]#[:-1000]
    state = data['x'][:-1]#[:-1000]
    beta = np.rad2deg(data['betas'])#[:-1000]
    x = data['x'][:-1][:-1000]
    wind_speed = data['v_wind'][:-1]#[:-1000]
    wave_eta = data['wave_eta'][:-1]#[:-1000]
    T_E = data['T_E'][:-1]#[:-1000]
    P_A = data['P_A'][:-1]#[:-1000]
    data.close()
    
    return state, wind_speed, wave_eta
    
#11 m/s
seeds_11 = [5922703, 7807870, 9240304]

#20 m/s
seeds_20 = [5563693, 5004676, 9608067]
seeds_20 = [8734247, 7239128, 3868119]

#large surge
seeds_large = [8734247, 7239128, 3868119]

state_11, wind_11, wave_11 = load_data(seeds_11)
state_20, wind_20, wave_20 = load_data(seeds_20)

f_wind_11, psd_wind_11 = plot_psd(wind_11, 0.5)
f_wave_11, psd_wave_11 = plot_psd(wave_11, 0.5)
f_surge_11, psd_surge_11 = plot_psd(state_11[:, 0], 0.5)



f_wind_20, psd_wind_20 = plot_psd(wind_20, 0.5)
f_wave_20, psd_wave_20 = plot_psd(wave_20, 0.5)
f_surge_20, psd_surge_20 = plot_psd(state_20[:, 0], 0.5)

def load_psd_percentiles(file_name):
    """
    Loads the PSD percentiles from a numpy file.
    
    Parameters:
    file_name (str): The name of the file to load the percentiles from.
    
    Returns:
    dict: A dictionary containing the median, 12.5th percentile, 87.5th percentile,
          37.5th percentile, and 62.5th percentile PSD values.
    """
    data = np.load(file_name)

    return {
        'median': data['median'],
        'p12_5': data['p12_5'],
        'p87_5': data['p87_5'],
        'p37_5': data['p37_5'],
        'p62_5': data['p62_5']
    }

# Load the PSD percentiles for each category
psd_wind_0_10 = load_psd_percentiles("psd_parallel/wind_0_10.npz")
psd_wind_20 = load_psd_percentiles("psd_parallel/wind_20.npz")

psd_wave_0_10 = load_psd_percentiles("psd_parallel/wave_0_10.npz")
psd_wave_20 = load_psd_percentiles("psd_parallel/wave_20.npz")

psd_surge_0_10 = load_psd_percentiles("psd_parallel/surge_0_10.npz")
psd_surge_20 = load_psd_percentiles("psd_parallel/surge_20.npz")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

def plot_psd_with_percentiles(ax, frequencies, median, p12_5, p87_5, p37_5, p62_5, label, color):
    ax.semilogy(frequencies, median, label=label, color=color)
    ax.fill_between(frequencies, p12_5, p87_5, color=color, alpha=0.3)
    ax.fill_between(frequencies, p37_5, p62_5, color=color, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.set_xscale('log')
    ax.grid(True)


# Assume that the frequencies are the same for all PSDs
frequencies = np.linspace(0.001, 1, len(psd_wind_0_10['median']))  # Replace with actual frequencies

# Create subplots

# Plot PSD of Wind
plot_psd_with_percentiles(axs[0], frequencies, psd_wind_0_10['median'], psd_wind_0_10['p12_5'], psd_wind_0_10['p87_5'], psd_wind_0_10['p37_5'], psd_wind_0_10['p62_5'], 'Wind Speed < 10 m/s', 'blue')
plot_psd_with_percentiles(axs[0], frequencies, psd_wind_20['median'], psd_wind_20['p12_5'], psd_wind_20['p87_5'], psd_wind_20['p37_5'], psd_wind_20['p62_5'], 'Wind Speed > 20 m/s', 'red')
axs[0].set_title('PSD of Wind')
axs[0].legend()

# Plot PSD of Wave
plot_psd_with_percentiles(axs[1], frequencies, psd_wave_0_10['median'], psd_wave_0_10['p12_5'], psd_wave_0_10['p87_5'], psd_wave_0_10['p37_5'], psd_wave_0_10['p62_5'], 'Wind Speed < 10 m/s', 'blue')
plot_psd_with_percentiles(axs[1], frequencies, psd_wave_20['median'], psd_wave_20['p12_5'], psd_wave_20['p87_5'], psd_wave_20['p37_5'], psd_wave_20['p62_5'], 'Wind Speed > 20 m/s', 'red')
axs[1].set_title('PSD of Wave')
axs[1].legend()

# Plot PSD of Surge
plot_psd_with_percentiles(axs[2], frequencies, psd_surge_0_10['median'], psd_surge_0_10['p12_5'], psd_surge_0_10['p87_5'], psd_surge_0_10['p37_5'], psd_surge_0_10['p62_5'], 'Wind Speed < 10 m/s', 'blue')
plot_psd_with_percentiles(axs[2], frequencies, psd_surge_20['median'], psd_surge_20['p12_5'], psd_surge_20['p87_5'], psd_surge_20['p37_5'], psd_surge_20['p62_5'], 'Wind Speed > 20 m/s', 'red')
axs[2].set_title('PSD of Surge')
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()



