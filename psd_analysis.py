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


seeds_11 = [5922703, 7807870, 9240304]
state, wind_speed, wave_eta = load_data(seeds_11)
frequencies_wind, psd = plot_psd(wind_speed)
frequencies_surge, psd_1 = plot_psd(state[:, 0])
frequencies_wave, psd_2 = plot_psd(wave_eta)

'''
#11 m/s
seeds_11 = [5922703, 7807870, 9240304]

#20 m/s
seeds_20 = [5563693, 5004676, 9608067]

#large surge
seeds_large = [8734247, 7239128, 3868119]

state_11, wind_11, wave_11 = load_data(seeds_11)
state_20, wind_20, wave_20 = load_data(seeds_20)
state_large, wind_large, wave_large = load_data(seeds_large)

f_wind_large, psd_wind_large = plot_psd(wind_large, 0.5)
f_wave_large, psd_wave_large = plot_psd(wave_large, 0.5)
f_surge_large, psd_surge_large = plot_psd(state_large[:, 0][600:1600], 0.5)

f_wind_11, psd_wind_11 = plot_psd(wind_11, 0.5)
f_wave_11, psd_wave_11 = plot_psd(wave_11, 0.5)
f_surge_11, psd_surge_11 = plot_psd(state_11[:, 0], 0.5)



f_wind_20, psd_wind_20 = plot_psd(wind_20, 0.5)
f_wave_20, psd_wave_20 = plot_psd(wave_20, 0.5)
f_surge_20, psd_surge_20 = plot_psd(state_20[:, 0], 0.5)
'''
#############################################################################


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

def plot_psd_with_percentiles(ax, frequencies, median, p12_5, p87_5, p37_5, p62_5, label, color):
    ax.semilogy(frequencies, median, label=label, color=color)
    ax.fill_between(frequencies, p12_5, p87_5, color=color, alpha=0.3)
    ax.fill_between(frequencies, p37_5, p62_5, color=color, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.set_xscale('log')
    ax.grid(True)
    
'''

psd_wind_MCMC = load_psd_percentiles("psd/wind_MCMC.npz")
psd_wind_pi0 = load_psd_percentiles("psd/wind_pi0.npz")
psd_wind_extreme = load_psd_percentiles("psd/wind_extreme.npz")

psd_wave_MCMC = load_psd_percentiles("psd/wave_MCMC.npz")
psd_wave_pi0 = load_psd_percentiles("psd/wave_pi0.npz")
psd_wave_extreme = load_psd_percentiles("psd/wave_extreme.npz")

psd_surge_MCMC = load_psd_percentiles("psd/surge_MCMC.npz")
psd_surge_pi0 = load_psd_percentiles("psd/surge_pi0.npz")
psd_surge_extreme = load_psd_percentiles("psd/surge_extreme.npz")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Create a dummy plot for the legend
dummy_median = axs[0].plot([], [], label='Median', color='black', linestyle='-')
dummy_central_25 = axs[0].fill_between([], [], [], color='black', alpha=0.5, label='Central 25%')
dummy_75_percentile = axs[0].fill_between([], [], [], color='black', alpha=0.3, label='Central 75%')

# Plot PSD of Wind for MCMC, pi0, and Extreme
plot_psd_with_percentiles(axs[0], frequencies_wind, psd_wind_MCMC['median'], psd_wind_MCMC['p12_5'], psd_wind_MCMC['p87_5'], psd_wind_MCMC['p37_5'], psd_wind_MCMC['p62_5'], 'Standard MCMC', 'blue')
plot_psd_with_percentiles(axs[0], frequencies_wind, psd_wind_pi0['median'], psd_wind_pi0['p12_5'], psd_wind_pi0['p87_5'], psd_wind_pi0['p37_5'], psd_wind_pi0['p62_5'], r'IMPS $\epsilon=0$', 'green')
plot_psd_with_percentiles(axs[0], frequencies_wind, psd_wind_extreme['median'], psd_wind_extreme['p12_5'], psd_wind_extreme['p87_5'], psd_wind_extreme['p37_5'], psd_wind_extreme['p62_5'], 'Extreme Surge > 9m', 'red')
axs[0].set_title('PSD of Wind')
axs[0].legend()

# Plot PSD of Wave for MCMC, pi0, and Extreme
plot_psd_with_percentiles(axs[1], frequencies_wave, psd_wave_MCMC['median'], psd_wave_MCMC['p12_5'], psd_wave_MCMC['p87_5'], psd_wave_MCMC['p37_5'], psd_wave_MCMC['p62_5'], 'Wave MCMC', 'blue')
plot_psd_with_percentiles(axs[1], frequencies_wave, psd_wave_pi0['median'], psd_wave_pi0['p12_5'], psd_wave_pi0['p87_5'], psd_wave_pi0['p37_5'], psd_wave_pi0['p62_5'], 'Wave pi0', 'green')
plot_psd_with_percentiles(axs[1], frequencies_wave, psd_wave_extreme['median'], psd_wave_extreme['p12_5'], psd_wave_extreme['p87_5'], psd_wave_extreme['p37_5'], psd_wave_extreme['p62_5'], 'Wave Extreme', 'red')
axs[1].set_title('PSD of Wave')
#axs[1].legend()

# Plot PSD of Surge for MCMC, pi0, and Extreme
plot_psd_with_percentiles(axs[2], frequencies_surge, psd_surge_MCMC['median'], psd_surge_MCMC['p12_5'], psd_surge_MCMC['p87_5'], psd_surge_MCMC['p37_5'], psd_surge_MCMC['p62_5'], 'Surge MCMC', 'blue')
plot_psd_with_percentiles(axs[2], frequencies_surge, psd_surge_pi0['median'], psd_surge_pi0['p12_5'], psd_surge_pi0['p87_5'], psd_surge_pi0['p37_5'], psd_surge_pi0['p62_5'], 'Surge pi0', 'green')
plot_psd_with_percentiles(axs[2], frequencies_surge, psd_surge_extreme['median'], psd_surge_extreme['p12_5'], psd_surge_extreme['p87_5'], psd_surge_extreme['p37_5'], psd_surge_extreme['p62_5'], 'Surge Extreme', 'red')
axs[2].set_title('PSD of Surge')
#axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()

'''
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
# Load the PSD percentiles for each category
psd_wind_0_10 = load_psd_percentiles("psd_parallel/wind_0_10.npz")
psd_wind_20 = load_psd_percentiles("psd_parallel/wind_20.npz")

psd_wave_0_10 = load_psd_percentiles("psd_parallel/wave_0_10.npz")
psd_wave_20 = load_psd_percentiles("psd_parallel/wave_20.npz")

psd_surge_0_10 = load_psd_percentiles("psd_parallel/surge_0_10.npz")
psd_surge_20 = load_psd_percentiles("psd_parallel/surge_20.npz")


# Assume that the frequencies are the same for all PSDs
#frequencies = np.linspace(0.001, 1, len(psd_wind_0_10['median']))  # Replace with actual frequencies

# Create subplots

# Create a dummy plot for the legend
dummy_median = axs[0].plot([], [], label='Median', color='black', linestyle='-')
dummy_central_25 = axs[0].fill_between([], [], [], color='black', alpha=0.5, label='Central 25%')
dummy_75_percentile = axs[0].fill_between([], [], [], color='black', alpha=0.3, label='Central 75%')

# Plot PSD of Wind
plot_psd_with_percentiles(axs[0], frequencies_wind, psd_wind_0_10['median'], psd_wind_0_10['p12_5'], psd_wind_0_10['p87_5'], psd_wind_0_10['p37_5'], psd_wind_0_10['p62_5'], 'Mean wind speed < 12 m/s', 'blue')
plot_psd_with_percentiles(axs[0], frequencies_wind, psd_wind_20['median'], psd_wind_20['p12_5'], psd_wind_20['p87_5'], psd_wind_20['p37_5'], psd_wind_20['p62_5'], 'Mean wind speed > 12 m/s', 'red')
axs[0].set_title('PSD of Wind')
#ax[0].plot(f_wind_large, psd_wind_large, color='black', label='Extreme Surge')
axs[0].legend()

# Plot PSD of Wave
plot_psd_with_percentiles(axs[1], frequencies_wave, psd_wave_0_10['median'], psd_wave_0_10['p12_5'], psd_wave_0_10['p87_5'], psd_wave_0_10['p37_5'], psd_wave_0_10['p62_5'], 'Wind Speed < 10 m/s', 'blue')
plot_psd_with_percentiles(axs[1], frequencies_wave, psd_wave_20['median'], psd_wave_20['p12_5'], psd_wave_20['p87_5'], psd_wave_20['p37_5'], psd_wave_20['p62_5'], 'Wind Speed > 20 m/s', 'red')
axs[1].set_title('PSD of Wave')
#axs[1].plot(f_wave_large, psd_wave_large, color='black')
#axs[1].legend()

# Plot PSD of Surge
plot_psd_with_percentiles(axs[2], frequencies_surge, psd_surge_0_10['median'], psd_surge_0_10['p12_5'], psd_surge_0_10['p87_5'], psd_surge_0_10['p37_5'], psd_surge_0_10['p62_5'], 'Wind Speed < 10 m/s', 'blue')
plot_psd_with_percentiles(axs[2], frequencies_surge, psd_surge_20['median'], psd_surge_20['p12_5'], psd_surge_20['p87_5'], psd_surge_20['p37_5'], psd_surge_20['p62_5'], 'Wind Speed > 20 m/s', 'red')
axs[2].set_title('PSD of Surge')
#axs[2].plot(f_surge_large, psd_surge_large, color='black')
#axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()



