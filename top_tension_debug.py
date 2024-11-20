# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:22:32 2024

@author: ghhh7
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_top_sample_data(index):
    """
    Load and plot the simulation data for the specified index from top_samples.npz.

    Parameters:
    ----------
    index : int
        The 1-based index (1-15) specifying which dataset to load from top_samples.
    top_samples_file : str
        Path to the top samples .npz file.
    figure_directory : str
        Directory where figures should be saved.
    """
    # Ensure the index is within the valid range
    if index < 1 or index > 15:
        raise ValueError("Index must be between 1 and 15.")
    
    # Load the top samples data
    data = np.load(f'reproduced_results/data/top_tension.npz', allow_pickle=True)
    
    # Extract the data for the specified index
    t = data['t'][:-1]
    state = data['state'][:, :, index - 1][:-1]  # Adjust for 1-based index
    wind_speed = data['wind_speed'][:, index - 1][:-1]
    wave_eta = data['wave_eta'][:, index - 1]
    rope_tension = data['rope_tension'][:, :, index - 1][:-1]
    print(np.max(rope_tension[:, 0]), np.argmax(rope_tension[:, 0]))
    data.close()
    
    percentile_file_path = 'reproduced_results/percentile_extreme.npz'
    data = np.load(percentile_file_path)

    percentile_87_5 = data['percentile_87_5'][:-1]
    percentile_12_5 = data['percentile_12_5'][:-1]

    percentile_62_5 = data['percentile_62_5'][:-1]
    percentile_37_5 = data['percentile_37_5'][:-1]

    percentile_50 = data['percentile_50'][:-1]

    max_state = data['max_state'][:-1]
    min_state = data['min_state'][:-1]
    data.close()

    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor Speed (rpm)']

    def plot_helper(ax):
        """Helper function to generate plots."""
        # Plot wind speed
        ax[0].plot(t, wind_speed, color='black', linewidth=0.5)
        ax[0].set_title('Wind Speed (m/s)', fontsize=15)
        ax[0].grid(True)

        # Plot wave elevation
        ax[1].plot(t, wave_eta, color='black', linewidth=0.5)
        ax[1].set_title('Wave Elevation (m)', fontsize=15)
        ax[1].grid(True)
        
        # Plot state variables with percentiles
        for j in range(4):
            ax[j+2].plot(t, state[:, j*2], color='black', linewidth=0.5)
            ax[j+2].fill_between(t, percentile_12_5[:, j*2], percentile_87_5[:, j*2], color='b', alpha=0.3, edgecolor='none')
            ax[j+2].fill_between(t, percentile_37_5[:, j*2], percentile_62_5[:, j*2], color='b', alpha=0.3, edgecolor='none')
            ax[j+2].plot(t, percentile_50[:, j*2], color='r', alpha=0.9, linewidth=0.5)
            ax[j+2].set_title(state_names[j*2], fontsize=15)
            ax[j+2].grid(True)

        # Plot rotor thrust


        # Add legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=1, label='Sample Trajectory'),
            Line2D([0], [0], color='r', lw=1, label='Median'),
            Line2D([0], [0], color='b', lw=8, alpha=0.6, label='Central 25th Percentile'),
            Line2D([0], [0], color='b', lw=8, alpha=0.3, label='Central 75th Percentile')
        ]
        ax[7].legend(handles=legend_elements, loc='center', fontsize=15)
        ax[7].axis('off')

    # Create plot
    fig, ax = plt.subplots(2, 4, figsize=(24, 6))
    ax = ax.flatten()
    plot_helper(ax)
    plt.tight_layout() 

    # Save and show plot

    plt.show()
    plt.close(fig)
    
    return np.std(state[:, 4])

# Example usage to load and plot the 3rd top sample:
#load_top_sample_data(1)

data = np.load(f'reproduced_results/data/top_tension.npz', allow_pickle=True)
seeds = data['seeds'][0]

sample_count = len(seeds)
    
    # Prepare noise array to store generated noise for each sample
noise_array = np.empty((31, sample_count))


    
for i, seed in enumerate(seeds):
    # Set the seed to the first component
    state_before = np.random.get_state()                                                                                                                                                                 
    np.random.seed(seed)  # Use only the first component
    white_noise_ml = np.random.uniform(-np.pi, np.pi, 31) 
    np.random.set_state(state_before)
    
    # Save generated noise in the noise array
    noise_array[:, i] = white_noise_ml

np.save("imps_tension_ml_pi0_ite0.npy", noise_array)

random_index = np.random.randint(0, 15)
sampling_source = np.load(f"imps_tension_ml_pi0_ite0.npy")[:, random_index]























