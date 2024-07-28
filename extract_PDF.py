# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:52:15 2024

@author: ghhh7
"""


import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,  norm
from gen_wind_Van_Der_Hoven import generate_frequencies
import pickle

def plot_density(data_lists,  figsize=(15, 4)):
    """
    Plot the density functions of multiple data lists in subplots.

    Parameters:
    data_lists : list of lists
        List of data lists to plot density functions for.
    titles : list of str
        List of titles for each subplot.
    figsize : tuple
        Size of the figure.
    """
    titles = ['Medium-Long Component', 'Turbulence', 'Wave']
    file_names = ['noise_pdfs/ml_pdf_nowave_1.pkl', 'noise_pdfs/turb_pdf_nowave_1.pkl', 'noise_pdfs/wave_pdf_nowave_1.pkl']
    
    fig, axs = plt.subplots(1, len(data_lists), figsize=figsize)
    
    for i, data_list in enumerate(data_lists):
        # Perform kernel density estimation
        density = gaussian_kde(data_list)
        xs = np.linspace(min(data_list), max(data_list), 1000)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        
        # Save the density function to a file
        pdf_values = density(xs)
        with open(file_names[i], 'wb') as f:
            pickle.dump((xs, pdf_values), f)
        
        # Plot the density function
        ax = axs[i]
        ax.plot(xs, density(xs), label='Density')
        ax.fill_between(xs, density(xs), alpha=0.5)
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.set_title(titles[i])
        ax.grid(True)

    x_norm = np.linspace(-4, 4, 1000)
    y_norm = norm.pdf(x_norm, 0, 1)
    axs[1].plot(x_norm, y_norm, label='N(0, 1) Normal Distribution', linestyle='--')
    axs[1].legend()
    plt.tight_layout()
    plt.show()
    
def plot_density_response(data_lists):
    """
    Plot the density functions of multiple data lists in subplots.

    Parameters:
    data_lists : list of lists
        List of data lists to plot density functions for.
    titles : list of str
        List of titles for each subplot.
    figsize : tuple
        Size of the figure.
    """
    titles = ['Medium-Long Component', 'Turbulence', 'Wave']
    units = ['Frequency (cycle/h)', 'Time (s)', 'Frequency (Cycle/s)']
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    
    for i, data_list in enumerate(data_lists):
        # Perform kernel density estimation
        density = gaussian_kde(data_list)
        xs = np.linspace(min(data_list), max(data_list), 1000)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        
        # Plot the density function
        ax = axs[0, i]
        ax.plot(xs, density(xs), label='Density')
        ax.fill_between(xs, density(xs), alpha=0.5)
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.set_title(titles[i])
        ax.grid(True)
    
        f_lm = generate_frequencies()[:31]
        t = list(range(1000))
        f = np.linspace(0.1, 0.28, 400) 
        
        x_axis = [f_lm, t, f]
        
        axs[1, i].plot(x_axis[i], data_list)
        axs[1, i].set_xlabel(units[i])
        axs[1, i].grid(True)
    
    # Plot N(0, 1) in axs[0, 1]
    # Plot N(0, 1) in axs[0, 1]
    x_norm = np.linspace(-4, 4, 1000)
    y_norm = norm.pdf(x_norm, 0, 1)
    axs[0, 1].plot(x_norm, y_norm, label='N(0, 1) Normal Distribution', linestyle='--')
    axs[0, 1].legend()

        
    plt.tight_layout()
    plt.show()


def generate_noise(seeds):
    
    # generate medium long component noise use the first seed
    state_before = np.random.get_state()
    np.random.seed(seeds[0])
    white_noise_ml = np.random.uniform(-np.pi, np.pi, 31) 
    np.random.set_state(state_before)
    
    
    # generate turbulence noise use the second seed
    state_before = np.random.get_state()
    np.random.seed(seeds[1])
    white_noise_turb = np.random.normal(0, 1, int(np.ceil(2000 / 180) * 180))[500:2000] # For turbulence component
    np.random.set_state(state_before)
    
    
    state_before = np.random.get_state()
    #wave_seed = np.random.randint(0, high=10**7)
    np.random.seed(seeds[2])
    random_phases = 2*np.pi*np.random.rand(400)
    np.random.set_state(state_before)
    
    return white_noise_ml, white_noise_turb, random_phases

def extract_seed_sets(file_path):
    with open(file_path, 'r') as file:
        data_string = file.read()
        
    # Regular expression to extract sets of seeds
    pattern = r'\[(\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, data_string)
    
    # Convert matches to a list of lists
    seed_sets = [[int(match[0]), int(match[1]), int(match[2])] for match in matches]
    
    return seed_sets

noise_ml = []
noise_turb = []
noise_wave = []

seeds = extract_seed_sets('reproduced_results/large_std/std_pitch_nowave_1.txt')

for seed in seeds:
    white_noise_ml, white_noise_turb, random_phases = generate_noise(seed)
    noise_ml.extend(white_noise_ml)
    noise_turb.extend(white_noise_turb)
    noise_wave.extend(random_phases)
    
plot_density([noise_ml, noise_turb, noise_wave])





