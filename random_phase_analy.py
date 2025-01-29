# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 03:37:41 2024

@author: Yihan Liu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle


def plot_pdf(ite):
    large_random_phase = np.load(f"reproduced_results/large_noise/extreme_large_noise_MCMC_ite{ite}.npy")
    
    
    for i in range(large_random_phase.shape[0]):
        data = large_random_phase[i]
        
        # KDE for PDF estimation
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        pdf = kde(x_range)
    
        # Plotting
        plt.figure()
        plt.plot(x_range, pdf, label=f"Phase {i+1}")
        plt.title(f"PDF for Phase {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid()
        plt.savefig(f"random_phase_distr/pdf/ite{ite}/phase_pdf_{i+1}.png")  # Save each plot
        plt.close()
        

def get_pdf_value(phase_value, data):
    """
    Computes the PDF value for a given phase value using the saved KDE for the specified index.

    Parameters:
        phase_value (float): The random phase number for which to compute the PDF value.
        index (int): The index of the saved KDE to use for evaluation.
        iteration (int): The iteration number to locate the saved KDE files.

    Returns:
        float: The PDF value for the given phase value.
    """

        
    kde = gaussian_kde(data)
    
    # Compute the PDF value
    pdf_value = kde.evaluate(phase_value)[0]
    
    return pdf_value        


def compute_weight(ite):
    past_phase = np.load(f"reproduced_results/large_noise/extreme_large_noise_MCMC.npy")
    
    large_random_phase = np.load(f"reproduced_results/large_noise/extreme_large_noise_MCMC_ite{ite}.npy")
    
    
    # For sample
    weight_sum = 0
    for i in range(large_random_phase.shape[1]):
        
        weight_per_sample = 1
        random_phase = large_random_phase[:, i]
        # in each sample, for each random phase
        for j in range(random_phase.shape[0]):
            kde = gaussian_kde(past_phase[j])
            pdf = kde(random_phase[j])
            weight_per_sample *= 1 / (2*np.pi*pdf)
            
            print(weight_per_sample)
            
        weight_sum += weight_per_sample
        
    R = weight_sum / 5000
    
    print(R)
    
compute_weight(1)
    
    
    

        
        