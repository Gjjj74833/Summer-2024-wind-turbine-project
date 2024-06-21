# -*- coding: utf-8 -*-
"""
Generate turbulent component

@author: Yihan Liu
"""
import numpy as np
import matplotlib.pyplot as plt


def gen_turbulence(v_bar, L, k_sigma_v, T_s, t_final, white_noise, 
                   delta_omega = 0.002, M = 5000, N = 100):
    """
    Generate turbulencec component for wind speed

    Parameters
    ----------
    v_bar : int
        Average wind speed
    L : int
        Turbulence length
    k_sigma_v : float
        Slope parameter
    T_s : int
        Time step
    t_final : int
        Sampling time interval
    white_noise : np.array
        the white noise with mean = 0 and std = 1, has length 

    Returns
    -------
    Array of wind speed with turbulence

    """
    
    # Step 1: Update the current values of the parameters in 
    # the turbulence component model
    T_F = L / v_bar
    sigma_v = k_sigma_v * v_bar
    K_F = np.sqrt((2 * np.pi * T_F)/(4.20654 * T_s))
    
    # Step 2: Calculate the discrete impulse response of the filter
    #delta_omega = 0.002 # Frequency step size
    #M = 5000 # Number of frequency points
    #N = 100 # Numerical parameters for convolution integration, divide 
            # finite integral from 0 to t to N regions
    
    # Discrete frequency domain P(omega)
    P = np.zeros(M + 1)
    for r in range(M + 1):
        P[r] = np.real(K_F / (1 + 1j * r * delta_omega * T_F)**(5/6))
    
    # Discrete impulse response h(k) === h(T_s*k), k range from 0 to N
    h = np.zeros(N + 1)
    for k in range(N + 1):
        h[k] = T_s * delta_omega * (2/np.pi) * np.sum(P * np.cos(k * np.arange(M + 1) * T_s * delta_omega))

    # Step 3: Generate the turbulence component in the interval using convolution
    N_t = int(t_final / T_s)
    v_t = np.zeros(N_t)
    
    # Zero-pad the white noise 
    white_noise_padded = np.pad(white_noise, (0, N), 'constant')
    
    for m in range(N_t):
        v_t[m] = T_s * np.sum(h * white_noise_padded[m : m + N + 1])
    
    v = v_bar + sigma_v * v_t
    
    plt.figure()
    plt.plot(np.arange(N_t) * T_s, v)
    plt.title('Wind Speed with Turbulence v(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('v(t)')
    plt.grid(True)
    plt.show()
    
    return v

def N_selection(L, v_bar, T_s, delta_omega, M, N):
    """
    select a proper N based on step size 
    """
    T_F = L / v_bar
    K_F = np.sqrt((2 * np.pi * T_F)/(4.20654 * T_s))
    P = np.zeros(M + 1)
    for r in range(M + 1):
        P[r] = np.real(K_F / (1 + 1j * r * delta_omega * T_F)**(5/6))
    
    
    
    h = np.zeros(N + 1)
    for k in range(N+1):
        h[k] = T_s * delta_omega * (2/np.pi) * np.sum(P * np.cos(k * np.arange(M + 1) * T_s * delta_omega))
    K_hat_F = T_s * np.sum(h)

    
    plt.plot(np.arange(M + 1) * delta_omega, P)
    plt.xscale('log')
    plt.title('Frequency Domain Representation P(omega)')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('P(omega)')
    plt.grid(True)
    plt.show()
    
    plt.plot(np.arange(N + 1) * T_s, h)
    plt.title('Impulse Response h(k)')
    plt.xlabel('Time (s)')
    plt.ylabel('h(k)')
    plt.grid(True)
    plt.show()
    
    print(K_hat_F, K_F)
        
    

t_final = 180
T_s = 1
N_t = int(t_final / T_s)


state_before = np.random.get_state()
np.random.seed(7619098)
white_noise = np.random.randn(N_t)
np.random.set_state(state_before)
v = gen_turbulence(20, 180, 0.13, T_s, t_final, white_noise)
    
#N_selection(180, 20, 1, 0.001, 10000, 100)
    
    
    
    