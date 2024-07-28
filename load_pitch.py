# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:41:00 2024

@author: ghhh7
"""
import numpy as np
import matplotlib.pyplot as plt
from gen_wind_Van_Der_Hoven import generate_wind

seed = [8572651, 3981393, 1062997]

pitch = np.load('large_std_pitch/[8572651 3981393 1062997].npy')

time_step = 0.5

time = np.arange(0, len(pitch) * time_step, time_step)

# Ensure the time array and pitch array have the same length
if len(time) > len(pitch):
    time = time[:len(pitch)]

# Plot pitch vs. time
plt.figure(figsize=(10, 6))
plt.plot(time, pitch, label='Pitch')
plt.xlabel('Time (s)')
plt.ylabel('Pitch')
plt.title('Pitch vs Time')
plt.grid(True)
plt.legend()
plt.show()