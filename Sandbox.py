import MZ_Wavelet_Transforms
import Jump_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# ()()()()()()()()()()()()()
# This file is utilized to test code, it doesn't have anything of value
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for the "bad" data
number_pre_jump = 100
pre_jump_value = 0.563
number_between_jump = 100
between_jump_value = 2.583
number_post_jump = 100
post_jump_value = 1.382
noise_level = 0.10
total_number = number_pre_jump + number_between_jump + number_post_jump

# Parameters for number of scales for the wavelet transform
number_scales = 3

# Parameters for analyzing the jump
jump_threshold = 0.5
alpha_threshold = 0.25
compression_threshold = 5

# -------------------------------
# Creating the data
# -------------------------------

# Creating the "bad" data set
pre_jump = np.empty(number_pre_jump)
for i in range(number_pre_jump):
    pre_jump[i] = pre_jump_value
between_jump = np.empty(number_pre_jump)
for i in range(number_between_jump):
    between_jump[i] = between_jump_value
post_jump = np.empty(number_post_jump)
for i in range(number_post_jump):
    post_jump[i] = post_jump_value
smooth_original_data_half = np.concatenate((pre_jump, between_jump))
smooth_original_data = np.concatenate((smooth_original_data_half, post_jump))
original_data = smooth_original_data + noise_level * np.random.randn(total_number)

# Creating the time axis
time = np.linspace(0, 1, total_number, endpoint=False)

# -------------------------------
# Running the wavelet transform
# -------------------------------

wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)

# -------------------------------
# Looking for jumps based on the new alpha values
# -------------------------------

# Using the efficient_compute_alpha_values function to calcualte alpha for entire data set
alpha_values = Jump_Analysis.efficient_compute_alpha_values(wavelet_transform, 1, np.array([99, 199]))

# Plotting the new alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time, alpha_values)
plt.title('Alpha Values as Calculated via the New Method')
plt.show()

alpha_jump_indexes = Jump_Analysis.compute_alpha_indexes(alpha_values, alpha_threshold)

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on New Alpha")
print(alpha_jump_indexes)
print ("-------------------------------------------------------------")
print("")