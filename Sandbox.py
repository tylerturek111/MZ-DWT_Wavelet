import MZ_Wavelet_Transforms
import Jump_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file is utilized to test code, it doesn't have anything of value
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
jump_threshold = 0.30
alpha_threshold = 0.2
compression_threshold = 5

# Parameters for the "bad" data
number_pre_jump = 100000
pre_jump_value = 0.563
number_between_jump = 100000
between_jump_value = 2.583
number_post_jump = 100000
post_jump_value = 1.382
noise_level = 0.05
total_number = number_pre_jump + number_between_jump + number_post_jump

# Parameters for number of scales for the wavelet transform
number_scales = 3

# -------------------------------
# Creating the data
# -------------------------------

# Creating the "bad" data set
pre_jump = np.empty(number_pre_jump)
for i in range(number_pre_jump):
    pre_jump[i] = pre_jump_value
between_jump = np.empty(number_between_jump)
for i in range(number_between_jump):
    between_jump[i] = between_jump_value
post_jump = np.empty(number_post_jump)
for i in range(number_post_jump):
    post_jump[i] = post_jump_value
smooth_original_data_half = np.concatenate((pre_jump, between_jump))
smooth_original_data = np.concatenate((smooth_original_data_half, post_jump))
original_data = smooth_original_data + noise_level * np.random.randn(total_number)

# Creating the time axis
time_axis = np.linspace(0, 1, total_number, endpoint=False)

# -------------------------------
# Running the wavelet transform
# -------------------------------

wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)

jump_indexes = Jump_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

# # -------------------------------
# # Actually analyzing time for jump indexes
# # -------------------------------

# ttime1 = time.time()
# original_jump_indexes = Jump_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)
# ttime2 = time.time()

# new_jump_indexes = Jump_Analysis.new_compute_jump_locations(wavelet_transform, jump_threshold)
# ttime3 = time.time()

# print ("-------------------------------------------------------------")
# print ("Times Under the Original Method", (ttime2 - ttime1) * 1000, "ms")
# print ("Times Under the New Method", (ttime3 - ttime2) * 1000, "ms")
# print ("-------------------------------------------------------------")
# print("")

# # Comparing Results
# print ("-------------------------------------------------------------")
# print("Old Result", original_jump_indexes)
# print("New Result", new_jump_indexes)
# print ("-------------------------------------------------------------")
# print("")

# -------------------------------
# Actually analyzing time
# -------------------------------

# New Time
time_temp_start = time.time()

the_stuff, _ = Jump_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

time_temp_end = time.time()

print ("-------------------------------------------------------------")
print ("Times Under the Optimized Method")
print("Computing Alpha Run Time", (time_temp_end - time_temp_start) * 1000, "ms")
print ("-------------------------------------------------------------")
print("")

# Old Time
time_temp2_start = time.time()

the_stuff2 = Jump_Analysis.efficient_compute_alpha_values(wavelet_transform, 1, jump_indexes)

time_temp2_intermediate = time.time()

_ = Jump_Analysis.compute_alpha_indexes(the_stuff2, alpha_threshold)

time_temp2_end = time.time()

print ("-------------------------------------------------------------")
print ("Times Under the Un-optimized Method")
print("Computing Alpha Run Time", (time_temp2_intermediate - time_temp2_start) * 1000, "ms")
print("Generating Indexes Run Time", (time_temp2_end - time_temp2_intermediate) * 1000, "ms")
print ("-------------------------------------------------------------")
print("")

# Comparing Results
print ("-------------------------------------------------------------")
print("Old Result", the_stuff2)
print("New Result", the_stuff)
print ("-------------------------------------------------------------")
print("")