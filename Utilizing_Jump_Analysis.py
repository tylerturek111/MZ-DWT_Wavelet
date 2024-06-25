import MZ_Wavelet_Transforms
import Jump_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# ()()()()()()()()()()()()()
# This file utilizes the functions in Jump_Analysis to look at jumps
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
jump_threshold = 0.30
alpha_threshold = 0.2
compression_threshold = 5

# Parameters for the "bad" data
number_pre_jump = 100
pre_jump_value = 0.563
number_between_jump = 100
between_jump_value = 2.583
number_post_jump = 100
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
# Generating Plots
# -------------------------------

# Plotting the original signal
plt.subplot(1, 4, 1)
plt.plot(time, original_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.grid(True)

# Plotting the processed signal
plt.subplot(1, 4, 2)
plt.plot(time, processed_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Processed Signal')
plt.grid(True)

# Plotting the wavelet transform
plt.subplot(1, 4, 3)
plt.plot(time, wavelet_transform)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Wavlet Transform')
plt.grid(True)

# Plotting the time series
plt.subplot(1, 4, 4)
plt.plot(time, time_series)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Series Remaining')
plt.grid(True)
plt.show()

# -------------------------------
# Looking for jumps based on behavior
# -------------------------------

# Using the compute_jump_locations function to find jumps
jump_indexes = Jump_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

# Printing the results
print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Behavior")
print(jump_indexes)
print ("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the old alpha values
# -------------------------------

# Using the compute_alpha_values function to calcualte alpha for entire data set
old_alpha_values = Jump_Analysis.compute_alpha_values(wavelet_transform)

# Plotting the alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time, old_alpha_values)
plt.title('Alpha Values as Calculated via the Old Method')
plt.show()

old_alpha_jump_indexes = Jump_Analysis.compute_alpha_indexes(old_alpha_values, alpha_threshold)

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Old Alpha")
print(old_alpha_jump_indexes)
print ("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the new alpha values
# -------------------------------

# Using the function to calcualte alpha for entire data set
alpha_values, alpha_jump_indexes = Jump_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

# Plotting the new alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time, alpha_values)
plt.title('Alpha Values as Calculated via the New Method')
plt.show()

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on New Alpha")
print(alpha_jump_indexes)
print ("-------------------------------------------------------------")
print("")

# -------------------------------
# Comparing old and new alpha values at the points of interest
# -------------------------------

print ("-------------------------------------------------------------")
print("At first jump (99), old alpha is", old_alpha_values[number_pre_jump - 1])
print("At first jump (99), new alpha is", alpha_values[number_pre_jump - 1])
print("At second jump (199), old alpha is", old_alpha_values[number_pre_jump + number_between_jump- 1])
print("At second jump (199), new alpha is", alpha_values[number_pre_jump + number_between_jump - 1])
print ("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking at ways to better compute alphas that matter
# -------------------------------

#
# Method 0: Possible jumps based on both behavior and alpha
#

behavior_jumps_indexes = np.array(jump_indexes)
combined = np.intersect1d(alpha_jump_indexes, jump_indexes)

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Beahvior AND Alpha")
print(combined)
print ("-------------------------------------------------------------")
print("")

#
# Method 1: Compressing the wavelet transform matrix and comparing it to the originally determined alpha values
#

# Compressing the transform matrix
compressed_wavelet_transform = np.empty((math.ceil(total_number / compression_threshold), number_scales))
current_count = 0
for col in range(number_scales):
    for i in range(total_number):
        if (i % compression_threshold == 0) and (i != 0):            
            compressed_wavelet_transform[int(i / compression_threshold - 1), col] = (current_count)
            current_count = wavelet_transform[i, col]
        elif i == (total_number - 1):
            compressed_wavelet_transform[math.ceil(total_number / compression_threshold) - 1, col] = (current_count)
        else:
            if wavelet_transform[i, col] > current_count:
                current_count = wavelet_transform[i, col]

# Computing alpha values of the compressed transform data
compressed_alpha_values, compressed_jump_indexes = Jump_Analysis.packaged_compute_alpha_values_and_indexes(compressed_wavelet_transform, 1, jump_threshold, alpha_threshold)

# Plotting the alpha values
if total_number % 5 == 0:
    compression_time = np.linspace(0, 1, int(total_number / compression_threshold), endpoint=False)
else: 
    compression_time = np.linspace(0, 1, int(total_number / compression_threshold) + 1, endpoint=False)
plt.axvline(x = ((number_pre_jump - compression_threshold) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - compression_threshold) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(compression_time, compressed_alpha_values)
plt.title('Alpha Values as Calculated via the New Method and Compressing the Wavelet Transform')
plt.show()

# Comparing the compressed alpha values with non-compressed alpha values in order to
# determine which alpha values are based truley on sizable jumps instead of random
# noise fluctuations
expanded_alpha_values =  np.repeat(compressed_alpha_values, compression_threshold)
expanded_alpha_jump_indexes = Jump_Analysis.compute_alpha_indexes(expanded_alpha_values, alpha_threshold)
method1_indexes = np.intersect1d(alpha_jump_indexes, expanded_alpha_jump_indexes)

# Attaching the original alpha values to the indexes we care about
method1_indexes_alpha = np.empty((0,), dtype = object)
for index in method1_indexes:
    method1_indexes_alpha = np.append(method1_indexes_alpha, [(index, alpha_values[index])])

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Alphas from Transform Compression")
print(method1_indexes)
print(method1_indexes_alpha)
print("----------")
print("Initially flagged singularities", compressed_jump_indexes)
print("Ultimately flagged locations", Jump_Analysis.compute_alpha_indexes(compressed_alpha_values, alpha_threshold))
print("Remember, data is compressed by a factor of", compression_threshold)
print ("-------------------------------------------------------------")
print("")

#
# Method 2: Compressing the original data and comparing it to the originally determined alpha values
#

# Compressing the signal data
compressed_data = np.array([])
for i in range(total_number):
    if i % compression_threshold == 0:
        compressed_data = np.append(compressed_data, original_data[i])

# Getting the wavelet transform for our compressed data
compressed2_wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, compressed_data)

# Generating alpha values for our compressed data
compressed2_alpha_values, compressed2_jump_indexes = Jump_Analysis.packaged_compute_alpha_values_and_indexes(compressed2_wavelet_transform, 1, jump_threshold, alpha_threshold)

# Plotting the alpha values
plt.axvline(x = ((number_pre_jump - compression_threshold) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - compression_threshold) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(compression_time, compressed2_alpha_values)
plt.title('Alpha Values as Calculated via the New Method and Compressing the Original Data')
plt.show()

# Comparing the compressed alpha values with non-compressed alpha values in order to
# determine which alpha values are based truley on sizable jumps instead of random
# noise fluctuations
expanded2_alpha_values =  np.repeat(compressed2_alpha_values, compression_threshold)
expanded2_alpha_jump_indexes = Jump_Analysis.compute_alpha_indexes(expanded2_alpha_values, alpha_threshold)
method2_indexes = np.intersect1d(alpha_jump_indexes, expanded2_alpha_jump_indexes)

# Attaching the original alpha values to the indexes we care about
method2_indexes_alpha = np.empty((0,), dtype = object)
for index in method2_indexes:
    method2_indexes_alpha = np.append(method2_indexes_alpha, [(index, alpha_values[index])])

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Alphas from Data Compression")
print(method2_indexes)
print(method2_indexes_alpha)
print("----------")
print("Initially flagged singularities", compressed2_jump_indexes)
print("Ultimately flagged locations", Jump_Analysis.compute_alpha_indexes(compressed2_alpha_values, alpha_threshold))
print("Remember, data is compressed by a factor of", compression_threshold)
print ("-------------------------------------------------------------")
print("")

# 
# Method 3: Combinig the results from Methods 1 and 2 to determine values
#

method3_indexes = np.intersect1d(method1_indexes, method2_indexes)

# Attaching the original alpha values to the indexes we care about
method3_indexes_alpha = np.empty((0,), dtype = object)
for index in method3_indexes:
    method3_indexes_alpha = np.append(method3_indexes_alpha, [(index, alpha_values[index])])

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Alphas from both Transform Compression and Data Compression")
print(method3_indexes)
print(method3_indexes_alpha)
print ("-------------------------------------------------------------")
print("")

#
# Method 4: Combining the results from Method 3 with the behavior values
#

method4_indexes = np.intersect1d(method3_indexes, jump_indexes)

# Attaching the original alpha values to the indexes we care about
method4_indexes_alpha = np.empty((0,), dtype = object)
for index in method3_indexes:
    method4_indexes_alpha = np.append(method4_indexes_alpha, [(index, alpha_values[index])])

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Behavior and Alphas from Transform and Data Compression")
print(method4_indexes)
print(method4_indexes_alpha)
print ("-------------------------------------------------------------")
print("")