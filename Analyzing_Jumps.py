import MZ_Wavelet_Transforms
import Singularity_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file utilizes the functions in Singularity_Analysis to look at "jumps"
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
jump_threshold = 0.50
alpha_threshold = 0.2
compression_threshold = 5

# Parameters for the "bad" data
number_pre_jump = 100000
pre_jump_value = 0.500
number_between_jump = 100000
between_jump_value = 1.500
number_post_jump = 100000
post_jump_value = 0.500
noise_level = 0.05
total_number = number_pre_jump + number_between_jump + number_post_jump

# Parameters for number of scales for the wavelet transform
number_scales = 3

# -------------------------------
# Creating the data
# -------------------------------

# Creating the "bad" data set
pre_jump = np.full(number_pre_jump, pre_jump_value)
between_jump = np.full(number_between_jump, between_jump_value)
post_jump = np.full(number_post_jump, post_jump_value)
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

# -------------------------------
# Generating Plots
# -------------------------------

# Plotting the original signal
plt.subplot(1, 4, 1)
plt.plot(time_axis, original_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.grid(True)

# Plotting the processed signal
plt.subplot(1, 4, 2)
plt.plot(time_axis, processed_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Processed Signal')
plt.grid(True)

# Plotting the wavelet transform
plt.subplot(1, 4, 3)
plt.plot(time_axis, wavelet_transform)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Wavlet Transform')
plt.grid(True)

# Plotting the time series
plt.subplot(1, 4, 4)
plt.plot(time_axis, time_series)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Series Remaining')
plt.grid(True)
plt.show()

# -------------------------------
# Looking for jumps based on behavior
# -------------------------------

# Start time
jump_method_start_time = time.time()

# Using the compute_jump_locations function to find jumps
jump_indexes = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

# Time to run
jump_method_run_time = time.time() - jump_method_start_time

# String for run time
jump_method_run_time_string = f"Run time {jump_method_run_time * 1000} ms"

# Printing the results
print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Behavior", "magenta")
print(jump_indexes)
Singularity_Analysis.print_colored(jump_method_run_time_string, "green")
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the old alpha values
# -------------------------------

# Start time
old_method_start_time = time.time()

# Using the compute_alpha_values function to calcualte alpha for entire data set
old_alpha_values = Singularity_Analysis.compute_alpha_values(wavelet_transform)

# Getting the indexes
old_alpha_jump_indexes = Singularity_Analysis.compute_alpha_indexes(old_alpha_values, alpha_threshold)

# Time to run
old_method_run_time = time.time() - old_method_start_time

# Plotting the alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time_axis, old_alpha_values)
plt.title('Alpha Values as Calculated via the Old Method')
plt.show()

# String for run time
old_method_run_time_string = f"Run time {old_method_run_time * 1000} ms"

print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Old Alpha", "magenta")
print(old_alpha_jump_indexes)
Singularity_Analysis.print_colored(old_method_run_time_string, "green")
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the new alpha values
# -------------------------------

# Start time
new_method_start_time = time.time()

# Using the function to calcualte alpha for entire data set
alpha_values, alpha_jump_indexes = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 5, jump_threshold, alpha_threshold)

# Time to run
new_method_run_time = time.time() - new_method_start_time

# Plotting the new alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time_axis, alpha_values)
plt.title('Alpha Values as Calculated via the New Method')
plt.show()

# String for run time
new_method_run_time_string = f"Run time {new_method_run_time * 1000} ms"

print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on New Alpha", "magenta")
print(alpha_jump_indexes)
Singularity_Analysis.print_colored(new_method_run_time_string, "green")
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Comparing old and new alpha values at the points of interest
# -------------------------------

print("-------------------------------------------------------------")
print("At first jump (99), old alpha is", old_alpha_values[number_pre_jump - 1])
print("At first jump (99), new alpha is", alpha_values[number_pre_jump - 1])
print("At second jump (199), old alpha is", old_alpha_values[number_pre_jump + number_between_jump- 1])
print("At second jump (199), new alpha is", alpha_values[number_pre_jump + number_between_jump - 1])
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking at ways to better compute alphas that matter
# -------------------------------

#
# Method 0: Possible jumps based on both behavior and alpha
#

behavior_jumps_indexes = np.array(jump_indexes)
combined = np.intersect1d(alpha_jump_indexes, jump_indexes)

print("-------------------------------------------------------------")
print("Indexes Where Jumps are Suspected based on Beahvior AND Alpha")
Singularity_Analysis.print_colored(combined, "cyan")
print("-------------------------------------------------------------")
print("")