import MZ_Wavelet_Transforms
import Singularity_Analysis
import Generating_Real_Data

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file utilizes the functions in Singularity_Analysis to look at "jumps" with real data
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
jump_threshold = 0.001
alpha_threshold = 0.2
compression_threshold = 5
total_number = 100

# Parameters for number of scales for the wavelet transform
number_scales = 3

# -------------------------------
# Getting the data
# -------------------------------

# Getting the real data
original_data = Generating_Real_Data.get_real_data(total_number)

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
plt.xlabel('Time (s)', fontsize = 6)
plt.ylabel('Amplitude', fontsize = 6)
plt.title('Original Signal', fontsize = 8)
plt.grid(True)

# Plotting the wavelet transform
plt.subplot(1, 4, 2)
plt.plot(time_axis, wavelet_transform)
plt.xlabel('Time (s)', fontsize = 6)
plt.ylabel('Value', fontsize = 6)
plt.title('Wavlet Transform', fontsize = 8)
plt.grid(True)

# -------------------------------
# Looking for jumps based on behavior
# -------------------------------

# Using the compute_jump_locations function to find jumps
jump_indexes = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

# Printing the results
print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Behavior", "magenta")
print(jump_indexes)
print("Number of jumps", jump_indexes.size)
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the old alpha values
# -------------------------------

# Using the compute_alpha_values function to calcualte alpha for entire data set
old_alpha_values = Singularity_Analysis.compute_alpha_values(wavelet_transform)

# Getting the indexes
old_alpha_jump_indexes = Singularity_Analysis.compute_alpha_indexes(old_alpha_values, alpha_threshold)

# Plotting the alpha values
plt.subplot(1, 4, 3)
plt.axhline(y = 0, color = 'r')
plt.plot(time_axis, old_alpha_values)
plt.xlabel('Time (s)', fontsize = 6)
plt.ylabel('Alpha', fontsize = 6)
plt.title('Alpha Values as Calculated via the Old Method', fontsize = 8)
plt.grid(True)

print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Old Alpha", "magenta")
print(old_alpha_jump_indexes)
print("Number of jumps", old_alpha_jump_indexes.size)
print("-------------------------------------------------------------")
print("")

# -------------------------------
# Looking for jumps based on the new alpha values
# -------------------------------

# Using the function to calcualte alpha for entire data set
alpha_values, alpha_jump_indexes = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

# Plotting the new alpha values
plt.subplot(1, 4, 4)
plt.axhline(y = 0, color = 'r')
plt.plot(time_axis, alpha_values)
plt.xlabel('Time (s)', fontsize = 6)
plt.ylabel('Alpha', fontsize = 6)
plt.title('Alpha Values as Calculated via the New Method', fontsize = 8)
plt.grid(True)

# Saving all of the figures
plt.savefig('plot1.png')


print("-------------------------------------------------------------")
Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on New Alpha", "magenta")
print(alpha_jump_indexes)
print("Number of jumps", alpha_jump_indexes.size)
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
