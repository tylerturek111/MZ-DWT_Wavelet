import MZ_Wavelet_Transforms
import Singularity_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file utilizes the functions in Singularity_Analysis to look at "glitches"
# ()()()()()()()()()()()()()

# -------------------------------
#region Defining the Parameters
# -------------------------------

# Parameters for analyzing the behavior and alphas
jump_threshold = 0.75
alpha_threshold = 0.20
glitch_treshold = 1.0
compression_threshold = 5

# Parameters for the "bad" glitch data
number_glitches = 100
number_between_glitches = 50
number_glitch_values = 3
flat_value = 1.023
glitch_value = 1000.532

# Parameters for the "bad" jump data
number_pre_jump = 100
pre_jump_value = 1.023
number_between_jump = 100
between_jump_value = 2.583
number_post_jump = 100
post_jump_value = 0.382

# Other parameters
noise_level = 0.05
number_scales = 3
total_number = number_between_glitches * (number_glitches + 1) + number_glitch_values * number_glitches + number_pre_jump + number_between_jump + number_post_jump
# total_number = number_between_glitches * (number_glitches + 1) + number_glitch_values * number_glitches

#endregion Defining the Parameters
# -------------------------------

# -------------------------------
#region Creating the "Bad" Data
# -------------------------------

# Creating the "bad" glitch data set
flat_block = np.full(number_between_glitches, flat_value)
glitch_block = np.full(number_glitch_values, glitch_value)
glitch_data = flat_block
for i in range(number_glitches):
    glitch_data = np.concatenate((glitch_data, glitch_block))
    glitch_data = np.concatenate((glitch_data, flat_block))
# smooth_original_data = glitch_data

# Creating the "bad" jump data set
pre_jump = np.full(number_pre_jump, pre_jump_value)
between_jump = np.full(number_between_jump, between_jump_value)
post_jump = np.full(number_post_jump, post_jump_value)
jump_data_intermediate = np.concatenate((pre_jump, between_jump))
jump_data = np.concatenate((jump_data_intermediate, post_jump))

# Stiching the "bad" data together and adding noise
smooth_original_data = np.concatenate((glitch_data, jump_data))
original_data = smooth_original_data + noise_level * np.random.randn(total_number)

# Creating the time axis
time_axis = np.linspace(0, 1, total_number, endpoint=False)

# endregion Creating the "Bad" Data
# -------------------------------


# -------------------------------
#region Running the Wavelet Transform
# -------------------------------

wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)
#endregion Running the Wavelet Transform
# -------------------------------

# -------------------------------
#region Generating Plots
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

#endregion Generating Plots
# -------------------------------

# -------------------------------
#region Adding Color
# -------------------------------

def print_colored(text, color):
    color_codes = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }
    
    if color in color_codes:
        print(f"{color_codes[color]}{text}{color_codes['reset']}")
    else:
        print(text)

# endregion Adding Color
# -------------------------------

# -------------------------------
#region Looking for Glitches
# -------------------------------

alpha_values, alpha_jump_indexes = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

# Start time
glitch_start_time = time.time()

# Using compute_glitch_locations to calculate the location and size of glitches
glitch_locations, glitch_sizes = Singularity_Analysis.compute_glitch_locations(wavelet_transform, alpha_values, alpha_jump_indexes, glitch_treshold)

# End time
glitch_run_time = time.time() - glitch_start_time

# String for run time
new_method_run_time_string = f"Run time {glitch_run_time * 1000} ms"

print("-------------------------------------------------------------")
print_colored("Location and Sizes of Suspected Glitches", "magenta")
for i in range(glitch_locations.shape[0]):
    print("Glitch of size", glitch_sizes[i], "starting at", glitch_locations[i])
print_colored(new_method_run_time_string, "green")    
print("-------------------------------------------------------------")
print("")

#endregion Looking for Glitches
# -------------------------------
