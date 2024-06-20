import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# ()()()()()()()()()()()()()
# This code is designed to find jumps and calculate alpha-values
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for the "bad" data
number_pre_jump = 100
pre_jump_value = 0.123
number_post_jump = 200
post_jump_value = 0.78
noise_level = 0.0005
total_number = number_pre_jump + number_post_jump

# Parameters for number of scales for the wavelet transform
number_scales = 5

# Parameters for analyzing the jump
jump_threshold = 0.5
alpha_threshold = 0.1

# -------------------------------
# Creating the data
# -------------------------------

# Creating the "bad" data set
pre_jump = np.empty(number_pre_jump)
for i in range(number_pre_jump):
    pre_jump[i] = pre_jump_value
post_jump = np.empty(number_post_jump)
for i in range(number_post_jump):
    post_jump[i] = post_jump_value
smooth_original_data = np.concatenate((pre_jump, post_jump))
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
# Looking for jumps
# -------------------------------

# Stores possible jumps for all levels
jump_indexes = []

for col in range(number_scales):
    level_indexes = []
    previous_value = 0
    for row in range(total_number):
        current_value = wavelet_transform[row, col]
        if (abs(current_value - previous_value) > jump_threshold):
            level_indexes.append(row)
    jump_indexes.append(level_indexes)

# Stores possible jumps at the first level
behavior_jumps_indexes = np.array(jump_indexes[0])

print ("Indexes Where Jumps are Suspected based on Behavior")
print(behavior_jumps_indexes)

# -------------------------------
# Computing alpha
# -------------------------------

alpha_values = np.empty(total_number)

# Creating global x (log of scale values)
normalized_scale = np.arange(number_scales) + 1
log_normalized_scale = np.log(normalized_scale)

# Computing alpha at each time value
for row in range(total_number):
    current_row = wavelet_transform[row, :]
    log_current_row = np.log(np.abs(current_row))
    alpha = np.polyfit(log_normalized_scale, log_current_row, 1)[0]
    alpha_values[row] = alpha

# Plotting the alpha values
plt.axvline(x = (number_pre_jump / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time, alpha_values)
plt.show()

# Looking for jumps based on alpha values
alpha_jump_indexes = []

for i in range(total_number):
    if (abs(alpha_values[i]) < alpha_threshold):
        alpha_jump_indexes.append(i)

print ("Indexes Where Jumps are Suspected based on Alpha")
print(alpha_jump_indexes)

# -------------------------------
# Possible jumps based on behavior and alpha
# -------------------------------

behavior_jumps_indexes = np.array(jump_indexes[0])
combined = np.intersect1d(alpha_jump_indexes, behavior_jumps_indexes)

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Beahvior AND Alpha")
print(combined)
print ("-------------------------------------------------------------")
