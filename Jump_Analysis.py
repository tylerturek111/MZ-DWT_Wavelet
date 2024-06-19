import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------
# Defining the parameters
# -------------------------------

number_pre_jump = 100
pre_jump_value = 0.123
number_post_jump = 100
post_jump_value = 0.78
noise_level = 0.05
total_number = number_pre_jump + number_post_jump
number_scales = 5
jump_threshold = 0.5


# -------------------------------
# Creating the data
# -------------------------------

# Creating the bad data set
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

# Plotting the new signal
plt.subplot(1, 4, 2)
plt.plot(time, processed_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('New Signal')
plt.grid(True)

# Plotting the wavelet transform
plt.subplot(1, 4, 3)
plt.plot(time, wavelet_transform)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Wavlet Transform Matrix')
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

jump_indexes = []

for col in range(number_scales):
    level_indexes = []
    previous_value = 0
    for row in range(total_number):
        current_value = wavelet_transform[row, col]
        if (abs(current_value - previous_value) > jump_threshold):
            level_indexes.append(row)
    jump_indexes.append(level_indexes)

print(jump_indexes)

# -------------------------------
# Computing alpha
# -------------------------------

alpha_values = np.empty(total_number)

# Creating global x (log of scale values)
normalized_scale = np.arange(number_scales) + 1

# Method 1 for determing x
log_normalized_scale = np.log(normalized_scale)

# Method 2 for determing x
# log_normalized_scale = np.empty(number_scales)
# for i in range(number_scales):
#     value = int(normalized_scale[i])
#     print(2 ** (-1 * value))
#     log_normalized_scale[i] = math.log(2 ** (-1 * value))

# Computing alpha at each time
for row in range(total_number):
    current_row = wavelet_transform[row, :]
    log_current_row = np.log(np.abs(current_row))
    alpha = np.polyfit(log_normalized_scale, log_current_row, 1)[0]
    
    # Method 1 for alpha
    alpha_values[row] = alpha

    # Method 2 for alpha
    # new_alpha = -alpha / math.log(2)
    # alpha_values[row] = new_alpha
print(alpha_values)

print("HERE")
print(alpha_values[99])
print(alpha_values[100])
print(alpha_values[101])
plt.axvline(x = (number_pre_jump / total_number), color = "r")
plt.plot(time, alpha_values)
plt.show()