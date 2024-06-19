import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Defining the parameters
# -------------------------------

number_pre_jump = 125
pre_jump_value = 0.123
number_post_jump = 75
post_jump_value = 0.78
noise_level = 0.05
total_number = number_pre_jump + number_post_jump
number_scales = 3
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
print(wavelet_transform)
length, scales_number = wavelet_transform.shape

jump_indexes = []

for col in range(scales_number):
    level_indexes = []
    previous_value = 0
    for row in range(length):
        current_value = wavelet_transform[row, col]
        if (abs(current_value - previous_value) > jump_threshold):
            level_indexes.append(row)
    jump_indexes.append(level_indexes)

print(jump_indexes)