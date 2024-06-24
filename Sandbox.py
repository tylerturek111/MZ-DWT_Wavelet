import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# ()()()()()()()()()()()()()
# This file is just for me to tryout code, it doesn't really do anything
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for the "bad" data
number_pre_jump = 100
pre_jump_value = 0.123
number_between_jump = 100
between_jump_value = 0.78
number_post_jump = 100
post_jump_value = 1.325
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

# A function to compute jump locations
def compute_jump_locations(transform_array):
    length, scales = transform_array.shape

    # Stores possible jumps for all levels
    # Function to look for jumps in the wavelet transform
    jump_indexes = np.array([], int)
    previous_value = 0
    for row in range(length):
        current_value = wavelet_transform[row, 0]
        if (abs(current_value - previous_value) > jump_threshold):
            jump_indexes = np.append(jump_indexes, row)
    return jump_indexes

# Using the compute_jump_locations function to find jumps
jump_indexes = compute_jump_locations(wavelet_transform)

# Printing the results
print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Behavior")
print(jump_indexes)
print ("-------------------------------------------------------------")
print("")

# -------------------------------
# Computing alpha for the entire transform array
# -------------------------------

# Function to compute alpha at each time point given an array
def compute_alpha_values(transform_array):
    length, scales = transform_array.shape
    alpha_values = np.empty(length)

    # Creating global x (log of scale values)
    normalized_scale = np.arange(scales) + 1
    log_normalized_scale = np.log(normalized_scale)

    # Computing alpha at each time value
    for row in range(length):
        current_row = transform_array[row, :]
        log_current_row = np.log(np.abs(current_row))
        alpha = np.polyfit(log_normalized_scale, log_current_row, 1)[0]
        alpha_values[row] = alpha
    return alpha_values


# Using the compute_alpha_values function to calcualte alpha for entire data set
alpha_values = compute_alpha_values(wavelet_transform)

# Function to compute indexes where alpha value is above a certain threshold
def compute_alpha_indexes(alpha_values, threshold):
    length = alpha_values.shape[0]

    # Looking for jumps based on alpha values
    alpha_jump_indexes = np.array([], int)
    for i in range(length):
        if (abs(alpha_values[i]) < threshold):
            alpha_jump_indexes = np.append(alpha_jump_indexes, i)
    return alpha_jump_indexes

alpha_jump_indexes = compute_alpha_indexes(alpha_values, alpha_threshold)

print ("-------------------------------------------------------------")
print ("Indexes Where Jumps are Suspected based on Alpha")
print(alpha_jump_indexes)
print ("-------------------------------------------------------------")
print("")

# Function to compute alpha in a more efficient manner
# transform_array: 2-D Numpy Array, The wavelet transform array obtained for the signal
# cone_slope: integer value, What the slope of the cone dividing the C and O regions should be. I believe the value
#     is probably just 1, but don't know for sure to I added this variable in. 
# singularity_locations: 1-D Numpy Array Location of singularities. Paper didn't provide information on how these 
#     could be found, though some implementation could be included later
def efficient_compute_alpha_values(transform_array, cone_slope, singularity_locations):
    # Getting some size data
    length, scales = transform_array.shape
    number_singularity = singularity_locations.shape[0]

    # Initializing arrays to store the relevant transform data needed for this implementation
    c_values = np.empty([number_singularity, scales])
    o_values = np.empty([number_singularity + 1, scales])
    o_values_count = np.empty([number_singularity + 1, scales])

    # Computing the coefficients that we actually care about
    for singularity in range(number_singularity):
        for scale in range(scales):
            # Calculating the relevent c coefficients
            transform_values = transform_array[:, scale]
            interested_c_indexes = np.arange(singularity_locations[singularity] - cone_slope * scale, singularity_locations[singularity] + cone_slope * scale + 1, 1)
            c_values[singularity, scale] = np.max(transform_values[interested_c_indexes])

            # Calculating the relevent o coefficients
            # The case of the first region between 0 and the first index
            if singularity == 0:
                interested_o_indexes = np.arange(0, singularity_locations[singularity] - cone_slope * scale, 1)
                o_values[0, scale] = np.mean(transform_values[interested_o_indexes])
            # All other intermediate regions
            else:
                interested_o_indexes = np.arange(singularity_locations[singularity - 1] + cone_slope * scale + 1, singularity_locations[singularity] - cone_slope * scale + 1)
                o_values[singularity, scale] = np.mean(transform_values[interested_o_indexes])
            # The last region between the last index and the end
            if singularity == (number_singularity - 1):
                interested_o_indexes2 = np.arange(singularity_locations[singularity] + cone_slope * scale + 1, length, 1)
                o_values[singularity + 1, scale] = np.mean(transform_values[interested_o_indexes2])

    # Creating global x (log of scale values)
    normalized_scale = np.arange(scales) + 1
    log_normalized_scale = np.log(normalized_scale)

    # Now computing the alphas with these reduced coefficients
    # Initializing arrays to store values
    c_alpha_values = np.empty(number_singularity)
    o_alpha_values = np.empty(number_singularity + 1)
    alpha_values = np.empty(length)

    # Calculating alpha values
    for singularity in range(number_singularity):
        # Computing the c alpha values
        current_c_row = c_values[singularity, :]
        log_current_c_row = np.log(np.abs(current_c_row))
        c_alpha = np.polyfit(log_normalized_scale, log_current_c_row, 1)[0]
        c_alpha_values[singularity] = c_alpha

        # Computing the o alpha values
        current_o_row = o_values[singularity, :]
        log_current_o_row = np.log(np.abs(current_o_row))
        o_alpha = np.polyfit(log_normalized_scale, log_current_o_row, 1)[0]
        o_alpha_values[singularity] = o_alpha
        # Computing the o alpha values for the last region
        if singularity == (number_singularity - 1):
            current_o_row = o_values[singularity + 1, :]
            log_current_o_row = np.log(np.abs(current_o_row))
            o_alpha = np.polyfit(log_normalized_scale, log_current_o_row, 1)[0]
            o_alpha_values[singularity + 1] = o_alpha
        
        # Assigning the alpha values to the proper indexes
        # For the c values
        alpha_values[singularity_locations[singularity]] = c_alpha_values[singularity]
        # For the o values
        if singularity == 0:
            alpha_values[0 : singularity_locations[singularity]] = o_alpha_values[singularity]
        else:
            alpha_values[singularity_locations[singularity - 1] + 1 : singularity_locations[singularity]] = o_alpha_values[singularity]
        if singularity == (number_singularity - 1):
            alpha_values[singularity_locations[singularity] + 1 : length] = o_alpha_values[singularity + 1]

    return(alpha_values)

# Using the efficient_compute_alpha_values function to calcualte alpha for entire data set
new_alpha_values = efficient_compute_alpha_values(wavelet_transform, 1, np.array([99, 199]))

# Plotting the new alpha values
plt.axvline(x = ((number_pre_jump - 1) / total_number), color = "r")
plt.axvline(x = ((number_pre_jump + number_between_jump - 1) / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(time, new_alpha_values)
plt.show()