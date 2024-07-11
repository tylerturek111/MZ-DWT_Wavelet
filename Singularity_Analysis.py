import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import time

# ()()()()()()()()()()()()()
# This file contains the functions that are used to find jumps and calculate alpha-values
# ()()()()()()()()()()()()()

# -------------------------------
# compute_jump_locations
# Looks for jumps by analyzing the behavior of the data
# transform_array : 2-D Numpy Array
#                   The wavelet transform array
# threshold       : integer
#                   The threshold above which to flag jumps
# Returns         : 1-D Numpy Array of integers
#                   The indexes of jumps flagged by behavior
# -------------------------------

def compute_jump_locations(transform_array, threshold):    
    # # Compute the absolute differences between consecutive indexes within the first column of the transform array
    # differences = np.abs(np.diff(transform_array[:, 0]))

    # # Find the indices where the difference exceeds the jump threshold
    # raw_jump_indexes = np.where(differences > threshold)[0] + 1
    # jump_indexes = raw_jump_indexes[::2]
    
    # return np.array(jump_indexes)

    jump_indexes = np.where(abs(transform_array[:, 0]) > threshold)[0]

    return np.array(jump_indexes)

# -------------------------------
# compute_alpha_values
# Computing alpha for the entire transform array using the old method
# transform_array : 2-D Numpy Array
#                   The wavelet transform array
# Returns         : 1-D Numpy Array of floats
#                   The alpha values at every time point
# -------------------------------

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

# -------------------------------
# compute_alpha_indexes
# Computing indexes where alpha value is above a certain threshold
# alpha_values : 1-D Numpy Array
#                The alpha values at every time point
# threshold    : integer
#                The alpha threshold below which jumps are flagged
# Returns      : 1-D Numpy Array of integers
#                The indexes of jumps flagged by alpha values
# -------------------------------

def compute_alpha_indexes(alpha_values, threshold):
    alpha_jump_indexes = np.where(abs(alpha_values) < threshold)[0]
    
    return alpha_jump_indexes

# -------------------------------
# efficient_compute_alpha_values
# Computing alpha for the entire transform array using the new optimized method
# transform_array       : 2-D Numpy Array
#                         The wavelet transform array
# cone_slope            : integer
#                         The slope of the c-region cone
# singularity_locations : 1-D Numpy Array
#                         The location of singularities that define the c regions
# Returns               : 1-D Numpy Array of floats
#                         The alpha values at every time point
# -------------------------------

def efficient_compute_alpha_values(transform_array, cone_slope, singularity_locations):
    # time_1 = time.time()
    
    # Getting some size data
    length, scales = transform_array.shape
    number_singularity = singularity_locations.shape[0]

    # Initializing arrays to store the relevant transform data needed for this implementation
    c_values = np.empty([number_singularity, scales])
    o_values = np.empty([number_singularity + 1, scales])
    o_values_count = np.empty([number_singularity + 1, scales])

    # Creating global x (log of scale values)
    log_normalized_scale = np.log(np.arange(scales) + 1)

    # Initializing arrays to store final alpha values
    c_alpha_values = np.empty(number_singularity)
    o_alpha_values = np.empty(number_singularity + 1)
    alpha_values = np.empty(length)

    # time_2 = time.time()

    # Computing the coefficients that we actually care about
    for scale in range(scales):
        for singularity in range(number_singularity):
            spread = cone_slope * scale

            # Calculating the relevent c coefficients
            transform_values = transform_array[:, scale]
            interested_c_indexes = np.arange(max(0, singularity_locations[singularity] - spread), min(singularity_locations[singularity] + spread + 1, length), 1)            
            c_values[singularity, scale] = transform_values[interested_c_indexes][np.argmax(np.abs(transform_values[interested_c_indexes]))]

            # Computing the proper start and end indexes for the o regions
            o_start_index = 0
            o_end_index = 0
            if singularity == 0:
                o_start_index = 0
                o_end_index = min(singularity_locations[singularity] - spread - 1, length)
            else:
                o_start_index = max(singularity_locations[singularity - 1] + spread + 1, 0)
                o_end_index = min(singularity_locations[singularity] - spread - 1, length)

            # Calculating the relevent o coefficients
            interested_o_indexes = np.arange(o_start_index, o_end_index, 1)
            o_values[singularity, scale] = np.mean(np.abs(transform_values[interested_o_indexes]))
            # o_values[singularity, scale] = np.mean(transform_values[interested_o_indexes])

        # The last region between the last index and the end
        if number_singularity != 0:
            o_start_index = max(singularity_locations[number_singularity - 1] + spread + 1, 0)
            o_end_index = length
            interested_o_indexes2 = np.arange(o_start_index, o_end_index, 1)
            o_values[number_singularity, scale] = np.mean(np.abs(transform_values[interested_o_indexes2]))
            # o_values[number_singularity, scale] = np.mean(transform_values[interested_o_indexes2])

    # Getting rid of all instances of 0 within the o_values to account for log(0) errors
    o_values[o_values == 0] = 0.000000001

    # time_3 = time.time()

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

        # Assigning the alpha values to the proper indexes
        # For the c values
        alpha_values[singularity_locations[singularity]] = c_alpha_values[singularity]
        # For the o values
        if singularity == 0:
            alpha_values[0 : singularity_locations[singularity]] = o_alpha_values[singularity]
        else:
            alpha_values[singularity_locations[singularity - 1] + 1 : singularity_locations[singularity]] = o_alpha_values[singularity]
    
    # Computing the o alpha values for the last region and assigning them
    if number_singularity != 0:
        current_o_row = o_values[number_singularity]
        log_current_o_row = np.log(np.abs(current_o_row))
        o_alpha = np.polyfit(log_normalized_scale, log_current_o_row, 1)[0]
        o_alpha_values[number_singularity] = o_alpha
        alpha_values[singularity_locations[number_singularity - 1] + 1 : length] = o_alpha_values[number_singularity]
    
    # time_4 = time.time()

    # print("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
    # print("Run times for efficient_compute_alpha_values FROM Jump_Analysis.py")
    # print("Run time for setup", (time_2 - time_1) * 1000, "ms")
    # print("Run time for calculating coefficients", (time_3 - time_2) * 1000, "ms")
    # print("Run time for actually calculating alpha", (time_4 - time_3) * 1000, "ms")
    # print("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
    # print("")

    return(alpha_values)

# -------------------------------
# packaged_compute_alpha_values_and_indexes
# Determine location of jump singularities and compute alpha using the new optimized
# method all at once and also generating the indexes with jumps
# transform_array  : 2-D Numpy Array
#                    The wavelet transform array
# cone_slope       : integer
#                    The slope of the c-region cone
# jump_threshold   : integer
#                    The threshold above which to flag jumps
# alpha_threshold  : integer
#                    The alpha threshold below which jumps are flagged
# RETURNS, returns : 1-D Numpy Array of floats
#                    The alpha values at every time point
# returns, RETURNS : 1-D Numpy Array of integers
#                    The indexes of jumps flagged by alpha values
# -------------------------------

def packaged_compute_alpha_values_and_indexes(transform_array, cone_slope, jump_threshold, alpha_threshold):
    # time1 = time.time()

    singularities = compute_jump_locations(transform_array, jump_threshold)
    
    # time2 = time.time()

    alpha_values = efficient_compute_alpha_values(transform_array, cone_slope, singularities)
    
    # time3 = time.time()

    alpha_indexes = compute_alpha_indexes(alpha_values, alpha_threshold)
    
    # time4 = time.time()

    # print("(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)")
    # print("Run times for packaged_compute_alpha_values_and_indexes FROM Jump_Analysis.py")
    # print("Run time for setup", (time2 - time1) * 1000, "ms")
    # print("Run time for calculating alpha", (time3 - time2) * 1000, "ms")
    # print("Run time for calculating indexes", (time4 - time3) * 1000, "ms")
    # print("TOTAL RUN TIME", (time4 - time1) * 1000, "ms")
    # print("(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)")
    # print("")

    return alpha_values, alpha_indexes

# -------------------------------
# compute_glitch_locations
# Determine location of glitch sinularities and compute their size
# This method exploits a property of the new alpha method in which nearby jumps result in intermediate alpha
# values being recorded as NaN, allowing us to locate nearby jumps and from that determine whether or not
# they are actually glitches
# transform_array  : 2-D Numpy Array
#                    The wavelet transform array
# alpha_values     : integer
#                    The alpha values at every time point
# alpha_indexes    : integer
#                    The indexes of jumps flagged by alpha values
# glitch_threshold : integer
#                  : The maximum wavelet transform deviation from the left to right of a glitch
# RETURNS, returns : 1-D Numpy Array of integers
#                    The location of suspected glitches
# returns, RETURNS : 1-D Numpy Array of integers
#                    The size of suspected glitches
# -------------------------------

def compute_glitch_locations(transform_array, alpha_values, alpha_indexes, glitch_threshold):
    # Get the start and end location of each NaN cluster
    nan_locations = np.where(np.isnan(alpha_values))[0]
    nan_clusters = np.split(nan_locations, np.where(np.diff(nan_locations) != 1)[0] + 1)
    nan_cluster_locations = np.array([(cluster[0], cluster[-1]) for cluster in nan_clusters])
    number_clusters, _ = nan_cluster_locations.shape

    # Initializing the array to put result, with each glitch location being referenced as (glitch location, glitch size)
    glitch_locations = np.empty((0,))
    glitch_sizes = np.empty((0, ))

    # Looking at each NaN cluster and determining if it is truly a glitch
    for i in range(number_clusters):
        # Calculating some relevant information
        start_index = nan_cluster_locations[i, 0]
        end_index = nan_cluster_locations[i, 1]
        left_value = transform_array[(start_index - 2), 0]
        right_value = transform_array[(end_index + 2), 0]
        medium_value = transform_array[(math.floor(np.mean(nan_cluster_locations[i, :])), 0)]

        # Determing if the NaN cluster is truly a glitch and computing its size
        if abs(left_value - right_value) < glitch_threshold:
            glitch_size = 0
            nan_size = end_index - start_index + 1
            if nan_size == 1:
                if abs(alpha_values[start_index - 1] - alpha_values[end_index + 1]) > 0.1:
                    glitch_size = 1
                else:
                    glitch_size = 2
            else:
                glitch_size = nan_size + 1
            glitch_locations = np.append(glitch_locations, int((start_index - 1)))
            glitch_sizes = np.append(glitch_sizes, int(glitch_size))

    return glitch_locations, glitch_sizes

# -------------------------------
# print_colored
# Allows output text to be colored
# text  : string
#                    The string to be printed
# color     : string
#                    The color of the string
# returns : string
#                    The string of text with colors
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