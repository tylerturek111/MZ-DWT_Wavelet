import MZ_Wavelet_Transforms

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math

# ()()()()()()()()()()()()()
# This file is used to store un-modified old code that might be needed later
# ()()()()()()()()()()()()()

# -------------------------------
# efficient_compute_alpha_values
# Computing alpha for the entire transform array using the new optimized method
# transform_array       : 2-D Numpy Array
#                         The wavelet transform array
# cone_slope            : integer
#                         The slope of the c-region cone
# singularity_locations : 1-D Numpy Array
#                         The location of singularities that define the c regions
# Returns               : 1-D Numpy Array
#                         The alpha values at every time point
# -------------------------------

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
            interested_c_indexes = np.arange(max(0, singularity_locations[singularity] - cone_slope * scale), min(singularity_locations[singularity] + cone_slope * scale + 1, length), 1)
            c_values[singularity, scale] = max(transform_values[interested_c_indexes], key=lambda x: abs(x))

            # Calculating the relevent o coefficients
            # The case of the first region between 0 and the first index
            if singularity == 0:
                interested_o_indexes = np.arange(0, min(singularity_locations[singularity] - cone_slope * scale, length), 1)
                o_values[0, scale] = np.mean(transform_values[interested_o_indexes])
            # All other intermediate regions
            else:
                interested_o_indexes = np.arange(max(singularity_locations[singularity - 1] + cone_slope * scale + 1, 0), min(singularity_locations[singularity] - cone_slope * scale + 1, length), 1)
                o_values[singularity, scale] = np.mean(transform_values[interested_o_indexes])
            # The last region between the last index and the end
            if singularity == (number_singularity - 1):
                interested_o_indexes2 = np.arange(max(singularity_locations[singularity] + cone_slope * scale + 1, 0), length, 1)
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
