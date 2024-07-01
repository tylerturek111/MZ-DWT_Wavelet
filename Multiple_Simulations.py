import MZ_Wavelet_Transforms
import Singularity_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file runs multiple simulations in order to generate 
# estimation of false positive and false negative rates for jumps
# ()()()()()()()()()()()()()

# -------------------------------
# create_jump_data
# Creates a dataset with some noise and a jump
# noise_level      : integer
#                    The amount of noise present in the data
# jump_size        : integer
#                    The size of the jump
# RETURNS, returns : 1-D Numpy Array of floats
#                    The signal itself
# returns, RETURNS : 1-D Numpy Array of floats
#                    The wavelet transform data
# -------------------------------

def create_jump_data(noise_level, jump_size):
    # Parameters for the data
    number_pre_jump = 100
    pre_jump_value = 1.000
    number_post_jump = 100
    post_jump_value = 1.000 + jump_size
    noise = noise_level
    total_number = number_pre_jump + number_post_jump

    # Parameters for number of scales for the wavelet transform
    number_scales = 3

    # Creating the data set
    pre_jump = np.full(number_pre_jump, pre_jump_value)
    post_jump = np.full(number_post_jump, post_jump_value)
    smooth_original_data = np.concatenate((pre_jump, post_jump))
    original_data = smooth_original_data + noise_level * np.random.randn(total_number)

    # Creating the time axis
    time_axis = np.linspace(0, 1, total_number, endpoint=False)

    # Running the wavelet transform
    wavelet_transform, _ = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)

    # Returning the stuff we care about
    return original_data, wavelet_transform

# -------------------------------
# run_tests
# Runs jump analysis multiple times based on the method selected
# index         : integer
#                 The method selected, with 1 for jumps, 2 for alpha, and 
#                 3 for combined
# number_trials : integer
#                 The number of trials to run
# returns       : 0
#                 Means nothing
# -------------------------------

def run_tests(index, number_trials):
    if (index != 1 and index != 2 and index != 3):
        raise ValueError("Please pass a Suitable Index (1-3)")

    # Paremeters that can be modified for this analysis
    jump_threshold = 1.00
    alpha_threshold = 0.75
    jump_size = 1.00
    noise_level = 0.05

    if index != 1:
        jump_threshold_old = jump_threshold
        jump_threshold = noise_level * 20

    # Other paremeters and variables
    proper_result_with_jump = np.array([99])
    proper_result_without_jump = np.array([])
    jump_threshold = noise_level * 10
    false_positive_count = 0
    false_negative_count = 0
    success_with_jumps_count = 0
    success_without_jumps_count = 0
    errors_elsewhere_with_jumps_count = 0
    errors_elsewhere_without_jumps_count = 0

    # Running the analysis number_trials times with jumps
    for i in range(number_trials):
        # Getting the proper set of indexes based on the method
        data_set, wavelet_transform = create_jump_data(noise_level, jump_size)
        jump_locations = np.array([])
        if index == 1:
            jump_locations = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)
        elif index == 2:
            _, jump_locations = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)
        elif index == 3:
            behavior_jump_locations = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold_old)
            _, alpha_jump_locations = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)
            jump_locations = np.intersect1d(behavior_jump_locations, alpha_jump_locations)
        
        # Computing number of successes, as well as false positive and false negatives
        if 99 in jump_locations:
            success_with_jumps_count = success_with_jumps_count + 1
            if jump_locations.size != 1:
                errors_elsewhere_with_jumps_count = errors_elsewhere_with_jumps_count + jump_locations.size - 1
        else:
            false_negative_count = false_negative_count + 1
            if jump_locations.size != 0:
                errors_elsewhere_with_jumps_count = errors_elsewhere_with_jumps_count + jump_locations.size

    # Running the analysis number_trials times without jumps
    for i in range(number_trials):
         # Getting the proper set of indexes based on the method
        data_set, wavelet_transform = create_jump_data(noise_level, 0)
        jump_locations = np.array([])
        if index == 1:
            jump_locations = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)
        elif index == 2:
            _, jump_locations = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)
        elif index == 3:
            behavior_jump_locations = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold_old)
            _, alpha_jump_locations = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)
            jump_locations = np.intersect1d(behavior_jump_locations, alpha_jump_locations)
        
        # Computing number of successes, as well as false positive and false negatives
        if 99 in jump_locations:
            false_positive_count = false_positive_count + 1
            if jump_locations.size != 1:
                errors_elsewhere_without_jumps_count = errors_elsewhere_without_jumps_count + jump_locations.size - 1
        else:
            success_without_jumps_count = success_without_jumps_count + 1
            if jump_locations.size != 0:
                errors_elsewhere_without_jumps_count = errors_elsewhere_without_jumps_count + jump_locations.size

    # Text for the header
    header_text = ""
    if index == 1:
        header_text = "TESTING LOOKING FOR JUMPS BASED ON BEHAVIOR"
    elif index == 2:
        header_text = "TESTING LOOKING FOR JUMPS BASED ON ALPHA"
    elif index == 3:
        header_text = "TESTING LOOKING FOR JUMPS BASED ON BEHAVIOR AND ALPHA"
    
    # Text for the results
    success_with_jump_text = f"Number of Successes With Jumps {success_with_jumps_count} "
    success_without_jump_text = f"Number of Successes Without Jumps {success_without_jumps_count} "
    false_negative_text = f"Number of False Negatives (doesn't flag jump at 99 when there was one) {false_negative_count} "
    false_positive_text = f"Number of False Positives (flags jump at 99 when there was none) {false_positive_count} "
    other_error_with_jump_text = f"Number of Non-Jumps Flagged with Jump Present {errors_elsewhere_with_jumps_count} "
    other_error_without_jump_text = f"Number of Non-Jumps Flagged without Jump Present {errors_elsewhere_without_jumps_count} "

    # Printing the results
    print("-------------------------------------------------------------")
    Singularity_Analysis.print_colored(header_text, "magenta")
    Singularity_Analysis.print_colored(success_with_jump_text, "green")
    Singularity_Analysis.print_colored(success_without_jump_text, "green")
    Singularity_Analysis.print_colored(false_negative_text, "red")
    Singularity_Analysis.print_colored(false_positive_text, "red")
    Singularity_Analysis.print_colored(other_error_with_jump_text, "yellow")
    Singularity_Analysis.print_colored(other_error_without_jump_text, "yellow")
    print("-------------------------------------------------------------")
    print("")

    # Ending the function
    return 0

# Running the trials for our different functions
run_tests(1, 1000)
run_tests(2, 1000)
run_tests(3, 1000)