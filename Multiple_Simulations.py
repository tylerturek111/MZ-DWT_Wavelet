import MZ_Wavelet_Transforms
import Singularity_Analysis

import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
# index              : integer
#                     The method selected, with 1 for jumps, 2 for alpha, and 3 for combined
# number_trials     : integer
#                     The number of trials to run
# jump_threshold    : integer
#                     The jump threshold
# alpha_threshold   : integer
#                     The alpha threshold
# jump_noise_ratio  : integer
#                     Ratio between size of noise and the jump
# R r r r           : integer
#                     Success ratio of finding the jump when it is there
# r R r r           : integer
#                     Number of errors elsewhere when there is a jump
# r r R r           : integer
#                     Success ratio of not finding the jump when there is none
# r r r R           : integer
#                     Number of errors elsewhere when there is no jump
# -------------------------------

def run_tests(index, number_trials, jump_threshold, alpha_threshold, jump_noise_ratio):
    if (index != 1 and index != 2 and index != 3):
        raise ValueError("Please pass a Suitable Index (1-3)")

    # Paremeters that can be modified for this analysis
    jump_threshold = jump_threshold
    alpha_threshold = alpha_threshold
    noise_level = 0.05
    jump_size = noise_level * jump_noise_ratio

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
    
    # # Text for the results
    # success_with_jump_text = f"Number of Successes With Jumps {success_with_jumps_count} "
    # success_without_jump_text = f"Number of Successes Without Jumps {success_without_jumps_count} "
    # false_negative_text = f"Number of False Negatives (doesn't flag jump at 99 when there was one) {false_negative_count} "
    # false_positive_text = f"Number of False Positives (flags jump at 99 when there was none) {false_positive_count} "
    # other_error_with_jump_text = f"Number of Non-Jumps Flagged with Jump Present {errors_elsewhere_with_jumps_count} "
    # other_error_without_jump_text = f"Number of Non-Jumps Flagged without Jump Present {errors_elsewhere_without_jumps_count} "

    # # Printing the results
    # print("-------------------------------------------------------------")
    # Singularity_Analysis.print_colored(header_text, "magenta")
    # Singularity_Analysis.print_colored(success_with_jump_text, "green")
    # Singularity_Analysis.print_colored(success_without_jump_text, "green")
    # Singularity_Analysis.print_colored(false_negative_text, "red")
    # Singularity_Analysis.print_colored(false_positive_text, "red")
    # Singularity_Analysis.print_colored(other_error_with_jump_text, "yellow")
    # Singularity_Analysis.print_colored(other_error_without_jump_text, "yellow")
    # print("-------------------------------------------------------------")
    # print("")

    # Values to be returned
    success_value_with_jump = success_with_jumps_count/number_trials
    other_error_value_with_jump = math.log(errors_elsewhere_with_jumps_count + 1)
    success_value_without_jump = success_without_jumps_count/number_trials
    other_error_value_without_jump = math.log(errors_elsewhere_without_jumps_count + 1)
    
    return success_value_with_jump, other_error_value_with_jump, success_value_without_jump, other_error_value_without_jump

# -------------------------------
# simulate_accuracy
# Runs tests with various thresholds to determine accuracy and printing the results in a heatmap
# jump_max          : integer
#                     The maximum jump threshold
# alpha_max         : integer
#                     The maximum alpha threshold
# jump_noise_ratio  : integer
#                     Ratio between size of noise and the jump
# R r r r           : integer
#                     Success ratio of finding the jump when it is there
# r R r r           : integer
#                     Number of errors elsewhere when there is a jump
# r r R r           : integer
#                     Success ratio of not finding the jump when there is none
# r r r R           : integer
#                     Number of errors elsewhere when there is no jump
# -------------------------------

def simulate_accuracy(jump_max, alpha_max, jump_noise_ratio):
    # Some parameters that can be modified
    jump_threshold_low = 0
    jump_threshold_high = jump_max
    alpha_threshold_low = 0
    alpha_threshold_high = alpha_max
    number_datapoints = 10
    number_trials = 100

    # Some other parameters and variables
    success_count_with_jump = np.empty((number_datapoints, number_datapoints))
    other_error_count_with_jump = np.empty((number_datapoints, number_datapoints))
    success_count_without_jump = np.empty((number_datapoints, number_datapoints))
    other_error_count_without_jump = np.empty((number_datapoints, number_datapoints))
    jump_threshold_values = np.linspace(jump_threshold_low, jump_threshold_high, number_datapoints)
    alpha_threshold_values = np.linspace(alpha_threshold_low, alpha_threshold_high, number_datapoints)

    for i in range(number_datapoints):
        for j in range(number_datapoints):
            current_success_with_jump, current_error_with_jump, current_success_without_jump, current_error_without_jump = run_tests(3, number_trials, jump_threshold_values[i], alpha_threshold_values[j], jump_noise_ratio)
            success_count_with_jump[i][j] = current_success_with_jump
            other_error_count_with_jump[i][j] = current_error_with_jump
            success_count_without_jump[i][j] = current_success_without_jump
            other_error_count_without_jump[i][j] = current_error_without_jump    

    # Plotting results
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))

    # Plot of successes with jump
    colors1 = [(1, 0, 0), (0, 1, 0)] 
    cmap_name1 = 'red_green'
    custom_cmap1 = mcolors.LinearSegmentedColormap.from_list(cmap_name1, colors1, N = 256)
    custom_cmap1.set_over('blue')
    im1 = axs[0, 0].imshow(success_count_with_jump, cmap = custom_cmap1, interpolation = 'nearest', vmax = 0.99)
    title_text1 = f"Succesfully Detecting Jumps for Different Thresholds with Ratio of {jump_noise_ratio} "
    axs[0, 0].set_title(title_text1, fontsize = 10)
    axs[0, 0].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[0, 0].set_ylabel('Jump Threshold', fontsize = 8)
    fig.colorbar(im1, ax = axs[0, 0], shrink = 0.6, pad = 0.05)
    axs[0, 0].invert_yaxis()
    xticks = np.linspace(alpha_threshold_low, alpha_threshold_high, 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[0, 0].set_xticks(np.linspace(0, number_datapoints - 1, 11))
    axs[0, 0].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(jump_threshold_low, jump_threshold_high, 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[0, 0].set_yticks(np.linspace(0, number_datapoints - 1, 11))
    axs[0, 0].set_yticklabels(ytick_labels, fontsize = 6)

    # Plot of other jumps with jump
    colors2 = [(0, 1, 0), (1, 0, 0)] 
    cmap_name2 = 'red_green'
    custom_cmap2 = mcolors.LinearSegmentedColormap.from_list(cmap_name2, colors2, N = 256)
    custom_cmap2.set_under('blue')
    im2 = axs[0, 1].imshow(other_error_count_with_jump, cmap = custom_cmap2, interpolation = 'nearest', vmin = 0.01)
    title_text2 = f"Log of Other Jumps Detected for Different Thresholds w/ Jumps with Ratio of {jump_noise_ratio} "
    axs[0, 1].set_title(title_text2, fontsize = 10)
    axs[0, 1].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[0, 1].set_ylabel('Jump Threshold', fontsize = 8)
    fig.colorbar(im2, ax = axs[0, 1], shrink = 0.6, pad = 0.05)
    axs[0, 1].invert_yaxis()
    xticks = np.linspace(alpha_threshold_low, alpha_threshold_high, 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[0, 1].set_xticks(np.linspace(0, number_datapoints - 1, 11))
    axs[0, 1].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(jump_threshold_low, jump_threshold_high, 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[0, 1].set_yticks(np.linspace(0, number_datapoints - 1, 11))
    axs[0, 1].set_yticklabels(ytick_labels, fontsize = 6)

    # Plot of successes without jump
    im1 = axs[1, 0].imshow(success_count_without_jump, cmap = custom_cmap1, interpolation = 'nearest', vmax = 0.99)
    title_text3 = f"Succesfully Not Detecting Jumps for Different Thresholds with Ratio of {jump_noise_ratio} "
    axs[1, 0].set_title(title_text3, fontsize = 10)
    axs[1, 0].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[1, 0].set_ylabel('Jump Threshold', fontsize = 8)
    fig.colorbar(im1, ax = axs[1, 0], shrink = 0.6, pad = 0.05)
    axs[1, 0].invert_yaxis()
    xticks = np.linspace(alpha_threshold_low, alpha_threshold_high, 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[1, 0].set_xticks(np.linspace(0, number_datapoints - 1, 11))
    axs[1, 0].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(jump_threshold_low, jump_threshold_high, 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[1, 0].set_yticks(np.linspace(0, number_datapoints - 1, 11))
    axs[1, 0].set_yticklabels(ytick_labels, fontsize = 6)

    # Plot of other jumps without jump
    im2 = axs[1, 1].imshow(other_error_count_without_jump, cmap = custom_cmap2, interpolation = 'nearest', vmin = 0.01)
    title_text4 = f"Log of Other Jumps Detected for Different Thresholds w/out Jumps with Ratio of {jump_noise_ratio} "
    axs[1, 1].set_title(title_text4, fontsize = 10)
    axs[1, 1].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[1, 1].set_ylabel('Jump Threshold', fontsize = 8)
    fig.colorbar(im2, ax = axs[1, 1], shrink = 0.6, pad = 0.05)
    axs[1, 1].invert_yaxis()
    xticks = np.linspace(alpha_threshold_low, alpha_threshold_high, 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[1, 1].set_xticks(np.linspace(0, number_datapoints - 1, 11))
    axs[1, 1].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(jump_threshold_low, jump_threshold_high, 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[1, 1].set_yticks(np.linspace(0, number_datapoints - 1, 11))
    axs[1, 1].set_yticklabels(ytick_labels, fontsize = 6)

    # Showing the plots
    plt.tight_layout()
    plt.show()


    # Now making a combined plot
    combined = (1 - success_count_with_jump) * 10 + other_error_count_with_jump + (1 - success_count_without_jump) * 10 + other_error_count_without_jump
    
    # Plotting results
    fig1, ax = fig, ax = plt.subplots(figsize=(10, 5))
    colors3 = [(0, 1, 0), (1, 0, 0)]  # Blue, Green, Red
    cmap_name3 = 'custom_color_scale'
    custom_cmap3 = mcolors.LinearSegmentedColormap.from_list(cmap_name3, colors3, N=256)
    custom_cmap3.set_under('blue')
    # Plot of combined results
    im2 = ax.imshow(combined, cmap = custom_cmap3, interpolation = 'nearest', vmin = 0.01)
    title_text5 = f"Combined Results for Different Thresholds with Ratio of {jump_noise_ratio} "
    ax.set_title(title_text5, fontsize = 12)
    ax.set_xlabel('Alpha Threshold', fontsize = 10)
    ax.set_ylabel('Jump Threshold', fontsize = 10)
    fig.colorbar(im2, ax = ax, shrink = 0.8, pad = 0.05)
    ax.invert_yaxis()
    xticks = np.linspace(alpha_threshold_low, alpha_threshold_high, 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    ax.set_xticks(np.linspace(0, number_datapoints - 1, 11))
    ax.set_xticklabels(xtick_labels, fontsize = 8)
    yticks = np.linspace(jump_threshold_low, jump_threshold_high, 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    ax.set_yticks(np.linspace(0, number_datapoints - 1, 11))
    ax.set_yticklabels(ytick_labels, fontsize = 8)

    plt.show()

# -------------------------------
# Running simulation of accuracy for numers different jump to noise ratios
# -------------------------------

# Jump to noise ratios that we care about
jump_noise_ratio_values = np.arange(20, 21, 1)

# Actually running the code
for i in range(jump_noise_ratio_values.size):
    simulate_accuracy(2, 2, jump_noise_ratio_values[i])