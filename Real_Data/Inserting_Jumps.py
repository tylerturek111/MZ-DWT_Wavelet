import Generating_Real_Data

import sys
import os

current_dir = os.path.dirname(__file__)
folder1_path = os.path.abspath(os.path.join(current_dir, '..', 'Wavelet_Code'))
sys.path.insert(0, folder1_path)

import MZ_Wavelet_Transforms
import Singularity_Analysis

sys.path.pop(0)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

# ()()()()()()()()()()()()()
# This file tests how the function works by adding in jumps
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the flag noise ratio (jump threshold equivalent)
flag_noise_ratio_start = 4
flag_noise_ratio_end = 5
flag_noise_ratio_step = 0.1

# Parameters for analyzing the alpha thresholds
alpha_threshold_start = 0
alpha_threshold_end = 1.00
alpha_threshold_step = 0.1

# Parameters for the jump size
jump_noise_ratio_start = 10
jump_noise_ratio_end = 10
jump_noise_ratio_step = 1
jump_location = 50000

# Parameters for the detectors to look at and the data points
detector_list = np.array([0, 1, 3, 6, 8, 9, 11, 14, 18, 21, 23])
start_index = 0
total_number = 100000

# Parameters for number of scales for the wavelet transform
number_scales = 3

# Other parameters
old_alpha_thresold = 0.1
smoothing_index = 100
window_size = 100
number_trials = 1
sqrt_band = 10

# -------------------------------
# create_jump_data
# Adds the proper jump to the data
# detector         : integer
#                    The detector to look at
# start            : integer
#                    The start index of the data
# length           : integer
#                    The length of data to look at
# jump_noise_ratio : integer
#                    Ratio between jump and noise
# jump_location    : integer
#                    The location of the jump
# R, r, r          : 1-D Numpy Array of floats
#                    The signal itself
# r, R, r          : 1-D Numpy Array of floats
#                    The wavelet transform data
# r, r, R          : integer
#                    The noise level
# -------------------------------

def create_jump_data(detector, start, length, jump_noise_ratio, jump_location):
    # Getting the real data
    raw_data, noise_level = Generating_Real_Data.get_real_data(start, length, detector)

    # Adding in the jump
    added_jump = np.zeros(length, dtype=float)
    added_jump[jump_location:] = jump_noise_ratio * sqrt_band * noise_level
    original_data = raw_data + added_jump

    # Running the wavelet transform
    wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
    processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)

    # Returning the stuff we care about
    return original_data, wavelet_transform, noise_level

# -------------------------------
# run_test
# Runs jump analysis for one specific criteria
# jump              : integer
#                     Jump threshold to be tested
# alpha             : integer
#                     Alpha threshold to be tested
# transform         : array
#                     The transform array
# jump_location     : integer
#                     The location of the jump
# RETURNS, returns  : integer
#                     1 if false positive occured, 0 otherwise
# returns, RETURNS  : integer
#                     1 if false negative occured, 0 otherwise
# -------------------------------

def run_test(jump, alpha, transform, jump_location):
    # Flags other jumps
    false_positive_count = 0.0
    # Doesn't flag the jump
    false_negative_count = 0.0

    # Computing jump locations with the developed methods
    jump_locations, temp = Singularity_Analysis.alpha_and_behavior_jumps(transform, jump, alpha)
    print(jump, alpha, jump_locations)
    # Computing the number of false positives and false negatives
    if jump_location in jump_locations:
        if jump_locations.size != 1:
            false_positive_count = 1
    else:
        false_negative_count = 1
        if jump_locations.size != 0:
            false_positive_count = 1

    # Returning the results
    return false_positive_count, false_negative_count

# -------------------------------
# determine_accuracy
# Runs jump analysis with varrying parameters and generate heat maps to determine accuracy
# ratios            : array
#                     The flag noise ratio to be tested
# alphas            : array
#                     The alpha values to be tested
# jump_location     : integer
#                     The location of the jump
# jump_noise_ratio  : integer
#                     Ratio between jump and noise
# detector          : array
#                     The array of detectors to look at
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def determine_accuracy(ratios, alphas, jump_location, jump_noise_ratio, detectors):
    # Saving false positive (falsely flags other points) and false negatives (doesn't flag the jump)
    false_positives = np.empty((ratios.size, alphas.size))
    false_negatives = np.empty((ratios.size, alphas.size))

    for detector in range(detectors.size):
        # Creating the data
        data, transform, noise = create_jump_data(detectors[detector], start_index, total_number, jump_noise_ratio, jump_location + 1)

        # Creating a plot of the data
        plt.plot(time_axis, data)
        plt.axvline(x = jump_location - 1, color = "r", linestyle = "--")
        file_name = f"Signal_{detector}.png"
        path = os.path.join(heat_maps_path, file_name)
        plt.savefig(path)
        plt.close()

        # Running this for the different sets of ratio's and alphas for each detector
        for ratio in range(ratios.size):
            for alpha in range(alphas.size):
                false_positive, false_negative = run_test(ratios[ratio] * sqrt_band * noise, alphas[alpha], transform, jump_location)
                false_positives[ratio, alpha] = false_positives[ratio, alpha] + false_positive 
                false_negatives[ratio, alpha] = false_negatives[ratio, alpha] + false_negative

    # Getting the counts to become ratios
    false_positives = false_positives / detectors.size
    false_negatives = false_negatives / detectors.size

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Plots of false negative ratios
    colors2 = [(0, 1, 0), (1, 0, 0)] 
    cmap_name2 = 'green_red'
    custom_cmap2 = mcolors.LinearSegmentedColormap.from_list(cmap_name2, colors2, N = 256)
    im2 = axs[0].imshow(false_negatives, cmap = custom_cmap2, interpolation = 'nearest', vmin = 0.01)
    title_text2 = f"False Negative Rate with Jump/Noise ratio of {jump_noise_ratio} \n (Green (red) for does (doesn't) flag the artificial jump)"
    axs[0].set_title(title_text2, fontsize = 10)
    axs[0].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[0].set_ylabel('Noise to Jump Threshold Ratio', fontsize = 8)
    fig.colorbar(im2, ax = axs[0], shrink = 0.2, pad = 0.05)
    axs[0].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[0].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[0].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[0].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[0].set_yticklabels(ytick_labels, fontsize = 6)
    for i in range(false_positives.shape[0]):
        for j in range(false_positives.shape[0]):
            text = axs[0].text(j, i, f'{false_negatives[i, j]:.2f}', ha='center', va='center', color='black', fontsize=4)

    # Plot of false positive ratio
    colors1 = [(0, 1, 0), (1, 0, 0)] 
    cmap_name1 = 'green_red'
    custom_cmap1 = mcolors.LinearSegmentedColormap.from_list(cmap_name1, colors1, N = 256)
    im1 = axs[1].imshow(false_positives, cmap = custom_cmap1, interpolation = 'nearest', vmax = 0.99)
    title_text1 = f"False Positive Rate with Jump/Noise Ratio of {jump_noise_ratio} \n (Green (red) for doesn't (does) flag other jumps)"
    axs[1].set_title(title_text1, fontsize = 10)
    axs[1].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[1].set_ylabel('Noise to Jump Threshold Ratio', fontsize = 8)
    fig.colorbar(im1, ax = axs[1], shrink = 0.2, pad = 0.05)
    axs[1].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[1].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[1].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[1].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[1].set_yticklabels(ytick_labels, fontsize = 6)
    for i in range(false_positives.shape[1]):
        for j in range(false_positives.shape[1]):
            text = axs[1].text(j, i, f'{false_positives[i, j]:.2f}', ha='center', va='center', color='black', fontsize=4)

    # Saving the plots
    plt.tight_layout()

    # Saving the plot
    file_name = f"Heat_Map_{jump_noise_ratio}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

    return 0

# -------------------------------
# Running simulation of accuracy for numers different jump to noise ratios
# -------------------------------

# Setting up the path to store the figures
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
heat_maps = 'Heat_Maps'
heat_maps_path = os.path.join(SO_location, heat_maps)
if not os.path.exists(heat_maps_path):
    os.makedirs(heat_maps_path)
        
time_axis = np.linspace(0, total_number, total_number, endpoint=False)

# Setting up the parameters that can be changed
ratios = np.arange(flag_noise_ratio_start, flag_noise_ratio_end + flag_noise_ratio_step, flag_noise_ratio_step)
alphas = np.arange(alpha_threshold_start, alpha_threshold_end + alpha_threshold_step, alpha_threshold_step)
added_jump = np.arange(jump_noise_ratio_start, jump_noise_ratio_end + jump_noise_ratio_step, jump_noise_ratio_step)

# Actually running the code
for jump_noise_ratio in added_jump:
        # Running the simulations
        _ = determine_accuracy(ratios, alphas, jump_location - 1, jump_noise_ratio, detector_list)

plt.close('all')