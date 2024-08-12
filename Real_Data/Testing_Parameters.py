import sys
import os

current_dir = os.path.dirname(__file__)
folder1_path = os.path.abspath(os.path.join(current_dir, '..', 'Wavelet_Code'))
sys.path.insert(0, folder1_path)

import MZ_Wavelet_Transforms
import Singularity_Analysis

sys.path.pop(0)

import Generating_Real_Data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import shutil
import math

# ()()()()()()()()()()()()()
# This file tests how the functions behave with various parameters
# ()()()()()()()()()()()()()

# -------------------------------
# Defining some variables
# -------------------------------

# Number of scales for the wavelet transform
number_scales = 3

# Square root of the band's frequency
sqrt_frequency = 10


# Global Variables
alpha_combined = np.array([])
false_positive_combined = np.array([])
false_negative_combined = np.array([])

# -------------------------------
# Setting up the storage location
# -------------------------------

# Location for the plots to be stored for heat maps (Part 1)
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
heat_maps_folder = 'Heat_Maps'
heat_maps_path = os.path.join(SO_location, heat_maps_folder)
if os.path.exists(heat_maps_path):
    shutil.rmtree(heat_maps_path)
os.makedirs(heat_maps_path)

# Location for the plots to be stored for parameter testing (Part 2)
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
parameter_tests_folder = 'Parameter_Tests'
parameter_tests_path = os.path.join(SO_location, parameter_tests_folder)
if os.path.exists(parameter_tests_path):
    shutil.rmtree(parameter_tests_path)
os.makedirs(parameter_tests_path)

# -------------------------------
# -------------------------------
# PART 1
# -------------------------------
# -------------------------------

# -------------------------------
# generate_detectors
# Gets a list of detectors that are suspected to not have anomalies in them
# detector_start      : integer
#                       The detector to start at
# detector_end        : integer
#                       The detector to end at
# time_start          : integer
#                       The signal's time to start at
# time_end            : integer
#                       The signal's time to end at
# ratio               : integer
#                       Ratio of anomaly threshold / noise ratio
# alpha               : integer
#                       The alpha threshold
# RETURNS, returns    : Array
#                       Array of the detectors that do not appear to have an anomaly
# -------------------------------

def generate_detectors(detector_start, detector_end, time_start, time_end, ratio, alpha):
    # Where the valid detectors will be stored
    detectors = np.empty(0, dtype = int)
    noises = np.empty(0, dtype = float)

    # Iterating through for each possible detector
    for detector in range(detector_start, detector_end):
        # Getting the proper real data
        raw_data, noise = Generating_Real_Data.get_real_data(time_start, time_end, detector)
        
        # Computing wavelet transform
        transform, _ = MZ_Wavelet_Transforms.forward_wavelet_transform(3, raw_data)

        # Computing possible anomalies
        anomalies, _ = Singularity_Analysis.alpha_and_behavior_jumps(transform, ratio * 10 * noise, alpha)

        # Testing if no anomalies are present
        if anomalies.size == 0:
            detectors = np.append(detectors, detector)

    return detectors


# -------------------------------
# create_jump_data
# Artificially insert a jump into the data
# detector         : integer
#                    The detector to look at
# time_start       : integer
#                    The start index of the data
# time_end         : integer
#                    The length of data to look at
# jump_ratio       : integer
#                    Ratio of jump / noise
# jump_location    : integer
#                    The location of the jump
# R, r, r          : 1-D Numpy Array of floats
#                    The signal itself
# r, R, r          : 1-D Numpy Array of floats
#                    The wavelet transform data
# r, r, R          : integer
#                    The noise level
# -------------------------------c

def create_jump_data(detector, time_start, time_end, jump_ratio, jump_location):
    # Getting the real data
    raw_data, noise = Generating_Real_Data.get_real_data(time_start, time_end, detector)

    # Adding in the jump
    added_jump = np.zeros(time_end - time_start, dtype = float)
    added_jump[jump_location:] = jump_ratio * sqrt_frequency * noise
    original_data = raw_data + added_jump

    # Running the wavelet transform
    wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)

    # Returning the stuff we care about
    return original_data, wavelet_transform, noise

# -------------------------------
# create_glitch_data
# Artificially insert a glitch into the data
# detector         : integer
#                    The detector to look at
# time_start       : integer
#                    The start index of the data
# time_end         : integer
#                    The length of data to look at
# glitch_ratio     : integer
#                    Ratio of jump / noise
# glitch_location  : integer
#                    The location of the jump
# glitch_size      : integer
#                    The size of the glitch
# R, r, r          : 1-D Numpy Array of floats
#                    The signal itself
# r, R, r          : 1-D Numpy Array of floats
#                    The wavelet transform data
# r, r, R          : integer
#                    The noise level
# -------------------------------c

def create_glitch_data(detector, time_start, time_end, glitch_ratio, glitch_location, glitch_size):
    # Getting the real data
    raw_data, noise = Generating_Real_Data.get_real_data(time_start, time_end, detector)

    # Adding in the glitch
    added_glitch = np.zeros(time_end - time_start, dtype = float)
    added_glitch[glitch_location:(glitch_location + glitch_size)] = glitch_ratio * sqrt_frequency * noise
    original_data = raw_data + added_glitch

    # Running the wavelet transform
    wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)

    # Returning the stuff we care about
    return original_data, wavelet_transform, noise

# -------------------------------
# run_test
# Runs anomalies analysis for one specific criteria
# anomaly_type      : integer
#                     The type of analysis to be done (0 for jumps, n for glitches of size n)
# anomaly           : integer
#                     Anomaly threshold to be tested
# alpha             : integer
#                     Alpha threshold to be tested
# transform         : array
#                     The transform array
# anomaly_location  : integer
#                     The location of the anomaly
# RETURNS, returns  : integer
#                     1 if false positive occured, 0 otherwise
# returns, RETURNS  : integer
#                     1 if false negative occured, 0 otherwise
# Temporary RETURNS : integer
# -------------------------------

def run_test(anomaly_type, anomaly, alpha, transform, anomaly_location):
    # Flags other anomalies
    false_positive_count = 0.0

    # Doesn't flag the anomaly
    false_negative_count = 0.0

    # Computing anomaly locations with the developed methods
    anomaly_locations, alpha_values = Singularity_Analysis.alpha_and_behavior_jumps(transform, anomaly, alpha)
    
    # Computing the number of false positives and false negatives
    # In the case of a jump
    if anomaly_type == 0:
        if anomaly_location in anomaly_locations:
            if anomaly_locations.size != 1:
                false_positive_count = 1
        else:
            false_negative_count = 1
            if anomaly_locations.size != 0:
                false_positive_count = 1
    # In the case of a glitch
    else:
        if anomaly_location in anomaly_locations and (anomaly_location + anomaly_type) in anomaly_locations:
            if anomaly_locations.size != 2:
                false_positive_count = 1
        else:
            false_negative_count = 1
            if anomaly_location in anomaly_locations or (anomaly_location + anomaly_type) in anomaly_locations:
                if anomaly_locations.size != 1:
                    false_positive_count = 1
            else:
                if anomaly_locations.size != 0:
                    false_positive_count = 1
    
    # Returning the results
    return false_positive_count, false_negative_count, alpha_values

# -------------------------------
# determine_accuracy
# Runs anomaly analysis with varrying parameters and generate heat maps to determine accuracy
# anomaly_type      : integer
#                     The type of analysis to be done (0 for jumps, n for glitches of size n)
# ratios            : array
#                     Ratios of anomaly threshold / noise
# alphas            : array
#                     The alpha values to be tested
# anomaly_ratio     : integer
#                     Ratio of anomaly / noise
# detector_start    : integer
#                     The detector to start at
# detector_end      : integer
#                     The detector to end at
# time_start        : integer
#                     The start index of the data
# time_end          : integer
#                     The length of data to look at
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def determine_accuracy(anomaly_type, ratios, alphas, anomaly_ratio, detector_start, detector_end, time_start, time_end):
    # Getting the detectors that we care about
    detectors = generate_detectors(detector_start, detector_end, time_start, time_end, 5, 0.75)

    # Saving false positive (falsely flags other points) and false negatives (doesn't flag the anomaly)
    false_positives = np.empty((ratios.size, alphas.size))
    false_negatives = np.empty((ratios.size, alphas.size))

    # Setting up alphas
    alpha_values = np.array([])
    alpha_data = np.array([])
    global alpha_combined

    # Setting up false positive and false negative global variables
    global false_positive_combined
    global false_negative_combined
    if false_positive_combined.size == 0:
        false_positive_combined = np.empty((ratios.size, alphas.size))
    if false_negative_combined.size == 0:
        false_negative_combined = np.empty((ratios.size, alphas.size))

    for detector in range(detectors.size):
        # Getting a random anomaly location (included a little bit of buffer)
        anomaly_location = random.randint(time_start + 100, time_end - 100)

        print("Current Detector:", detectors[detector], "Anomaly of Size:", anomaly_type)

        # Creating the data
        if anomaly_type == 0:
            data, transform, noise = create_jump_data(detectors[detector], time_start, time_end, anomaly_ratio, anomaly_location)
            anomaly_text = "Jump"
            anomaly_text0 = "Jump"
        else:
            data, transform, noise = create_glitch_data(detectors[detector], time_start, time_end, anomaly_ratio, anomaly_location, anomaly_type)        
            anomaly_text = f"Glitch of Size {anomaly_type}"
            anomaly_text0 = "Glitch"

        # The time axis
        time_axis = np.linspace(0, time_end - time_start, time_end - time_start, endpoint=False)

        # # Creating a plot of the data
        # plt.plot(time_axis, data)
        # plt.axvline(x = anomaly_location - 1, color = "r", linestyle = "--")
        # file_name = f"Signal_{detectors[detector]}.png"
        # path = os.path.join(heat_maps_path, file_name)
        # plt.savefig(path)
        # plt.close()

        # # Creating a plot of the transform
        # plt.plot(time_axis, transform)
        # plt.axhline(y = noise * 10 * ratios[0], color = "g", linestyle = "--")
        # plt.axhline(y = - 1 * noise * 10 * ratios[0], color = "g", linestyle = "--")
        # plt.axhline(y = noise * 10 * ratios[ratios.size - 1], color = "r", linestyle = "--")
        # plt.axhline(y = - 1 * noise * 10 * ratios[ratios.size - 1], color = "r", linestyle = "--")
        # file_name = f"Transform_{detectors[detector]}.png"
        # path = os.path.join(heat_maps_path, file_name)
        # plt.savefig(path)
        # plt.close()

        # Running this for the different sets of ratio's and alphas for each detector
        run_through = 0
        for ratio in range(ratios.size):
            for alpha in range(alphas.size):
                if run_through == 0:
                    false_positive, false_negative, alpha_data = run_test(anomaly_type, ratios[ratio] * sqrt_frequency * noise, alphas[alpha], transform, anomaly_location - 1)
                    run_through = 1
                else:
                    false_positive, false_negative, _ = run_test(anomaly_type, ratios[ratio] * sqrt_frequency * noise, alphas[alpha], transform, anomaly_location - 1)
                false_positives[ratio, alpha] = false_positives[ratio, alpha] + false_positive 
                false_negatives[ratio, alpha] = false_negatives[ratio, alpha] + false_negative
        
        # Saving alpha values that we care about
        alpha_values = np.append(alpha_values, alpha_data[anomaly_location - 1])
        alpha_values = np.append(alpha_values, alpha_data[anomaly_location + anomaly_type - 1])

        # # Creating a plot of the alphas
        # time_axis2 = np.linspace(anomaly_location - 10, anomaly_location + 10, 20, endpoint=False)
        # plt.plot(time_axis2, alpha_data[anomaly_location - 10 : anomaly_location + 10])
        # plt.axvline(x = anomaly_location - 1, color = "g", linestyle = "--")
        # plt.axvline(x = anomaly_location + anomaly_type - 1, color = "g", linestyle = "--")
        # file_name = f"Alpha_{anomaly_type}_{detectors[detector]}.png"
        # path = os.path.join(heat_maps_path, file_name)
        # plt.savefig(path)
        # plt.close()

    # Getting the counts to become ratios
    false_positives = false_positives / detectors.size
    false_negatives = false_negatives / detectors.size
    combined = false_positives + false_negatives

    # Getting global count
    false_positive_combined = false_positive_combined + false_positives
    false_negative_combined = false_negative_combined + false_negatives

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Colors
    colors = [(0, 1, 0), (1, 0, 0)] 
    cmap_name = 'green_red'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N = 256)

    # Plots of false negative ratio
    im1 = axs[0].imshow(false_negatives, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 1.0)
    title_text1 = f"False Negative Rate with Artificially Added {anomaly_text} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Does (Doesn't) Flags the Artificial {anomaly_text0}"
    axs[0].set_title(title_text1, fontsize = 10)
    axs[0].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[0].set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 8)
    fig.colorbar(im1, ax = axs[0], shrink = 0.2, pad = 0.05)
    axs[0].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[0].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[0].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[0].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[0].set_yticklabels(ytick_labels, fontsize = 6)
    if np.max(ratios) > anomaly_ratio:        
        axs[0].axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(false_negatives.shape[0]):
        for j in range(false_negatives.shape[1]):
            text = axs[0].text(j, i, f'{false_negatives[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 4)

    # Plot of false positive ratio
    im2 = axs[1].imshow(false_positives, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 1.0)
    title_text2 = f"False Positive Rate with Artificially Added {anomaly_text} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Doesn't (Does) Flag Other Anomalies"
    axs[1].set_title(title_text2, fontsize = 10)
    axs[1].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[1].set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 8)
    fig.colorbar(im2, ax = axs[1], shrink = 0.2, pad = 0.05)
    axs[1].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[1].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[1].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[1].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[1].set_yticklabels(ytick_labels, fontsize = 6)
    if np.max(ratios) > anomaly_ratio:        
        axs[1].axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(false_positives.shape[0]):
        for j in range(false_positives.shape[1]):
            text = axs[1].text(j, i, f'{false_positives[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 4)
    plt.tight_layout()
    file_name = f"Heat_Map_{anomaly_text0}_{anomaly_ratio}_{anomaly_type}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

    # Plot of combined information
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    im3 = axs.imshow(combined, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 2.0)
    title_text2 = f"Combined Results with Artificially Added {anomaly_text} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Good (Bad)"
    axs.set_title(title_text2, fontsize = 14)
    axs.set_xlabel('Alpha Threshold', fontsize = 12)
    axs.set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 12)
    fig.colorbar(im3, ax = axs, shrink = 0.2, pad = 0.05)
    axs.invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs.set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs.set_xticklabels(xtick_labels, fontsize = 10)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs.set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs.set_yticklabels(ytick_labels, fontsize = 10)
    if np.max(ratios) > anomaly_ratio:        
        axs.axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            text = axs.text(j, i, f'{combined[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 8)
    file_name = f"Combined_Heat_Map_{anomaly_text0}_{anomaly_type}_{anomaly_ratio}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

    # Plot of the alpha values
    bin_values = np.arange(-2, 2 + 0.10, 0.10)
    if anomaly_type !=  0:
        # Saving the alpha values
        if alpha_combined.size == 0:
            alpha_combined = alpha_values
        else:
            alpha_combined = np.concatenate((alpha_combined, alpha_values))

        # Plotting alpha values for a particular glitch size
        plt.hist(alpha_values, bins = bin_values, edgecolor='black')
        title_text3 = f"Histograph of Alpha Values with {anomaly_text}"
        plt.title(title_text3, fontsize = 14)
        plt.xlabel('Alpha Value')
        plt.ylabel('Frequency')
        plt.xlim(-2, 2)
        file_name = f"Alpha_Values_{anomaly_text0}_{anomaly_type}.png"
        path = os.path.join(heat_maps_path, file_name)
        plt.savefig(path)

        plt.close()

        # Setting up our particular parameters
        log_alpha_values = np.abs(alpha_values)
        log_bin_values = np.logspace(-1, 0, 41)

        # Plotting the alpha values for a particular glitch size with a log scale
        plt.hist(log_alpha_values, bins = log_bin_values, edgecolor='black')
        title_text3 = f"Logarithmic Histograph of Alpha Values with {anomaly_text}"
        plt.title(title_text3, fontsize = 14)
        plt.xscale('log')
        plt.xlabel('Log of Alpha Value')
        plt.ylabel('Frequency')
        file_name = f"Alpha_Values_Log_{anomaly_text0}_{anomaly_type}.png"
        path = os.path.join(heat_maps_path, file_name)
        plt.savefig(path)

        plt.close()
    
    return 0

# -------------------------------
# generate_summary_plots
# Generates summary plots of the data
# anomaly_type      : integer
#                     The type of analysis to be done (0 for jumps, n for glitches of size n)
# anomaly_ratio     : integer
#                     Ratio of anomaly / noise
# ratios            : array
#                     Ratios of anomaly threshold / noise
# alphas            : array
#                     The alpha values to be tested
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def generate_summary_plots(anomaly_type, anomaly_ratio, ratios, alphas):
    # Plotting all the alpha values
    plt.hist(alpha_combined, bins=10, edgecolor='black')
    title_text4 = f"Histograph of Alpha Values for all Glitch Sizes"
    plt.title(title_text4, fontsize = 14)
    plt.xlabel('Alpha Value')
    plt.ylabel('Frequency')
    plt.xlim(-2, 2)
    file_name = f"SUMMARY_Alpha_Values.png"
    path = os.path.join(heat_maps_path, file_name)
    plt.savefig(path)

    plt.close()

    # Setting up our particular parameters
    log_alpha_combined = np.abs(alpha_combined)
    log_bin_values = np.logspace(-1, 0, 41)

    # Plotting all the alpha values with a log scale
    plt.hist(log_alpha_combined, bins = log_bin_values, edgecolor='black')
    title_text3 = f"Logirthmic Histograph of Alpha Values for all Glitch Sizes"
    plt.title(title_text3, fontsize = 14)
    plt.xscale('log')
    plt.xlabel('Log of Alpha Value')
    plt.ylabel('Frequency')
    file_name = f"SUMMARY_Alpha_Values_Log.png"
    path = os.path.join(heat_maps_path, file_name)
    plt.savefig(path)

    plt.close()

    # Setting up stuff for plots
    global false_positive_combined
    global false_negative_combined
    combined_combined = false_positive_combined + false_negative_combined
    if anomaly_type == 0:
        anomaly_text = "Jump"
        anomaly_text0 = "Jump"
    else:
        anomaly_text = f"Glitch of Size {anomaly_type}"
        anomaly_text0 = "Glitch"

    # Plotting false positive and false negative combined
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    colors = [(0, 1, 0), (1, 0, 0)] 
    cmap_name = 'green_red'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N = 256)

    # Plots of false negative ratio combined
    im1 = axs[0].imshow(false_negative_combined, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 5.0)
    title_text1 = f"False Negative Rate Across all {anomaly_text0} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Does (Doesn't) Flags the Artificial {anomaly_text0}"
    axs[0].set_title(title_text1, fontsize = 10)
    axs[0].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[0].set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 8)
    fig.colorbar(im1, ax = axs[0], shrink = 0.2, pad = 0.05)
    axs[0].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[0].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[0].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[0].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[0].set_yticklabels(ytick_labels, fontsize = 6)
    if np.max(ratios) > anomaly_ratio:        
        axs[0].axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(false_negative_combined.shape[0]):
        for j in range(false_negative_combined.shape[1]):
            text = axs[0].text(j, i, f'{false_negative_combined[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 4)
    
    # Plot of false positive ratio combined
    im2 = axs[1].imshow(false_positive_combined, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 5.0)
    title_text2 = f"False Positive Rate Across all {anomaly_text0} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Doesn't (Does) Flag Other Anomalies"
    axs[1].set_title(title_text2, fontsize = 10)
    axs[1].set_xlabel('Alpha Threshold', fontsize = 8)
    axs[1].set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 8)
    fig.colorbar(im2, ax = axs[1], shrink = 0.2, pad = 0.05)
    axs[1].invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs[1].set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs[1].set_xticklabels(xtick_labels, fontsize = 6)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs[1].set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs[1].set_yticklabels(ytick_labels, fontsize = 6)
    if np.max(ratios) > anomaly_ratio:        
        axs[1].axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(false_positive_combined.shape[0]):
        for j in range(false_positive_combined.shape[1]):
            text = axs[1].text(j, i, f'{false_positive_combined[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 4)
    
    # Saving the plots
    plt.tight_layout()
    file_name = f"SUMMARY_Heat_Map_{anomaly_text0}_{anomaly_ratio}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

    # Plot of combined information
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    im3 = axs.imshow(combined_combined, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = 5.0)
    title_text2 = f"Combined Results Across all {anomaly_text0} \n With {anomaly_text0}/Noise Ratio of {anomaly_ratio} \n Green (Red) Indicates Good (Bad)"
    axs.set_title(title_text2, fontsize = 14)
    axs.set_xlabel('Alpha Threshold', fontsize = 12)
    axs.set_ylabel('Anomaly Threshold / Noise Ratio', fontsize = 12)
    fig.colorbar(im3, ax = axs, shrink = 0.2, pad = 0.05)
    axs.invert_yaxis()
    xticks = np.linspace(alphas[0], alphas[alphas.size - 1], 11)
    xtick_labels = [f'{tick:.2f}' for tick in xticks]
    axs.set_xticks(np.linspace(0, alphas.size - 1, 11))
    axs.set_xticklabels(xtick_labels, fontsize = 10)
    yticks = np.linspace(ratios[0], ratios[ratios.size - 1], 11)
    ytick_labels = [f'{tick:.2f}' for tick in yticks]
    axs.set_yticks(np.linspace(0, ratios.size - 1, 11))
    axs.set_yticklabels(ytick_labels, fontsize = 10)
    if np.max(ratios) > anomaly_ratio:        
        axs.axhline(y = anomaly_ratio - ratios[0], color = 'blue', linestyle = '--', linewidth = 3)
    for i in range(combined_combined.shape[0]):
        for j in range(combined_combined.shape[1]):
            text = axs.text(j, i, f'{combined_combined[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 8)
    file_name = f"SUMMARY_Combined_Heat_Map_{anomaly_text0}_{anomaly_ratio}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

# -------------------------------
# -------------------------------
# PART 2
# -------------------------------
# -------------------------------

# -------------------------------
# parameter_tests
# Runs anomaly analysis with sepcified parameters across different sizes to test accuracy
# anomaly_lengths   : array
#                     The length of the analysis (0 for jumps, n for glitches of size n)
# ratio             : integer
#                     Ratio of anomaly threshold / noise
# alpha             : integer
#                     The alpha value to be tested
# anomaly_ratios    : array
#                     Ratios of anomaly / noise
# detector_start    : integer
#                     The detector to start at
# detector_end      : integer
#                     The detector to end at
# time_start        : integer
#                     The start index of the data
# time_end          : integer
#                     The length of data to look at
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def parameter_tests(anomaly_lengths, ratio, alpha, anomaly_ratios, detector_start, detector_end, time_start, time_end):
    # Getting the detectors that we care about
    detectors = generate_detectors(detector_start, detector_end, time_start, time_end, 5, 0.75)

    # Setting up where to store false positives and false negatives
    false_positives = np.zeros((anomaly_lengths.size, anomaly_ratios.size))
    false_negatives = np.zeros((anomaly_lengths.size, anomaly_ratios.size))

    # Setting up the time axis
    time_axis = np.linspace(0, time_end - time_start, time_end - time_start, endpoint=False)

    # Iterating across
    for h_size in range(anomaly_lengths.size):
        for v_size in range(anomaly_ratios.size):
            for detector in range(detectors.size):
                # Output to track progress
                print("Anomaly of Size", anomaly_lengths[h_size], "Anomaly Ratio", anomaly_ratios[v_size], "Detector", detectors[detector])
                
                # Setting up random anomaly location
                anomaly_location = random.randint(time_start + 100, time_end - 100)

                # Creating data based on jump or glitch
                if anomaly_lengths[h_size] == 0:
                    data, transform, noise = create_jump_data(detectors[detector], time_start, time_end, anomaly_ratios[v_size], anomaly_location)
                else:
                    data, transform, noise = create_glitch_data(detectors[detector], time_start, time_end, anomaly_ratios[v_size], anomaly_location, anomaly_lengths[h_size])

                # Running the test
                false_positive, false_negative, _ = run_test(anomaly_lengths[h_size], ratio * sqrt_frequency * noise, alpha, transform, anomaly_location - 1)

                # Savint he false positive value
                false_positives[h_size, v_size] = false_positives[h_size, v_size] + false_positive 
                false_negatives[h_size, v_size] = false_negatives[h_size, v_size] + false_negative
    
    # Converting false positive and false negative counts to a ratio
    false_positives = np.round(false_positives / detectors.size, 2)
    false_negatives = np.round(false_negatives / detectors.size, 2)

    print("Positive", false_positives)
    print("Negative", false_negatives)

    # Creating graphs of the data based on anomaly sizes
    fig, axes = plt.subplots(anomaly_lengths.size, 2, figsize=(12, 15))
    fig.suptitle(f"False Positive and Negative Rates \n For Varing Anomaly Sizes and Lengths", fontsize = 20)
    for i in range(anomaly_lengths.size):
        if anomaly_lengths[i] == 0:
            subtitle_text = "Jumps"
        else:
            subtitle_text = f"Glitches of Length {anomaly_lengths[i]}"
        # False negatives in left column
        ax_negative = axes[i, 0]
        bars_negative = ax_negative.bar(anomaly_ratios, false_negatives[i], width = 0.5, color= "red", alpha = 0.7)
        ax_negative.set_title(f"False Negatives for {subtitle_text}", fontsize = 14)
        ax_negative.set_xlabel("Anomaly Size", fontsize = 10)
        ax_negative.set_ylabel("False Negatives", fontsize = 10)
        ax_negative.set_ylim(0, 1.2)
        ax_negative.set_yticks(np.arange(0, 1.4, 6))
        ax_negative.set_xticks(anomaly_ratios) 
        ax_negative.set_xticks(np.arange(min(anomaly_ratios), max(anomaly_ratios) + 1, 1))
        ax_negative.set_xticklabels([f"{x:.1f}" for x in np.arange(min(anomaly_ratios), max(anomaly_ratios) + 1, 1)], rotation=45)
        for bar in bars_negative:
            height = bar.get_height()
            ax_negative.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha = "center", va = "bottom", fontsize=9)

        
        # False positives in right column
        ax_positive = axes[i, 1]
        bars_positive = ax_positive.bar(anomaly_ratios, false_positives[i], width = 0.5, color= "red", alpha = 0.7)
        ax_positive.set_title(f"False Positives for {subtitle_text}", fontsize = 14)
        ax_positive.set_xlabel("Anomaly Size", fontsize = 10)
        ax_positive.set_ylabel("False Negatives", fontsize = 10)
        ax_positive.set_ylim(0, 1.2)
        ax_positive.set_yticks(np.arange(0, 1.4, 6))
        ax_positive.set_xticks(anomaly_ratios) 
        ax_positive.set_xticks(np.arange(min(anomaly_ratios), max(anomaly_ratios) + 1, 1))
        ax_positive.set_xticklabels([f"{x:.1f}" for x in np.arange(min(anomaly_ratios), max(anomaly_ratios) + 1, 1)], rotation=45)
        for bar in bars_positive:
            height = bar.get_height()
            ax_positive.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha = "center", va = "bottom", fontsize=9)

    # Add separate titles for left and right columns
    fig.text(0.50, 0.95, f"Anomaly Threshold of {ratio} times Noise and Alpha Threshold of {alpha}", ha = "center" , va = "center", fontsize = 16)
    fig.text(0.25, 0.85, "False Negatives", ha = "center", va = "center", fontsize = 14)
    fig.text(0.75, 0.85, "False Positives", ha = "center", va = "center", fontsize = 14)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Saving the plot
    file_name = f"Parameter_Tests.png"
    path = os.path.join(parameter_tests_path, file_name)
    fig.savefig(path)

    # Closing it all out
    plt.close()

    return 0