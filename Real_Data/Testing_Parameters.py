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

# -------------------------------
# Setting up the storage location
# -------------------------------

# Location for the plots to be stored
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
heat_maps = 'Heat_Maps'
heat_maps_path = os.path.join(SO_location, heat_maps)
if os.path.exists(heat_maps_path):
    shutil.rmtree(heat_maps_path)
os.makedirs(heat_maps_path)

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
# -------------------------------

def run_test(anomaly_type, anomaly, alpha, transform, anomaly_location):
    # Flags other anomalies
    false_positive_count = 0.0

    # Doesn't flag the anomaly
    false_negative_count = 0.0

    # Computing anomaly locations with the developed methods
    anomaly_locations, _ = Singularity_Analysis.alpha_and_behavior_jumps(transform, anomaly, alpha)

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
    return false_positive_count, false_negative_count

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

    for detector in range(detectors.size):
        # Getting a random anomaly location (included a little bit of buffer)
        # anomaly_location = random.randint(time_start + 100, time_end - 100)
        anomaly_location = 50000

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

        # Creating a plot of the data
        plt.plot(time_axis, data)
        plt.axvline(x = anomaly_location - 1, color = "r", linestyle = "--")
        file_name = f"Signal_{detectors[detector]}.png"
        path = os.path.join(heat_maps_path, file_name)
        plt.savefig(path)
        plt.close()

        # Creating a plot of the transform
        plt.plot(time_axis, transform)
        plt.axhline(y = noise * 10 * ratios[0], color = "g", linestyle = "--")
        plt.axhline(y = - 1 * noise * 10 * ratios[0], color = "g", linestyle = "--")
        plt.axhline(y = noise * 10 * ratios[ratios.size - 1], color = "r", linestyle = "--")
        plt.axhline(y = - 1 * noise * 10 * ratios[ratios.size - 1], color = "r", linestyle = "--")
        file_name = f"Transform_{detectors[detector]}.png"
        path = os.path.join(heat_maps_path, file_name)
        plt.savefig(path)
        plt.close()

        # Running this for the different sets of ratio's and alphas for each detector
        for ratio in range(ratios.size):
            for alpha in range(alphas.size):
                false_positive, false_negative = run_test(anomaly_type, ratios[ratio] * sqrt_frequency * noise, alphas[alpha], transform, anomaly_location - 1)
                false_positives[ratio, alpha] = false_positives[ratio, alpha] + false_positive 
                false_negatives[ratio, alpha] = false_negatives[ratio, alpha] + false_negative

    # Getting the counts to become ratios
    false_positives = false_positives / detectors.size
    false_negatives = false_negatives / detectors.size

    combined = false_positives + false_negatives

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
    for i in range(false_positives.shape[0]):
        for j in range(false_positives.shape[1]):
            text = axs[1].text(j, i, f'{false_positives[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 4)
    
    # Saving the plots
    plt.tight_layout()

    # Saving the plots
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
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            text = axs.text(j, i, f'{combined[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 8)

    # Saving the plot
    file_name = f"Combined_Heat_Map_{anomaly_text0}_{anomaly_ratio}_{anomaly_type}.png"
    path = os.path.join(heat_maps_path, file_name)
    fig.savefig(path)

    plt.close()

    return 0