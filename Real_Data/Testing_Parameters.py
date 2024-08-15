import sys
import os

current_dir = os.path.dirname(__file__)
folder1_path = os.path.abspath(os.path.join(current_dir, '..', 'Wavelet_Code'))
sys.path.insert(0, folder1_path)

import MZ_Wavelet_Transforms
import Singularity_Analysis

sys.path.pop(0)

import Generating_Real_Data
import Creating_Graphs

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
# -------------------------------

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
# calculate_false_positive
# Compute whether a false positive occured at a particular parameter
# anomaly_length    : integer
#                     The length of anomaly to be analyzed (0 for jumps, n for glitches of size n)
# anomaly           : integer
#                     Anomaly threshold to be tested
# alpha             : integer
#                     Alpha threshold to be tested
# transform         : array
#                     The transform array
# anomaly_location  : integer
#                     The location of the anomaly
# RETURNS           : integer
#                     1 if false n positive occured, 0 otherwise
# -------------------------------

def calculate_false_positive(anomaly_length, anomaly, alpha, transform, anomaly_location):
    # Flags other anomalies
    false_positive_count = 0

    # Computing anomaly locations with the developed methods
    anomaly_locations, alpha_values = Singularity_Analysis.alpha_and_behavior_jumps(transform, anomaly, alpha)
    number_anomalies = anomaly_locations.size

    # Computing false positive
    
    # In the case of a jump
    if anomaly_length == 0:
        if anomaly_location in anomaly_locations:
            if number_anomalies != 1:
                false_positive_count = 1
        else:
            if number_anomalies != 0:
                false_positive_count = 1
            
    # In the case of a glitch
    else:
        if anomaly_location in anomaly_locations and (anomaly_location + anomaly_length) in anomaly_locations:
            if number_anomalies != 2:
                false_positive_count = 1
        else:
            if anomaly_location in anomaly_locations or (anomaly_location + anomaly_length) in anomaly_locations:
                if number_anomalies != 1:
                    false_positive_count = 1
            else:
                if number_anomalies != 0:
                    false_positive_count = 1
    
    return false_positive_count

# -------------------------------
# calculate_false_negative
# Compute whether a false negative occured at a particular parameter
# anomaly_length    : integer
#                     The length of anomaly to be analyzed (0 for jumps, n for glitches of size n)
# anomaly           : integer
#                     Anomaly threshold to be tested
# alpha             : integer
#                     Alpha threshold to be tested
# transform         : array
#                     The transform array
# anomaly_location  : integer
#                     The location of the anomaly
# RETURNS           : integer
#                     1 if false n negative occured, 0 otherwise
# -------------------------------

def calculate_false_negative(anomaly_length, anomaly, alpha, transform, anomaly_location):
    # Doesn't flag the anomaly
    false_negative_count = 0

    # Computing anomaly locations with the developed methods
    anomaly_locations, alpha_values = Singularity_Analysis.alpha_and_behavior_jumps(transform, anomaly, alpha)
    number_anomalies = anomaly_locations.size

    # Computing false negative

    # In the case of a jump
    if anomaly_length == 0:
        if anomaly_location not in anomaly_locations:
            false_negative_count = 1
        
    # In the case of a glitch
    else:
        if anomaly_location not in anomaly_locations or (anomaly_location + anomaly_length) not in anomaly_locations:
            false_negative_count = 1
    
    return false_negative_count

# -------------------------------
# calculate_falses
# Calculates the false negative results
# detectors         : array
#                     The detector that we care about
# time_start        : integer
#                     The start index of the data
# time_end          : integer
#                     The length of data to look at
# anomaly_size      : array
#                     Ratio of anomaly / noise
# anomaly_length    : array
#                     The length of anomalies to be analyzed (0 for jumps, n for glitches of size n)
# ratios            : array
#                     Ratios of anomaly threshold / noise
# alphas            : array
#                     The alpha values to be tested
# analysis_type     : integer
#                     The type of calculation (0 for positives, 1 for negatives)
# RETURNS           : array
#                     False negative values or false positive
# -------------------------------

def calculate_falses(detectors, time_start, time_end, anomaly_size, anomaly_length, ratios, alphas, analysis_type):
    # Where false positives or negatives will be stored
    false_values = np.zeros((ratios.size, alphas.size))

    for detector in range(detectors.size):
        # Getting a random anomaly location
        anomaly_location = random.randint(time_start + 100, time_end - 100)

        # Creating the data
        if anomaly_length == 0:
            data, transform, noise = create_jump_data(detectors[detector], time_start, time_end, anomaly_size, anomaly_location)
        else:
            data, transform, noise = create_glitch_data(detectors[detector], time_start, time_end, anomaly_size, anomaly_location, anomaly_length)        

        # Running this for the different sets of ratio's and alphas for each detector
        for ratio in range(ratios.size):
            for alpha in range(alphas.size):
                if analysis_type == 0:
                    false_positive = calculate_false_positive(anomaly_length, ratios[ratio] * sqrt_frequency * noise, alphas[alpha], transform, anomaly_location - 1)
                    false_values[ratio, alpha] += false_positive
                else:
                    false_negative = calculate_false_negative(anomaly_length, ratios[ratio] * sqrt_frequency * noise, alphas[alpha], transform, anomaly_location - 1)
                    false_values[ratio, alpha] += false_negative
    
    false_values = np.round(false_values / detectors.size, 2)

    return false_values
        
# -------------------------------
# determine_accuracy
# Runs anomaly analysis with varrying parameters and generate heat maps to determine accuracy
# anomaly_lengths   : array
#                     The lengths of anomalies to be analyzed (0 for jumps, n for glitches of size n)
# ratios            : array
#                     Ratios of anomaly threshold / noise
# alphas            : array
#                     The alpha values to be tested
# anomaly_sizes     : array
#                     Ratios of anomaly / noise
# detectors         : array
#                     The detector that we care about
# time_start        : integer
#                     The start index of the data
# time_end          : integer
#                     The length of data to look at
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def determine_accuracy(anomaly_lengths, ratios, alphas, anomaly_sizes, detectors, time_start, time_end):
    # Calculating the false positive (falsely flags other points) 
    false_positives = calculate_falses(detectors, time_start, time_end, anomaly_sizes[0], anomaly_lengths[0], ratios, alphas, 0)

    # Creating a heat map for the false positives
    _ = Creating_Graphs.create_heatmap(false_positives, 1, f"False Positive Rates \n Green (Red) Indicates Doesn't (Does) Flag Other Anomalies", "Alpha Threshold", "Anomaly Threshold to Noise Ratio", alphas, ratios, 0, 0, f"False_Positives.png")

    # The time axis
    time_axis = np.linspace(0, time_end - time_start, time_end - time_start, endpoint=False)

    for anomaly_size in anomaly_sizes:
        # Creating variable for cross-glitch_length analysis
        false_negatives_combined = np.zeros((ratios.size, alphas.size))

        for anomaly_length in anomaly_lengths:
            # Output to track progress
            print("Part 1, Anomaly Ratio:", anomaly_size, "Anomaly Length:", anomaly_length)

            # Creating variables for naming
            if anomaly_length == 0:
                general_text = "Jump"
                specific_text = "Jump"
            else:
                general_text = "Glitch"
                specific_text = f"Glitch of Size {anomaly_length}"
            
            # Calculating false negatives
            false_negatives = calculate_falses(detectors, time_start, time_end, anomaly_size, anomaly_length, ratios, alphas, 1)

            # Saving false negatives for cross-glitch_length analysis
            false_negatives_combined += false_negatives

            # Creating heat maps for the false negatives
            _ = Creating_Graphs.create_heatmap(false_negatives, 1, f"False Negative Rate with Artificially Added {specific_text} \n With {general_text}/Noise Ratio of {anomaly_size} \n Green (Red) Indicates Does (Doesn't) Flags the Artificial {specific_text}", "Alpha Threshold", "Anomaly Threshold to Noise Ratio", alphas, ratios, anomaly_size, 0, f"False_Negative_{general_text}_{anomaly_length}_{anomaly_size}.png")

            # Calculating total false ratios (false_positives + false_negatives)
            total_falses = false_positives + false_negatives

            # Creating heat maps for the total falses
            _ = Creating_Graphs.create_heatmap(total_falses, 1, f"Total Falses with Artificially Added {specific_text} \n With {general_text}/Noise Ratio of {anomaly_size} \n Green (Red) Indicates Good (Bad)", "Alpha Threshold", "Anomaly Threshold to Noise Ratio", alphas, ratios, anomaly_size, 0, f"Total_Falses_{general_text}_{anomaly_length}_{anomaly_size}.png")

        # Creating heat maps for the false negatives for cross-glitch_length analysis
        _ = Creating_Graphs.create_heatmap(false_negatives_combined, anomaly_lengths.size, f"False Negative Rate Across all Anomalies \n With Anomaly to Noise Ratio of {anomaly_size} \n Green (Red) Indicates Does (Doesn't) Flags the Artificial {general_text}", "Alpha Threshold", "Anomaly Threshold to Noise Ratio", alphas, ratios, anomaly_size, 0, f"Cross-Glitch_Length_False_Negatives_{general_text}_{anomaly_size}.png")

        # Calculating total false ratios (false_positives + false_negatives) for cross-glitch_length analysis
        total_falses_combined = false_positives + false_negatives_combined

        # Creating heat maps for the total falses for cross-glitch_length analysis
        _ = Creating_Graphs.create_heatmap(false_negatives_combined, anomaly_lengths.size, f"Total Falses Results Across all Anomalies \n With Anomaly to Noise Ratio of {anomaly_size} \n Green (Red) Indicates Good (Bad)", "Alpha Threshold", "Anomaly Threshold to Noise Ratio", alphas, ratios, anomaly_size, 0, f"Cross-Glitch_Length_Total_Falses_{general_text}_{anomaly_size}.png")
    
    return 0

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
# anomaly_sizes     : array
#                     Ratios of anomaly / noise
# detectors         : integer
#                     The detector we care about
# time_start        : integer
#                     The start index of the data
# time_end          : integer
#                     The length of data to look at
# RETURNS           : integer
#                     Means nothing
# -------------------------------

def parameter_tests(anomaly_lengths, ratio, alpha, anomaly_sizes, detectors, time_start, time_end):    
    # Calculating the false positives
    false_positives = calculate_falses(detectors, time_start, time_end, anomaly_sizes[0], anomaly_lengths[0], np.array([ratio]), np.array([alpha]), 0).item()

    # Setting up variable to store the false_negatives
    false_negatives = np.empty((anomaly_lengths.size, anomaly_sizes.size))

    # Setting up the time axis
    time_axis = np.linspace(0, time_end - time_start, time_end - time_start, endpoint=False)

    # Iterating across the anomaly lengths and ratios
    for anomaly_size in range(anomaly_sizes.size):
        for anomaly_length in range(anomaly_lengths.size):
            # Output to track progress
            print("Part 2, Anomaly Ratio:", anomaly_size, "Anomaly Length:", anomaly_length)
                
            # Calculating the false negative
            false_negative = calculate_falses(detectors, time_start, time_end, anomaly_sizes[anomaly_size], anomaly_lengths[anomaly_length], np.array([ratio]), np.array([alpha]), 1).item()

            # Saving false negative to our false negatives array
            false_negatives[anomaly_length, anomaly_size] = false_negative

    _ = Creating_Graphs.create_bar_graph(false_negatives, f"False Negative Rates \n For Varing Anomaly Sizes and Lengths", f"Anomaly Threshold of {ratio} times Noise and Alpha Threshold of {alpha}", f"False Positive Rate was {false_positives}", "False Negative Rate for", "Anomaly Size", "False Negative Rate", anomaly_lengths, anomaly_sizes, 1, "Parameter_Tests.png")

    return 0