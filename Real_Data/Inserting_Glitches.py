import Testing_Parameters
import numpy as np

# ()()()()()()()()()()()()()
# This file tests how the function works by adding in jumps
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# --------------------------------------------------------------
# Parameters for analyzing anomaly and alpha thresholds

# Anomaly Threshold (anomaly threshold = noise * ratio) values to be tested
ratio_start = 8
ratio_end = 13
ratio_step = 0.5

# Alpha Threshold values to be tested
alpha_start = 0
alpha_end = 2.0
alpha_step = 0.2

# --------------------------------------------------------------
# Parameters for analyzing performance at a particular alpha and jumps

target_ratio = 8

target_alpha = 1.5

# --------------------------------------------------------------
# Paramters used across both cases

# Glitch Height Sizes (jump size = noise * jump_ratio) to be tested 
glitch_ratio_start = 1
glitch_ratio_end = 15
glitch_ratio_step = 1

# Glitch sizes to be tested
glitch_size_start = 0
glitch_size_end = 5
glitch_size_step = 1

# The detectors to possibly look at
detector_start = 0
detector_end = 25

# Data poitns to look at
time_start = 0
time_end = 100000

# -------------------------------
# Part 1: Running simulations with different parameters
# -------------------------------

# Setting up the parameters that can be changed
ratios = np.arange(ratio_start, ratio_end + ratio_step, ratio_step)
alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
glitch_ratios = np.arange(glitch_ratio_start, glitch_ratio_end + glitch_ratio_step, glitch_ratio_step)
glitch_sizes = np.arange(glitch_size_start, glitch_size_end + glitch_size_step, glitch_size_step)

# Actually running the code
for glitch_ratio in glitch_ratios:
    for glitch_size in glitch_sizes:
        # Running the simulations
        _ = Testing_Parameters.determine_accuracy(glitch_size, ratios, alphas, glitch_ratio, detector_start, detector_end, time_start, time_end)
    _ = Testing_Parameters.generate_summary_plots(glitch_size, glitch_ratio, ratios, alphas)

print("Completed 1")

# -------------------------------
# Part 2: Running tests with defined parameters
# -------------------------------

anomaly_types = np.arange(glitch_size_start, glitch_size_end + glitch_size_step, glitch_size_step)
glitch_ratios = np.arange(glitch_ratio_start, glitch_ratio_end + glitch_ratio_step, glitch_ratio_step)

_ = Testing_Parameters.parameter_tests(anomaly_types, target_ratio, target_alpha, glitch_ratios, detector_start, detector_end, time_start, time_end)

print("Completed 2")