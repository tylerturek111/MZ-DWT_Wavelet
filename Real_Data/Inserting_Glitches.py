import Testing_Parameters
import numpy as np

# ()()()()()()()()()()()()()
# This file tests how the function works by adding in jumps
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# --------------------------------------------------------------
# Parameters for Part 1: analyzing anomaly and alpha thresholds

# Anomaly Threshold (anomaly threshold = noise * ratio) values to be tested
ratio_start = 2
ratio_end = 10
ratio_step = 1

# Alpha Threshold values to be tested
alpha_start = 0
alpha_end = 1.6
alpha_step = 0.2

# --------------------------------------------------------------
# Parameters for Part 2: analyzing performance at a particular alpha and jumps

target_ratio = 6

target_alpha = 1.5

# --------------------------------------------------------------
# Paramters used for both Parts 1 and 2

# Glitch Height Sizes (jump size = noise * jump_ratio) to be tested 
anomaly_size_start = 5
anomaly_size_end = 10
anomaly_size_step = 1

# Glitch sizes to be tested
anomaly_length_start = 0
anomaly_length_end = 5
anomaly_length_step = 1

# The detectors to possibly look at
detector_start = 0
detector_end = 10

# Data poitns to look at
time_start = 0
time_end = 100000

# -------------------------------
# Part 1: Running simulations with different parameters
# -------------------------------

# Setting up the parameters that can be changed
ratios = np.arange(ratio_start, ratio_end + ratio_step, ratio_step)
alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
anomaly_sizes = np.arange(anomaly_size_start, anomaly_size_end + anomaly_size_step, anomaly_size_step)
anomaly_lengths = np.arange(anomaly_length_start, anomaly_length_end + anomaly_length_step, anomaly_length_step)
detectors = Testing_Parameters.generate_detectors(detector_start, detector_end, time_start, time_end, 5, 0.75)

print(ratios)

# Running the simulations
_ = Testing_Parameters.determine_accuracy(anomaly_lengths, ratios, alphas, anomaly_sizes, detectors, time_start, time_end)

print("Completed 1")

# -------------------------------
# Part 2: Running tests with defined parameters
# -------------------------------

anomaly_lengths = np.arange(anomaly_length_start, anomaly_length_end + anomaly_length_step, anomaly_length_step)
anomaly_sizes = np.arange(anomaly_size_start, anomaly_size_end + anomaly_size_step, anomaly_size_step)
detectors = Testing_Parameters.generate_detectors(detector_start, detector_end, time_start, time_end, 5, 0.75)

_ = Testing_Parameters.parameter_tests(anomaly_lengths, target_ratio, target_alpha, anomaly_sizes, detectors, time_start, time_end)

print("Completed 2")