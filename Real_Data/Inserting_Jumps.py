import Testing_Parameters
import numpy as np

# ()()()()()()()()()()()()()
# This file tests how the function works by adding in jumps
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Anomaly Threshold (anomaly threshold = noise * ratio) values to be tested
ratio_start = 4
ratio_end = 5
ratio_step = 0.1

# Alpha Threshold values to be tested
alpha_start = 0
alpha_end = 1.00
alpha_step = 0.1

# Jump Sizes (jump size = noise * jump_ratio) to be tested 
jump_ratio_start = 8
jump_ratio_end = 10
jump_ratio_step = 1

# The detectors to possibly look at
detector_start = 0
detector_end = 10

# Data poitns to look at
time_start = 0
time_end = 100000

# -------------------------------
# Running simulations with different parameters
# -------------------------------

# Setting up the parameters that can be changed
ratios = np.arange(ratio_start, ratio_end + ratio_step, ratio_step)
alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
jump_ratios = np.arange(jump_ratio_start, jump_ratio_end + jump_ratio_step, jump_ratio_step)

# Actually running the code
for jump_ratio in jump_ratios:
        # Running the simulations
        _ = Testing_Parameters.determine_accuracy(0, ratios, alphas, jump_ratio, detector_start, detector_end, time_start, time_end)

print("Completed")