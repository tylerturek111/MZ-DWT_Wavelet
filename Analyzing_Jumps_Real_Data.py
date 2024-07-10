import MZ_Wavelet_Transforms
import Singularity_Analysis
import Generating_Real_Data

import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ()()()()()()()()()()()()()
# This file utilizes the functions in Singularity_Analysis to look at "jumps" with real data
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
jump_threshold = 0.005
alpha_threshold = 0.75
old_alpha_thresold = 0.1
sqrt_synchronous = 5
seperation = 25
smoothing_index = 100

# Paraemeters for the data to look at
start_index = 0
total_number = 10000
detector_number = 0

# Parameters for number of scales for the wavelet transform
number_scales = 3

# -------------------------------
# Looking at multiple detectprs at once
# -------------------------------

# Setting up the graphs
fig1, axs1 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig2, axs2 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig3, axs3 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig4, axs4 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))

# Time axis
time_axis = np.linspace(0, 10000, total_number, endpoint=False)
time_axis2 = np.linspace(0, 10000, total_number - smoothing_index + 1, endpoint=False)

for i in range(sqrt_synchronous):
    for j in range(sqrt_synchronous):
        # Setting up the output print
        title_text = f"For Detector Number {detector_number + (i * sqrt_synchronous + j) * seperation}"
        print("-------------------------------------------------------------")
        Singularity_Analysis.print_colored(title_text, "blue")

        # Getting the real data
        original_data = Generating_Real_Data.get_real_data(start_index, total_number, detector_number + (i * sqrt_synchronous + j) * seperation)
        
        # Getting the power of noise and SNR for the signal itself
        smooth_data = np.convolve(original_data, np.ones(smoothing_index)/smoothing_index, mode='valid')
        deviation_data = original_data[0 : total_number - smoothing_index + 1] - smooth_data
        noise_power = np.mean(deviation_data**2)
        signal_power = np.mean(original_data**2)
        signal_noise_ratio = signal_power / noise_power

        # Running the wavelet transform
        wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
        processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)
        
        # Getting the power of noise and SNR for the wavelet transform 
        first_wavelet_transform = wavelet_transform[:, 0]
        smooth_wavelet = np.convolve(first_wavelet_transform, np.ones(smoothing_index)/smoothing_index, mode='valid')
        deviation_wavelet = first_wavelet_transform[0 : total_number - smoothing_index + 1] - smooth_wavelet
        noise_power_wavelet = np.mean(deviation_wavelet**2)
        signal_power_wavelet = np.mean(first_wavelet_transform**2)
        signal_noise_ratio_wavelet = signal_power_wavelet / noise_power_wavelet

        # Modifying jump threshold based on standard deviation of the wavelet transform
        standard_deviation = np.std(wavelet_transform)
        jump_threshold = 5 * standard_deviation

        # Plotting the original signal
        axs1[i, j].plot(time_axis, original_data)
        axs1[i, j].plot(time_axis2, smooth_data)
        axs1[i, j].set_title(f"Original Signal of Detector {detector_number + (i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs1[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs1[i, j].set_ylabel("Amplitude", fontsize = 6)
        axs1[i, j].tick_params(axis = "both", labelsize = 4)
        axs1[i, j].grid(True)

        # Plotting the wavelet transform
        axs2[i, j].plot(time_axis, wavelet_transform)
        axs2[i, j].plot(time_axis2, smooth_wavelet)
        axs2[i, j].set_title(f"Wavlet Transform of Detector {detector_number + (i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs2[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs2[i, j].set_ylabel("Value", fontsize = 6)
        axs2[i, j].tick_params(axis = "both", labelsize = 4)
        axs2[i, j].axhline(y = jump_threshold, color = "r", linestyle = "--")
        axs2[i, j].axhline(y =  -1 * jump_threshold, color = "r", linestyle = "--")
        axs2[i, j].grid(True)

        #
        # Looking for jumps based on behavior
        #
        jump_indexes = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

        # Printing the behavior results
        jump_text = f"Indexes Where Jumps are Suspected based on Behavior ({jump_indexes.size} total Jumps): {jump_indexes}"
        Singularity_Analysis.print_colored(jump_text, "yellow")

        # 
        # Looking for jumps based on the old alpha values
        # 
        old_alpha_values = Singularity_Analysis.compute_alpha_values(wavelet_transform)
        old_alpha_jump_indexes = Singularity_Analysis.compute_alpha_indexes(old_alpha_values, old_alpha_thresold)

        # Plotting the alpha values
        axs3[i, j].plot(time_axis, old_alpha_values)
        axs3[i, j].set_title(f"Alpha from Old Method of Detector {detector_number + (i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs3[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs3[i, j].set_ylabel("Alpha", fontsize = 6)
        axs3[i, j].tick_params(axis = "both", labelsize = 4)
        axs3[i, j].grid(True)

        old_alpha_text = f"Indexes Where Jumps are Suspected based on Old Alpha ({old_alpha_jump_indexes.size} total Jumps): {old_alpha_jump_indexes}"
        Singularity_Analysis.print_colored(old_alpha_text, "yellow")

        # 
        # Looking for jumps based on the new alpha values
        # 
        alpha_values, alpha_jump_indexes = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

        # Plotting the new alpha values
        axs4[i, j].plot(time_axis, alpha_values, marker = ".")
        axs4[i, j].set_title(f"Alpha from New Method of Detector {detector_number + (i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs4[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs4[i, j].set_ylabel("Alpha", fontsize = 6)
        axs4[i, j].tick_params(axis = "both", labelsize = 4)
        axs4[i, j].axhline(y = alpha_threshold, color = "r", linestyle = "--")
        axs4[i, j].axhline(y = -1 * alpha_threshold, color = "r", linestyle = "--")
        axs4[i, j].grid(True)

        new_alpha_text = f"Indexes Where Jumps are Suspected based on New Alpha ({alpha_jump_indexes.size} total Jumps): {alpha_jump_indexes}"

        Singularity_Analysis.print_colored(new_alpha_text, "yellow")

        #
        # Possible jumps based on both behavior and alpha
        #
        behavior_jumps_indexes = np.array(jump_indexes)
        combined = np.intersect1d(alpha_jump_indexes, jump_indexes)

        combined_text = f"Indexes Where Jumps are Suspected based on Beahvior AND Alpha ({combined.size} total Jumps): {combined}"
        Singularity_Analysis.print_colored(combined_text, "cyan")

        # Ending the printing output for each detector
        print("-------------------------------------------------------------")
        print("")

fig1.savefig("Signals.png")
fig2.savefig("Transforms.png")
fig3.savefig("Old_Alphas.png")
fig4.savefig("New_Alphas.png")