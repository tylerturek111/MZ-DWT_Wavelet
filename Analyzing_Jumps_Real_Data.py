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
sqrt_synchronous = 5
seperation = 10

# Paraemeters for the data to look at
start_index = 0
total_number = 10000
detector_number = 100

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
time_axis = np.linspace(0, 1, total_number, endpoint=False)

for i in range(sqrt_synchronous):
    for j in range(sqrt_synchronous):
        # Setting up the output print
        title_text = f"For Detector Number {(detector_number + i * sqrt_synchronous + j) * seperation}"
        print("-------------------------------------------------------------")
        Singularity_Analysis.print_colored(title_text, "blue")

        # Getting the real data
        original_data = Generating_Real_Data.get_real_data(start_index, total_number, (detector_number + i * sqrt_synchronous + j) * seperation)

        # Running the wavelet transform
        wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
        processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)

        # Plotting the original signal
        axs1[i, j].plot(time_axis, original_data)
        axs1[i, j].set_title(f"Original Signal of Detector {(detector_number + i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs1[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs1[i, j].set_ylabel("Amplitude", fontsize = 6)
        axs1[i, j].tick_params(axis = "both", labelsize = 4)
        axs1[i, j].grid(True)

        # Plotting the wavelet transform
        axs2[i, j].plot(time_axis, wavelet_transform)
        axs2[i, j].set_title(f"Wavlet Transform of Detector {(detector_number + i * sqrt_synchronous + j) * seperation}", fontsize = 8)
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
        old_alpha_jump_indexes = Singularity_Analysis.compute_alpha_indexes(old_alpha_values, alpha_threshold)

        # Plotting the alpha values
        axs3[i, j].plot(time_axis, old_alpha_values)
        axs3[i, j].set_title(f"Alpha from Old Method of Detector {(detector_number + i * sqrt_synchronous + j) * seperation}", fontsize = 8)
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
        axs4[i, j].plot(time_axis, alpha_values)
        axs4[i, j].set_title(f"Alpha from New Method of Detector {(detector_number + i * sqrt_synchronous + j) * seperation}", fontsize = 8)
        axs4[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs4[i, j].set_ylabel("Alpha", fontsize = 6)
        axs4[i, j].tick_params(axis = "both", labelsize = 4)
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










# # -------------------------------
# # Getting the data
# # -------------------------------

# # Getting the real data
# original_data = Generating_Real_Data.get_real_data(start_index, total_number, detector_number)

# # Creating the time axis
# time_axis = np.linspace(0, 1, total_number, endpoint=False)

# # -------------------------------
# # Running the wavelet transform
# # -------------------------------

# wavelet_transform, time_series = MZ_Wavelet_Transforms.forward_wavelet_transform(number_scales, original_data)
# processed_data = MZ_Wavelet_Transforms.inverse_wavelet_transform(wavelet_transform, time_series)

# # -------------------------------
# # Generating Plots
# # -------------------------------

# # Plotting the original signal
# plt.figure(1)
# plt.plot(time_axis, original_data)
# plt.xlabel("Time (s)", fontsize = 6)
# plt.ylabel("Amplitude", fontsize = 6)
# plt.title(f"Original Signal of Detector {detector_number}", fontsize = 8)
# plt.grid(True)
# plt.savefig("Original_Signal.png")

# # Plotting the wavelet transform
# plt.figure(2)
# plt.plot(time_axis, wavelet_transform)
# plt.xlabel("Time (s)", fontsize = 6)
# plt.ylabel("Value", fontsize = 6)
# plt.title(f"Wavlet Transform of Detector {detector_number}", fontsize = 8)
# plt.grid(True)
# plt.savefig("Wavelet_Transform.png")

# # -------------------------------
# # Looking for jumps based on behavior
# # -------------------------------

# # Using the compute_jump_locations function to find jumps
# jump_indexes = Singularity_Analysis.compute_jump_locations(wavelet_transform, jump_threshold)

# # Printing the results
# print("-------------------------------------------------------------")
# Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Behavior", "magenta")
# print(jump_indexes)
# print("Number of jumps", jump_indexes.size)
# print("-------------------------------------------------------------")
# print("")

# # -------------------------------
# # Looking for jumps based on the old alpha values
# # -------------------------------

# # Using the compute_alpha_values function to calcualte alpha for entire data set
# old_alpha_values = Singularity_Analysis.compute_alpha_values(wavelet_transform)

# # Getting the indexes
# old_alpha_jump_indexes = Singularity_Analysis.compute_alpha_indexes(old_alpha_values, alpha_threshold)

# # Plotting the alpha values
# plt.figure(3)
# plt.axhline(y = 0, color = "r")
# plt.plot(time_axis, old_alpha_values)
# plt.xlabel("Time (s)", fontsize = 6)
# plt.ylabel("Alpha", fontsize = 6)
# plt.title(f"Alpha Values as Calculated via the Old Method of Detector {detector_number}", fontsize = 8)
# plt.grid(True)
# plt.savefig("Old_Alpha.png")

# print("-------------------------------------------------------------")
# Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on Old Alpha", "magenta")
# print(old_alpha_jump_indexes)
# print("Number of jumps", old_alpha_jump_indexes.size)
# print("-------------------------------------------------------------")
# print("")

# # -------------------------------
# # Looking for jumps based on the new alpha values
# # -------------------------------

# # Using the function to calcualte alpha for entire data set
# alpha_values, alpha_jump_indexes = Singularity_Analysis.packaged_compute_alpha_values_and_indexes(wavelet_transform, 1, jump_threshold, alpha_threshold)

# # Plotting the new alpha values
# plt.figure(4)
# plt.axhline(y = 0, color = "r")
# plt.plot(time_axis, alpha_values)
# plt.xlabel("Time (s)", fontsize = 6)
# plt.ylabel("Alpha", fontsize = 6)
# plt.title(f"Alpha Values as Calculated via the New Method of Detector {detector_number}", fontsize = 8)
# plt.grid(True)
# plt.savefig("New_Alpha.png")


# print("-------------------------------------------------------------")
# Singularity_Analysis.print_colored("Indexes Where Jumps are Suspected based on New Alpha", "magenta")
# print(alpha_jump_indexes)
# print("Number of jumps", alpha_jump_indexes.size)
# print("-------------------------------------------------------------")
# print("")

# # -------------------------------
# # Looking at ways to better compute alphas that matter
# # -------------------------------

# #
# # Method 0: Possible jumps based on both behavior and alpha
# #

# behavior_jumps_indexes = np.array(jump_indexes)
# combined = np.intersect1d(alpha_jump_indexes, jump_indexes)

# print("-------------------------------------------------------------")
# print("Indexes Where Jumps are Suspected based on Beahvior AND Alpha")
# Singularity_Analysis.print_colored(combined, "cyan")
# print("-------------------------------------------------------------")
# print("")
