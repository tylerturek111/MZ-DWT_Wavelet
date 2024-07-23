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
import math

# ()()()()()()()()()()()()()
# This file utilizes the functions in Singularity_Analysis to look at "jumps" with real data
# ()()()()()()()()()()()()()

# -------------------------------
# Defining the parameters
# -------------------------------

# Parameters for analyzing the jump
flag_noise_ratio = 50
alpha_threshold = 0.75

# Parameters for the detectors to look at
detector_number = 0
seperation = 1
sqrt_synchronous = 4

# Paraemeters for the time data to look at
time_start = 0
time_end = 100000

# Parameters for number of scales for the wavelet transform
number_scales = 3

# Other parameters
old_alpha_thresold = 0.1
smoothing_index = 100
window_size = 100

# -------------------------------
# Setting up stuff
# -------------------------------

# Setting up the graphs
fig1, axs1 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig2, axs2 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig3, axs3 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))
fig4, axs4 = plt.subplots(sqrt_synchronous, sqrt_synchronous, figsize=(15, 15))

# Setting up the time axis
time_axis = np.linspace(0, total_number, total_number, endpoint=False)
time_axis2 = np.linspace(0, total_number, total_number - smoothing_index + 1, endpoint=False)
summary_stats = []

# Setting up the path to store the figures
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

# Setting up where to store jump plots
jump_plots = 'Jump_Plots'
jump_plots_path = os.path.join(SO_location, jump_plots)
if not os.path.exists(jump_plots_path):
    os.makedirs(jump_plots_path)

# Setting up where to store summary plots
summary_plots = 'Summary_Plots'
summary_plots_path = os.path.join(SO_location, summary_plots)
if not os.path.exists(summary_plots_path):
    os.makedirs(summary_plots_path)

# -------------------------------
# Looking at multiple detectprs at once
# -------------------------------

for i in range(sqrt_synchronous):
    for j in range(sqrt_synchronous):
        # The current detector
        current_detector = detector_number + (i * sqrt_synchronous + j) * seperation

        # Setting up the output print
        title_text = f"For Detector Number {current_detector}"
        print("-------------------------------------------------------------")
        Singularity_Analysis.print_colored(title_text, "blue")

        # Getting the real data
        original_data, noise_level = Generating_Real_Data.get_real_data(time_start, time_end, current_detector)

        print(f"Noise Level for {i}, {j}: {noise_level}")
        # Getting the power of noise and SNR for the signal itself
        smooth_data = np.convolve(original_data, np.ones(smoothing_index)/smoothing_index, mode='valid')
        deviation_data = original_data[0 : total_number - smoothing_index + 1] - smooth_data
        noise_power = np.mean(deviation_data**2)
        signal_power = np.mean(original_data**2)
        signal_noise_ratio = signal_power / noise_power
        signal_noise_ratio_plus = 10 * math.log(signal_power / noise_power)

        print(f"Signal Power {signal_power}")
        print(f"Noise Power {noise_power}")

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
        standard_deviation2 = np.std(original_data)
        jump_threshold = flag_noise_ratio * noise_level

        print(f"Standard Deviation for {i}, {j}: {standard_deviation}")
        print(f"Standard Deviation 2 for {i}, {j}: {standard_deviation2}")
        print(f"Ratio for {i}, {j}: {standard_deviation / noise_level}")
        print(f"SNR {signal_noise_ratio}")
        print(f"SNR Plus {signal_noise_ratio_plus}")

        # Plotting the original signal
        axs1[i, j].plot(time_axis, original_data)
        # axs1[i, j].plot(time_axis2, smooth_data)
        axs1[i, j].set_title(f"Original Signal of Detector {current_detector}", fontsize = 8)
        axs1[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs1[i, j].set_ylabel("Amplitude", fontsize = 6)
        axs1[i, j].tick_params(axis = "both", labelsize = 4)
        axs1[i, j].grid(True)

        # Plotting the wavelet transform
        axs2[i, j].plot(time_axis, wavelet_transform)
        # axs2[i, j].plot(time_axis2, smooth_wavelet)
        axs2[i, j].set_title(f"Wavlet Transform of Detector {current_detector}", fontsize = 8)
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
        axs3[i, j].set_title(f"Alpha from Old Method of Detector {current_detector}", fontsize = 8)
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
        axs4[i, j].set_title(f"Alpha from New Method of Detector {current_detector}", fontsize = 8)
        axs4[i, j].set_xlabel("Time (s)", fontsize = 6)
        axs4[i, j].set_ylabel("Alpha", fontsize = 6)
        axs4[i, j].tick_params(axis = "both", labelsize = 4)
        axs4[i, j].axhline(y = alpha_threshold, color = "r", linestyle = "--")
        axs4[i, j].axhline(y = -1 * alpha_threshold, color = "r", linestyle = "--")
        axs4[i, j].axhline(y = 0, color = "g")
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

        # Adding the location of humps to signal graph
        for k in range(combined.size):
            axs1[i, j].axvline(x = combined[k], color = "r", linestyle = "--")

        #
        # Saving the possible jumps all together and generating plots of their locations
        # 
        if combined.shape[0] != 0:
            summary_stats.append([current_detector, combined])
            for j in range(combined.size):
                # Start and end of the graph
                graph_start_value = max(0, combined[j] - window_size)
                graph_end_value = min(combined[j] + window_size, start_index + total_number)

                # Setting up the x-axis
                time_axis3 = np.linspace(graph_start_value, graph_end_value, graph_end_value - graph_start_value, endpoint=False)
                
                # Actually creating the plots
                fig5, ax5 = plt.subplots(figsize=(10, 10))

                ax5.plot(time_axis3, original_data[graph_start_value : graph_end_value])
                ax5.set_title(f"Detector {current_detector} Around Point {combined[j]}")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Amplitude")
                ax5.axvline(x = combined[j], color = "r", linestyle = "--")
                ax5.axvline(x = combined[j] + 1, color = "r", linestyle = "--")

                # Saving the plot
                file_name = f"{current_detector}_{combined[j]}"
                path = os.path.join(jump_plots_path, file_name)
                fig5.savefig(path)
        
        plt.close()

        # Ending the printing output for each detector
        print("-------------------------------------------------------------")
        print("")

# Printing summary location of all of the jumps
summary_stats = np.array(summary_stats, dtype = object)
Singularity_Analysis.print_colored("(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(", "red")
Singularity_Analysis.print_colored("SUMMARY OF ALL OF THE LOCATED JUMPS", "cyan")
for i in range(summary_stats.shape[0]):
    current_text = f"Detector {summary_stats[i, 0]}: {summary_stats[i, 1]}"
    Singularity_Analysis.print_colored(current_text, "green")
Singularity_Analysis.print_colored("(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(-)(", "red")

# Saving the summary plots
path1 = os.path.join(summary_plots_path, "Signals")
path2 = os.path.join(summary_plots_path, "Transforms")
path3 = os.path.join(summary_plots_path, "Old_Alpha")
path4 = os.path.join(summary_plots_path, "New_Alpha")
fig1.savefig(path1)
fig2.savefig(path2)
fig3.savefig(path3)
fig4.savefig(path4)

plt.close('all')
