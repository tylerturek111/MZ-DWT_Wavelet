import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil

# ()()()()()()()()()()()()()
# This file creates the graphs for Testing_Parameters
# ()()()()()()()()()()()()()


# -------------------------------
# Setting up the storage location
# -------------------------------

# STORAGE LOCATION 1
# Location for the plots to be stored for heat maps (Part 1, storage location 1)
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
location_1_name = 'Heat_Maps2'
location_1_path = os.path.join(SO_location, location_1_name)
if os.path.exists(location_1_path):
    shutil.rmtree(location_1_path)
os.makedirs(location_1_path)

# STORAGE LOCATION 2
# Location for the plots to be stored for parameter testing (Part 2, stp)
SO_location = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
location_2_name = 'Parameter_Tests2'
location_2_path = os.path.join(SO_location, location_2_name)
if os.path.exists(location_2_path):
    shutil.rmtree(location_2_path)
os.makedirs(location_2_path)

# -------------------------------
# create_heatmap
# Creating a heatmap for data
# data                : array
#                       The data for the heatmap
# max_value           : integer
#                       The max value for the heat map
# graph_title         : string
#                       The title of the graph
# x_axis_title        : string
#                       The name of the x-axis
# y_axis_title        : string
#                       The name of the y-axis
# x_axis              : array
#                       The x-axis scale
# y_axis              : array
#                       The y-axis scale
# horizontal_line     : integer
#                       The y value for a horizontal line
# storage_location    : integer
#                       What storage location to save the figure
# file_name           : string
#                       The name for the file
# RETURNS,            : integer
#                       Means nothing
# -------------------------------

def create_heatmap(data, max_value, graph_title, x_axis_title, y_axis_title, x_axis, y_axis, horizontal_line, storage_location, file_name):
    # Creating the plot
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # Colors for the plot
    colors = [(0, 1, 0), (1, 0, 0)] 
    cmap_name = 'green_red'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N = 256)

    # Creating the plot
    im1 = axs.imshow(data, cmap = custom_cmap, interpolation = 'nearest', vmin = 0.0, vmax = max_value)

    # Adding a title and axis labels
    axs.set_title(graph_title, fontsize = 20)
    axs.set_xlabel(x_axis_title, fontsize = 16)
    axs.set_ylabel(y_axis_title, fontsize = 16)

    # Modifying the color bar
    fig.colorbar(im1, ax = axs, shrink = 0.2, pad = 0.05)

    # Altering the x-axis
    # xticks = np.linspace(x_axis[0], x_axis[x_axis.size - 1], 11)
    # xtick_labels = [f"{tick:.2f}" for tick in xticks]
    # axs.set_xticks(np.linspace(0, x_axis.size - 1, 11))
    # axs.set_xticklabels(xtick_labels, fontsize = 12)

    # Altering the y-axis
    axs.invert_yaxis()
    # yticks = np.linspace(y_axis, y_axis[y_axis.size - 1], 11)
    # ytick_labels = [f"{tick:.2f}" for tick in yticks]
    # axs.set_yticks(np.linspace(0, y_axis.size - 1, 11))
    # axs.set_yticklabels(ytick_labels, fontsize = 12)

    # Adding in a horizontal line
    if np.max(y_axis) > horizontal_line and np.min(y_axis) < horizontal_line:
        axs.axhline(y = horizontal_line - y_axis[0], color = "blue", linestyle = "--", linewidth = 3)
    
    # Adding in values for each of the data points
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = axs.text(j, i, f"{data[i, j]:.2f}", ha = "center", va = "center", color = "black", fontsize = 8)

    # Saving the graph
    if storage_location == 0:
        path = os.path.join(location_1_path, file_name)
    else:
        path = os.path.join(location_2_path, file_name)
    fig.savefig(path)

    # Closing the figure
    plt.close()

# -------------------------------
# create_bar_graphs
# Creating a set of muliple bar graph for data
# data                : array
#                       The data for the bar graph
# graph_title         : string
#                       The title of the graph
# subtitle1           : string
#                       The first subtitle
# subtitle2           : string
#                       The second subtitle
# sub_graph_title     : string
#                       The start of the title for each sub graph
# x_axis_title        : string
#                       The name of the x-axis
# y_axis_title        : string
#                       The name of the y-axis
# seperation          : array
#                       The variable that seperates the multiple sub-graphs
# x_axis              : array
#                       The x-axis scale
# storage_location    : integer
#                       What storage location to save the figure
# file_name           : string
#                       The name for the file
# RETURNS,            : integer
#                       Means nothing
# -------------------------------

def create_bar_graph(data, graph_title, subtitle1, subtitle2, sub_graph_title, x_axis_title, y_axis_title, seperation, x_axis, storage_location, file_name):

    # Creating the plot
    fig, axes = plt.subplots(seperation.size, 1, figsize=(12, 15))

    # Adding a title
    fig.suptitle(f"False Negative Rates \n For Varing Anomaly Sizes and Lengths", fontsize = 30, fontweight = "bold")
    
    # Iterating through for each of the subgraphs
    for i in range(seperation.size):
        # Specifying some graph text
        if seperation[i] == 0:
            subtitle_text = "Jumps"
        else:
            subtitle_text = f"Glitches of Length {seperation[i]}"

        # Creating the plot
        ax_negative = axes[i]
        bars_negative = ax_negative.bar(x_axis, data[i], width = 0.5, color = "red", alpha = 0.7)
       
       # Adding a subtitle and axis labels
        ax_negative.set_title(f"{sub_graph_title} {subtitle_text}", fontsize = 14)
        ax_negative.set_xlabel(x_axis_title, fontsize = 10)
        ax_negative.set_ylabel(y_axis_title, fontsize = 10)

        # Altering the x-axis
        ax_negative.set_xticks(x_axis) 
        ax_negative.set_xticks(np.arange(min(x_axis), max(x_axis) + 1, 1))
        ax_negative.set_xticklabels([f"{x:.1f}" for x in np.arange(min(x_axis), max(x_axis) + 1, 1)], rotation=45)

        # Altering the y-axis
        ax_negative.set_ylim(0, 1.2)
        ax_negative.set_yticks(np.arange(0, 1.4, 6))
    
        # Adding in values for each of the data points
        for bar in bars_negative:
            height = bar.get_height()
            ax_negative.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha = "center", va = "bottom", fontsize=9)

    # Adding overall subtitles
    fig.text(0.50, 0.89, subtitle1, ha = "center" , va = "center", fontsize = 25)
    fig.text(0.50, 0.86, subtitle2, ha = "center" , va = "center", fontsize = 22)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Saving the graph
    if storage_location == 0:
        path = os.path.join(location_1_path, file_name)
    else:
        path = os.path.join(location_2_path, file_name)
    fig.savefig(path)

    # Closing the figure
    plt.close()