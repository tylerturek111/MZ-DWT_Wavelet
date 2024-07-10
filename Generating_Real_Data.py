import MZ_Wavelet_Transforms
import Singularity_Analysis

from sotodlib.core import AxisManager

# ()()()()()()()()()()()()()
# This file processes the raw real data so that it can be utilized
# ()()()()()()()()()()()()()

# -------------------------------
# Extracting the raw data to improve compute time
# -------------------------------

# Loading in the raw real data
global_raw_data = AxisManager.load("/mnt/welch/SO/obs_1704900313_lati1_111.h5")

# Focusing only on detectors that are set up properly
global_raw_data.restrict("dets", global_raw_data.det_cal.bg > 0)

# -------------------------------
# get_real_data
# Generates real data for further analysis
# start_index     : integer
#                   The point at which jump analysis start
# length          : integer
#                   The number of samples to be analyzed
# detector_number : integer
#                   The detector number to be analyzed
# Returns         : 1-D Numpy Array of floats
#                   The modified actual data to look at
# -------------------------------

def get_real_data(start_index, length, detector_number):
    current_raw_data = global_raw_data

    # Only focusing on the first "length" samples for every detector
    current_raw_data.restrict("samps", slice(start_index, start_index + length))

    # Further processing the data
    current_raw_data.preprocess
    current_raw_data.preprocess.noise
    current_raw_data.preprocess.noise.white_noise

    return current_raw_data.signal[detector_number]

# def get_real_data(start_index, length, detector_number):
#     # Loading in the raw real data
#     raw_data = AxisManager.load("/mnt/welch/SO/obs_1704900313_lati1_111.h5")

#     # Only focusing on the first "length" samples for every detector
#     raw_data.restrict("samps", slice(start_index, start_index + length))

#     # Focusing only on detectors that are set up properly
#     raw_data.restrict("dets", raw_data.det_cal.bg > 0)

#     # Further processing the data
#     raw_data.preprocess
#     raw_data.preprocess.noise
#     raw_data.preprocess.noise.white_noise

#     return raw_data.signal[detector_number]