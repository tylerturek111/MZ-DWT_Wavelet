import sys
import os

current_dir = os.path.dirname(__file__)
folder1_path = os.path.abspath(os.path.join(current_dir, '..', 'Wavelet_Code'))
sys.path.insert(0, folder1_path)

import MZ_Wavelet_Transforms
import Singularity_Analysis

sys.path.pop(0)

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
global_raw_data.restrict("dets", global_raw_data.det_cal.bg >= 0)

# Focusing only on detectors that have accompanying background data
global_raw_data.restrict("dets", global_raw_data.preprocess.noise.white_noise > 0)

# -------------------------------
# get_real_data
# Generates real data for further analysis
# start_index      : integer
#                    The point at which jump analysis start
# length           : integer
#                    The number of samples to be analyzed
# detector_number  : integer
#                    The detector number to be analyzed
# RETURNS, returns : 1-D Numpy Array of floats
#                    The modified actual data to look at
# returns, RETURNS : integer
#                    The noise value of a given detector
# -------------------------------

def get_real_data(start_index, length, detector_number):
    current_raw_data = global_raw_data

    # Only focusing on the first "length" samples for every detector
    current_raw_data.restrict("samps", slice(start_index, start_index + length))

    return current_raw_data.signal[detector_number], current_raw_data.preprocess.noise.white_noise[detector_number]