import pywt
import numpy as np

# (--------------------------------------------)
# Attempt 1: Creating a wavelet based on the MZ-DWT Wavelet's parameter.
# (--------------------------------------------)

# data = [1, 2, 3, 4, 5, 6, 7, 8]

# dec_lo = [0, 0, 0.1250, 0.3750, 0.3750, 0.1250, 0]
# dec_hi = [0, 0, 0, 2, -2, 0, 0]
# rec_lo = [0, 0, 0.1250, 0.3750, 0.3750, 0.1250, 0]
# rec_hi = [0.0078, 0.0547, 0.1719, -0.1719, -0.0547, -0.0078, 0]
# filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
# MZWavelet = pywt.Wavelet(name="MZ-DWTWavelet", filter_bank=filter_bank)


# # result1, result2 = pywt.dwt(data, MZWavelet)
# result = pywt.wavedec(data, MZWavelet, mode='symmetric', level=, axis=-1)

# print(result)

# (--------------------------------------------)
# Attempt 2: Converting the matlab code into python code
# (--------------------------------------------)

#
# Forward Wavelet Transform
#

# Returns the wavelet transform matrix and time series remaining
def forward_wavelet_transform(scales_number, data_vector):
    length = len(data_vector)

    # Normalization coefficients
    lamda = normalization_coefficients(scales_number)

    # Filter coefficients
    H = np.array([0.125, 0.375, 0.375, 0.125])
    G = np.array([2.0, -2.0])

    # Convolution offsets
    Gn = np.array([2])
    Hn = np.array([3])
    for i in range(1, scales_number):
        number_zeros = 2 ** i - 1
        Gn = np.append(Gn, ((number_zeros + 1) / 2) + 1)
        Hn = np.append(Hn, ((number_zeros + 1) / 2) + number_zeros + 2)

    # Compute the wavelet transform at each scale
    time_series = np.concatenate([np.flip(data_vector), data_vector, np.flip(data_vector)])
    wavelet_transform = np.empty((0, length))
    for i in range(0, scales_number):
        number_zeros = 2 ** i - 1
        Gz = insert_zeros(G, number_zeros)
        Hz = insert_zeros(H, number_zeros)

        # Computing wavelet transform at scale J and storing it in wavelet-transform
        current_wavelet_transform = (1 / lamda[i]) * np.convolve(time_series, Gz)
        current_wavelet_transform = current_wavelet_transform[length + int(Gn[i]) - 1 : 2 * length + int(Gn[i]) - 1]
        wavelet_transform = np.vstack([wavelet_transform, np.conjugate(current_wavelet_transform)])
        # wavelet_transform = np.hstack([wavelet_transform, np.transpose(current_wavelet_transform)])

        # Compute next time series
        time_series2 = np.convolve(time_series, Hz)
        time_series2 = time_series2[length + int(Hn[i]) -1 : 2 * length + int(Hn[i]) - 1]
        time_series = np.concatenate([np.flip(time_series2), time_series2, np.flip(time_series2)])

    time_series = time_series[length : 2 * length]
    wavelet_transform = np.transpose(wavelet_transform)

    return wavelet_transform, time_series

#
# Inverse Wavelet Transform
#

# Returns the reconstructed time series
def inverse_wavelet_transform(wavelet_transform, series_remaining):
    length, scales_number = wavelet_transform.shape

    # Normalization coefficients
    lamda = normalization_coefficients(scales_number)

    # Filter coefficients
    H = np.array([0.125, 0.375, 0.375, 0.125])
    K = np.array([0.0078125, 0.054685, 0.171875, -0.171875, -0.054685, -0.0078125])

    # Convolution offsets
    Kn = np.array([3])
    Hn = np.array([2])

    for i in range(1, scales_number):
        number_zeros = 2 ** i - 1
        Kn = np.append(Kn, ((number_zeros + 1) / 2) + 2 * number_zeros + 3)
        Hn = np.append(Hn, ((number_zeros + 1) / 2) + number_zeros + 2)

    # Recursively compute inverse wavelet transform through the scales
    series_remaining = series_remaining.flatten()
    series_remaining1 = np.concatenate((series_remaining, series_remaining, series_remaining))
    series_remaining1 = series_remaining1.flatten()

    for i in range(scales_number, 0, -1):
        # print("()()()()()() Iteration ()()()()()()")
        # print(i)
        number_zeros = 2 ** (i - 1) - 1
        Kz = insert_zeros(K, number_zeros)
        Hz = insert_zeros(H, number_zeros)

        wavelet_transform_seperate = wavelet_transform[ : , i -1]
        wavelet_transform_seperate = wavelet_transform_seperate.flatten()
        wavelet_transform_extension = np.concatenate((wavelet_transform_seperate, wavelet_transform_seperate, wavelet_transform_seperate))
        wavelet_transform_extension = wavelet_transform_extension.flatten()
        
        A1 = lamda[i - 1] * np.convolve(Kz, wavelet_transform_extension)
        A1 = A1[length + int(Kn[i - 1]) - 1: 2 * length + int(Kn[i - 1]) - 1]

        A2 = np.convolve(Hz, series_remaining1)
        A2 = A2[length + int(Hn[i - 1]) - 1 : (2 * length) + int(Hn[i - 1]) - 1]
        
        series_remaining1 = A1 + A2
        series_remaining1 = np.concatenate((np.transpose(series_remaining1), np.transpose(series_remaining1), np.transpose(series_remaining1)))
        # series_remaining1 = np.concatenate((np.flip(np.transpose(series_remaining1)), series_remaining1, np.flip(np.transpose(series_remaining1))))
        # series_remaining1 = np.concatenate((np.transpose(np.flip(series_remaining1)), np.transpose(series_remaining1), np.transpose(np.flip(series_remaining1))))
        series_remaining1 = series_remaining1.flatten()
    reconstructed_time_series = series_remaining1[length: 2 * length]
    return reconstructed_time_series

#
# Helper functions
#

# Generates normalization coefficients
def normalization_coefficients(scales_number):
    lamda = [1.5, 1.12, 1.03, 1.01]
    if (scales_number > 4):
        lamda.extend([1] * (scales_number - 4))
    return lamda

# Insert's zeros into the proper places
def insert_zeros(original, number_zeros):
    if (number_zeros == 0):
        return original
    elif (number_zeros > 0):
        new_length = (number_zeros + 1) * len(original)  
        new = np.zeros(new_length)  
        index = np.arange(0, new_length - number_zeros, number_zeros + 1)  
        new[index] = original  
        return new
    else:
        return 0

# (--------------------------------------------)
# Testing the Code
# (--------------------------------------------)

# Testing forward
data = [911, 719, 657, 220, 795, 210, 392, 533, 428, 145, 629, 883, 372, 405, 173, 310, 178, 697, 252, 246, 629, 409, 523, 632, 913, 151, 539, 898, 884, 367, 840, 443, 530, 926, 800, 805, 730, 752, 651, 313, 774, 960, 850, 693, 690, 237, 155, 299, 808, 197, 982, 749, 510, 777, 540, 548, 215, 234, 295, 320, 365, 624, 620, 244, 289, 478, 524, 256, 896, 510, 731, 779, 657, 791, 395, 163, 524, 287, 751, 929, 841, 713, 782, 638, 424, 414, 716, 953, 154, 271, 224, 537, 159, 368, 833]
number_scales = 5
wavelet_transform, series_remaining = forward_wavelet_transform(number_scales, data)
# print(wavelet_transform)
# print(series_remaining)

# Testing reverse
reconstructed_time_series = inverse_wavelet_transform(wavelet_transform, series_remaining)
# print(reconstructed_time_series)