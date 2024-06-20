import pywt
import numpy as np
import matplotlib.pyplot as plt

# ()()()()()()()()()()()()()
# This file is just for me to tryout code, it doesn't really do anything
# ()()()()()()()()()()()()()

t = np.linspace(0, 1, 200, endpoint=False)
t2 = np.linspace(0, 1, 30, endpoint=False)



# Creating the bad data set
number_pre_jump = 125
pre_jump_value = 0.123
number_post_jump = 75
post_jump_value = 0.78
noise_level = 0.05
total_number = number_pre_jump + number_post_jump
number_scales = 3
jump_threshold = 0.5

pre_jump = np.empty(number_pre_jump)
for i in range(number_pre_jump):
    pre_jump[i] = pre_jump_value
post_jump = np.empty(number_post_jump)
for i in range(number_post_jump):
    post_jump[i] = post_jump_value
smooth_original_data = np.concatenate((pre_jump, post_jump))
sig = smooth_original_data + noise_level * np.random.randn(total_number)
widths = np.arange(1, 31)

cwtmatr, freqs = pywt.cwt(sig, widths, 'cgau1')

cwtmatr = np.transpose(cwtmatr)
freqs = np.transpose(freqs)

plt.subplot(1, 3, 1)
plt.plot(t, sig)

plt.subplot(1, 3, 2)
plt.plot(t, cwtmatr)

plt.subplot(1, 3, 3)
plt.plot(t2, freqs)

plt.show()
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
#plt.show() 

wavelet_transform = cwtmatr
print(wavelet_transform)

alpha_values = np.empty(total_number)

# Creating global x (log of scale values)
normalized_scale = np.arange(30) + 1

# Method 1 for determing x
log_normalized_scale = np.log(normalized_scale)

# Method 2 for determing x
# log_normalized_scale = np.empty(number_scales)
# for i in range(number_scales):
#     value = int(normalized_scale[i])
#     print(2 ** (-1 * value))
#     log_normalized_scale[i] = math.log(2 ** (-1 * value))

# Computing alpha at each time
for row in range(total_number):
    current_row = wavelet_transform[row, :]
    log_current_row = np.log(np.abs(current_row))
    print(log_current_row)
    alpha = np.polyfit(log_normalized_scale, log_current_row, 1)[0]
    
    # Method 1 for alpha
    alpha_values[row] = alpha

    # Method 2 for alpha
    # new_alpha = -alpha / math.log(2)
    # alpha_values[row] = new_alpha
print(alpha_values)

print("HERE")
plt.axvline(x = (number_pre_jump / total_number), color = "r")
plt.axhline(y = 0, color = 'r')
plt.plot(t, alpha_values)
plt.show()