import pywt
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)


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
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
plt.show() 

plt.plot(t, sig)
plt.show()