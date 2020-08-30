# https://stackoverflow.com/questions/21940911/python-image-distortion

import numpy as np
import matplotlib.pyplot as plt

img = lena()

A = img.shape[0] / 3.0
w = 2.0 / img.shape[1]

shift = lambda x: A * np.sin(2.0*np.pi*x * w)

for i in range(img.shape[0]):
    img[:, i] = np.roll(img[:,i], int(shift(i)))

plt.imshow(img, cmap=plt.cm.gray)
plt.show()