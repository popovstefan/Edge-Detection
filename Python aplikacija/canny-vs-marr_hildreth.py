import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Spravuvanje so argumentite
imagePath = sys.argv[1]
if not os.path.isfile(imagePath):
	raise ValueError('File not found')
image = cv2.imread(imagePath, 0)

t_low = int(sys.argv[2])
t_high = int(sys.argv[3])
if not (0 < t_low and t_low < t_high and t_high < 255):
	raise ValueError('0 < t_low < t_high < 255 doesn\'t hold')

sigma = abs(float(sys.argv[4]))

# Canny
canny = cv2.Canny(image, t_low, t_high, L2gradient=True)

## Marr-Hildreth

size = int(2 * (np.ceil(3 * sigma)) + 1)

x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))

normal = 1 / (2.0 * np.pi * sigma ** 2)

kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal # LoG filter

kern_size = kernel.shape[0]
log = np.zeros_like(image, dtype=float)

# applying filter
for i in range(image.shape[0] - (kern_size - 1)):
	for j in range(image.shape[1] - (kern_size - 1)):
		window = image[i : i + kern_size, j : j + kern_size] * kernel
		log[i, j] = np.sum(window)

log = log.astype(np.int64, copy=False)

zero_crossing = np.zeros_like(log)

# computing zero crossing
for i in range(log.shape[0] - (kern_size - 1)):
	for j in range(log.shape[1] - (kern_size - 1)):
		if log[i][j] == 0:
			if (log[i][j - 1] < 0 and log[i][j + 1] > 0) or (log[i][j - 1] < 0 and log[i][j + 1] < 0) or (log[i - 1][j] < 0 and log[i + 1][j] > 0) or (log[i - 1][j] > 0 and log[i + 1][j] < 0):
				zero_crossing[i][j] = 255
		if log[i][j] < 0:
			if (log[i][j - 1] > 0) or (log[i][j + 1] > 0) or (log[i - 1][j] > 0) or (log[i + 1][j] > 0):
				zero_crossing[i][j] = 255
##############


plt.subplot(121)
plt.title('Canny edge detector\n t_low = ' + str(t_low) + ', t_high = ' + str(t_high))
plt.imshow(canny, cmap='gray')
plt.subplot(122)
plt.title('Marr-Hildreth, sigma = ' + str(sigma))
plt.imshow(zero_crossing, cmap='gray')
plt.show()