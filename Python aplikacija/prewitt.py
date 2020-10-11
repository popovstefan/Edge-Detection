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
threshold = int(sys.argv[2])

# Definicija na operatorot
prewitt_x = np.array([ [-1, 0, 1], [-1, 0, 1], [-1, 0, 1] ], dtype='float32')
prewitt_y = np.array([ [-1, -1, -1], [0, 0, 0], [1, 1, 1] ], dtype='float32')

# Konvolucija vo nasoka na x- oskata
filtered_image_x = cv2.filter2D(image, -1, prewitt_x)
filtered_image_x = abs(filtered_image_x)
filtered_image_x = filtered_image_x / np.amax(filtered_image_x[:])

# Konvolucija vo nasoka na y- oskata
filtered_image_y = cv2.filter2D(image, -1, prewitt_y)
filtered_image_y = abs(filtered_image_y)
filtered_image_y = filtered_image_y / np.amax(filtered_image_y[:])

# Presmetuvanje na magnitudata
magnitude = cv2.add(np.power(filtered_image_x, 2), np.power(filtered_image_y, 2))

# Postavuvanje prag
_, out = cv2.threshold(magnitude, float(threshold) / 255, 255, cv2.THRESH_BINARY)

# Prikaz na rezultat
plt.title('Prewitt operator, threshold = ' + str(threshold))
plt.imshow(out, cmap='gray')
plt.show()