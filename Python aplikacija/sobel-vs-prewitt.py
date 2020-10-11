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

dx = int(sys.argv[2])
dy = int(sys.argv[3])

if dx + dy == 0:
	raise ValueError('Both dx and dy cannot be zeros.')

threshold = int(sys.argv[4])
if threshold > 254 or threshold < 1:
	raise ValueError('Threshold must be a number between 1 and 254')

# Upotreba na Sobel operatorot
sobel = cv2.Sobel(image, cv2.CV_8U, dx, dy, ksize=5)

# Definicija na Prewitt operatorot
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
_, prewitt = cv2.threshold(magnitude, float(threshold) / 255, 255, cv2.THRESH_BINARY)

# Prikaz na rezultatot
plt.subplot(121)
plt.title('Sobel operator dx =' + str(dx) + ', dy = ' + str(dy))
plt.imshow(sobel, cmap='gray')
plt.subplot(122)
plt.title('Prewitt operator, threshold = ' + str(threshold))
plt.imshow(prewitt, cmap='gray')
plt.show()