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

t_low = int(sys.argv[4])
t_high = int(sys.argv[5])

if not (0 < t_low and t_low < t_high and t_high < 255):
	raise ValueError('0 < t_low < t_high < 255 doesn\'t hold')

sobelImage = cv2.Sobel(image, cv2.CV_8U, dx, dy, ksize=5)
cannyImage = cv2.Canny(image, t_low, t_high, L2gradient=True)

plt.subplot(121)
plt.title('Sobel, dx = ' + str(dx) + ', dy = ' + str(dy))
plt.imshow(sobelImage, cmap='gray')
plt.subplot(122)
plt.title('Canny, t_low = ' + str(t_low) + ', t_high = ' + str(t_high))
plt.imshow(cannyImage, cmap='gray')
plt.show()
