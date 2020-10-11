import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


imagePath = sys.argv[1]
if not os.path.isfile(imagePath):
	raise ValueError('File not found')
image = cv2.imread(imagePath, 0)

t_low = int(sys.argv[2])
t_high = int(sys.argv[3])

if not (0 < t_low and t_low < t_high and t_high < 255):
	raise ValueError('0 < t_low < t_high < 255 doesn\'t hold')

canny = cv2.Canny(image, t_low, t_high, L2gradient=True)

plt.title('Canny edge detector, t_low = ' + str(t_low) + ', t_high = ' + str(t_high))
plt.imshow(canny, cmap='gray')
plt.show()