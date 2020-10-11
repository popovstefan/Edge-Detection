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

# Upotreba na Sobel operatorot
sobelx8u = cv2.Sobel(image, cv2.CV_8U, dx, dy, ksize=5)

# Prikaz na rezultatot
plt.title('Sobel operator dx =' + str(dx) + ', dy = ' + str(dy))
plt.imshow(sobelx8u, cmap='gray')
plt.show()
