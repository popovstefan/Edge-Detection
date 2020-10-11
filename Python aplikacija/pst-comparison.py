import sys
import os
import math
import numpy as np
import cv2
import mahotas as mh
from matplotlib import pyplot as plt

# Spravuvanje so argumentite

imagePath = sys.argv[1]
if not os.path.isfile(imagePath):
    raise ValueError('File not found')
image = cv2.imread(imagePath, 0)

LPF = float(sys.argv[2])

Phase_strength = float(sys.argv[3])

Warp_strength = float(sys.argv[4])

Threshold_min = float(sys.argv[5])

Threshold_max = float(sys.argv[6])

#####################################
# Phase-stretch-transform algoritam #
#####################################
L = 0.5
x = np.linspace(-L, L, image.shape[0])
y = np.linspace(-L, L, image.shape[1])
[X1, Y1] = (np.meshgrid(x, y))
X = X1.T
Y = Y1.T
[THETA, RHO] = [np.arctan2(Y, X), np.hypot(X, Y)] # cartesian to polar coordinates
 
# Apply localization kernel to the original image to reduce noise
Image_orig_f = np.fft.fft2(image)
expo = np.fft.fftshift(np.exp(-np.power((np.divide(RHO, math.sqrt((LPF ** 2) / np.log(2)))), 2)))
Image_orig_filtered = np.real(np.fft.ifft2((np.multiply(Image_orig_f, expo))))

# Constructing the PST Kernel
PST_Kernel_1 = np.multiply(np.dot(RHO, Warp_strength), np.arctan(np.dot(RHO, Warp_strength))) - 0.5 * np.log(1 + np.power(np.dot(RHO, Warp_strength), 2))
PST_Kernel = PST_Kernel_1 / np.max(PST_Kernel_1) * Phase_strength

# Apply the PST Kernel
temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(Image_orig_filtered))
Image_orig_filtered_PST = np.fft.ifft2(temp)

# Calculate phase of the transformed image
PHI_features = np.angle(Image_orig_filtered_PST)

# find image sharp transitions by thresholding the phase
features = np.zeros((PHI_features.shape[0], PHI_features.shape[1]))
features[PHI_features > Threshold_max] = 1 # Bi-threshold decision
features[PHI_features < Threshold_min] = 1 # as the output phase has both positive and negative values
features[image < (np.amax(image) / 20)] = 0 # Removing edges in the very dark areas of the image (noise)

# apply binary morphological operations to clean the transformed image 
out = features
out = mh.thin(out, 1)
out = mh.bwperim(out, 4)
out = mh.thin(out, 1)
out = mh.erode(out, np.ones((1, 1)));

overlay = mh.overlay(image, out)

plt.subplot(131)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('PST edge detection')
plt.imshow(out, cmap='gray')
plt.subplot(133)
plt.title('Overlay')
plt.imshow(overlay, cmap='gray')
plt.show()
