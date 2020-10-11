import sys
import os
import math
import numpy as np
import cv2
import mahotas as mh
from matplotlib import pyplot as plt

#####################################
# Phase-stretch-transform algoritam #
#####################################
def pst_algorithm(image, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max, Morph_flag):
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

	if Morph_flag == 0:
	    out = PHI_features
	    out = (out / np.max(out)) * 3
	else:
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

	return out

if __name__ == '__main__':
	image = cv2.imread('girl-sitting.JPG', 0)
	LPF = 2
	Threshold_min = -1
	Threshold_max = 0.047
	Morph_flag = 1

	Warp_strength = 0.0001
	Phase_strength = 5
	out1 = pst_algorithm(image=image, LPF=LPF, Phase_strength=Phase_strength, Warp_strength=Warp_strength, Threshold_min=Threshold_min, Threshold_max=Threshold_max, Morph_flag=Morph_flag)
	Warp_strength = 14
	Phase_strength = 5
	out2 = pst_algorithm(image=image, LPF=LPF, Phase_strength=Phase_strength, Warp_strength=Warp_strength, Threshold_min=Threshold_min, Threshold_max=Threshold_max, Morph_flag=Morph_flag)
	Warp_strength = 80
	Phase_strength = 5
	out3 = pst_algorithm(image=image, LPF=LPF, Phase_strength=Phase_strength, Warp_strength=Warp_strength, Threshold_min=Threshold_min, Threshold_max=Threshold_max, Morph_flag=Morph_flag)
	Warp_strength = 14
	Phase_strength = 3
	out4 = pst_algorithm(image=image, LPF=LPF, Phase_strength=Phase_strength, Warp_strength=Warp_strength, Threshold_min=Threshold_min, Threshold_max=Threshold_max, Morph_flag=Morph_flag)
	Warp_strength = 14
	Phase_strength = 50
	out5 = pst_algorithm(image=image, LPF=LPF, Phase_strength=Phase_strength, Warp_strength=Warp_strength, Threshold_min=Threshold_min, Threshold_max=Threshold_max, Morph_flag=Morph_flag)

	plt.subplot(231)
	plt.title('warp_str = 0, phase_str = 5')
	plt.imshow(out1, cmap='gray')

	plt.subplot(232)
	plt.title('warp_str = 14, phase_str = 5')
	plt.imshow(out2, cmap='gray')

	plt.subplot(233)
	plt.title('warp_str = 80, phase_str = 5')
	plt.imshow(out3, cmap='gray')

	plt.subplot(234)
	plt.title('\nwarp_str = 14, phase_str = 3')
	plt.imshow(out4, cmap='gray')

	plt.subplot(235)
	plt.title('\nwarp_str = 14, phase_str = 50')
	plt.imshow(out5, cmap='gray')

	plt.subplot(236)
	plt.title('\noriginal')
	plt.imshow(image, cmap='gray')

	plt.show()