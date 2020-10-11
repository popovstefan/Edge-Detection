import sys
import os
import cv2
import math
import numpy as np
import mahotas as mh
from matplotlib import pyplot as plt

imagePath = sys.argv[1]

if not os.path.isfile(imagePath):
	raise ValueError('File not found')

image = cv2.imread(imagePath, 0)
threshold = float(sys.argv[2])


g1 = [[1, math.sqrt(2), 1], [0, 0, 0], [-1, -math.sqrt(2), -1]]

for row in range(len(g1)):
	for column in range(len(g1[row])):
		g1[row][column] *= (1 / 2 * math.sqrt(2))


g2 = [[1, 0, -1], [math.sqrt(2), 0, -math.sqrt(2)], [1, 0, -1]]
for row in range(len(g2)):
	for column in range(len(g2[row])):
		g2[row][column] *= (1 / 2 * math.sqrt(2))

g3 = [[-1, 0, math.sqrt(2)], [1, 0, -1], [-math.sqrt(2), 1, 0]]
for row in range(len(g3)):
	for column in range(len(g3[row])):
		g3[row][column] *= (1 / 2 * math.sqrt(2))

g4 =[[math.sqrt(2), -1, 0], [-1, 0, 1], [0, 1, -math.sqrt(2)]]
for row in range(len(g4)):
	for column in range(len(g4[row])):
		g4[row][column] *= (1 / 2 * math.sqrt(2))

g5 = [[0, 1, 0], [-1, 0, -1], [0, 1, 0]]
for row in range(len(g5)):
	for column in range(len(g5[row])):
		g5[row][column] *= (1 / 2)

g6 = [[-1, 0, 1], [0, 0, 0], [1, 0, -1]]
for row in range(len(g6)):
	for column in range(len(g6[row])):
		g6[row][column] *= (1 / 2)

g7 = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
for row in range(len(g7)):
	for column in range(len(g7[row])):
		g7[row][column] *= (1 / 6)

g8 = [[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]
for row in range(len(g8)):
	for column in range(len(g8[row])):
		g8[row][column] *= (1 / 6)

g9 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
for row in range(len(g9)):
	for column in range(len(g9[row])):
		g9[row][column] *= (1 / 3)


g1 = np.array([np.array(x, dtype='float32') for x in g1], dtype='float32')
g2 = np.array([np.array(x, dtype='float32') for x in g2], dtype='float32')
g3 = np.array([np.array(x, dtype='float32') for x in g3], dtype='float32')
g4 = np.array([np.array(x, dtype='float32') for x in g4], dtype='float32')
g5 = np.array([np.array(x, dtype='float32') for x in g5], dtype='float32')
g6 = np.array([np.array(x, dtype='float32') for x in g6], dtype='float32')
g7 = np.array([np.array(x, dtype='float32') for x in g7], dtype='float32')
g8 = np.array([np.array(x, dtype='float32') for x in g8], dtype='float32')
g9 = np.array([np.array(x, dtype='float32') for x in g9], dtype='float32')



filtered_image1 = cv2.filter2D(src=image, ddepth=-1, kernel=g1)
filtered_image1 = np.square(filtered_image1)
filtered_image2 = cv2.filter2D(src=image, ddepth=-1, kernel=g2)
filtered_image2 = np.square(filtered_image2)
filtered_image3 = cv2.filter2D(src=image, ddepth=-1, kernel=g3)
filtered_image3 = np.square(filtered_image3)
filtered_image4 = cv2.filter2D(src=image, ddepth=-1, kernel=g4)
filtered_image4 = np.square(filtered_image4)
filtered_image5 = cv2.filter2D(src=image, ddepth=-1, kernel=g5)
filtered_image5 = np.square(filtered_image5)
filtered_image6 = cv2.filter2D(src=image, ddepth=-1, kernel=g6)
filtered_image6 = np.square(filtered_image6)
filtered_image7 = cv2.filter2D(src=image, ddepth=-1, kernel=g7)
filtered_image7 = np.square(filtered_image7)
filtered_image8 = cv2.filter2D(src=image, ddepth=-1, kernel=g8)
filtered_image8 = np.square(filtered_image8)
filtered_image9 = cv2.filter2D(src=image, ddepth=-1, kernel=g9)
filtered_image9 = np.square(filtered_image9)


S = cv2.add(filtered_image1, filtered_image2)
S = cv2.add(S, filtered_image3)
S = cv2.add(S, filtered_image4)
S = cv2.add(S, filtered_image5)
S = cv2.add(S, filtered_image6)
S = cv2.add(S, filtered_image7)
S = cv2.add(S, filtered_image8)
S = cv2.add(S, filtered_image9)


M = cv2.add(filtered_image1, filtered_image2)
M = cv2.add(M, filtered_image3)
M = cv2.add(M, filtered_image4)

final = np.cos(np.sqrt(np.divide(M, S, where=M!=0)))

_, final = cv2.threshold(final, threshold / 100, 255, cv2.THRESH_BINARY)

plt.title('Frei-chen, threshold = ' + str(int(threshold)) + '%')
plt.imshow(final, cmap='gray')
plt.show()