import sys
import os
from cnnClass import PyCNN

sourceImagePath = sys.argv[1]
if not os.path.isfile(sourceImagePath):
	raise ValueError('File not found')

destinationImagePath = sys.argv[2]

cnn = PyCNN()

cnn.edgeDetection(sourceImagePath, destinationImagePath)