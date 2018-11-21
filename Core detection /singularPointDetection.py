import cv2
import numpy as np
from math import pi


# import Normalisation
import coherenceOrientationField
import direction_field
import poincare_index


##Read the image
#I = cv2.imread("/home/neharika/Desktop/DIP/Project/dummy4.tif");

def singularPointDetection(I):

	#I = Normalisation.Normalisation(I)
	
	foreground_mask = coherenceOrientationField.coherenceOrientationField(I)

	# cv2.imshow('foreground_mask', foreground_mask)
	# cv2.waitKey(0)

	directionField = direction_field.direction_field(I, foreground_mask)

	singularPoints = poincare_index.poincare_index(I, foreground_mask, directionField)

	return singularPoints