import cv2
import numpy as np
from math import pi
from coherenceOrientationField import *

def singularPointDetection(path):	
	#returns the singular points
	image = cv2.imread(path)
	return Singular_point_detection(image)

	
