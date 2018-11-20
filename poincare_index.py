import numpy as np 
import scipy as sp
from math import *
import cv2
def Poincare_value (test_image, direction_image):
	n = 4
	Delta = np.zeros(n)
	direction_image_0 = cv2.imread('fingerprint_01.png',0)
	Size = direction_image_0.shape
	direction_image = np.float32(direction_image_0)
	Poincare_index = np.zeros(Size)
	print(Size)
	for x in range (0, Size[0]-1):
		for y in range (1, Size[1]-1):
			print ('x ', x, ' y ', y )
			Delta[0] = direction_image[x][y] - direction_image[x+1][y]
			Delta[1] = direction_image[x+1][y] - direction_image[x+1][y+1]
			Delta[2] = direction_image[x+1][y+1] - direction_image[x][y+1]
			Delta[3] = direction_image[x][y+1] - direction_image[x][y]
			for k in range (0, n):
				if (abs(Delta[k]) < pi/2):
					Delta[k] = Delta[k]
				elif(Delta[k] < (-1)*pi/2):
					Delta[k] = Delta[k] + pi
				elif  (Delta[k] > pi/2):
					Delta[k] =  Delta[k] - pi
			Poincare_index[x][y] = Delta.sum()/(2*pi)
	return Poincare_index 
## retrns a matrix with poincare value at each pixel
## Runtime : 7 secfor an image (480X400) 
				