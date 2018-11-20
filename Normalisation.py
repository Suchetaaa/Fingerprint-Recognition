import numpy as np 
import scipy as sp
import math

def Normalisation (test_image, M_0, V_0):
	Mean = mean.test_image()
	output_image = np.matrix(test_image.size)
	Size = test_image.shape()
	V_i = test_image.var()
	for i in range (0, Size[0]):
		for j in range (0, Size[1]):
			if (test_image[i][j] > Mean):
				output_image[i][j] = M_0 + math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i)
			else:
				output_image[i][j] = M_0 - math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i) 


	return output_image



