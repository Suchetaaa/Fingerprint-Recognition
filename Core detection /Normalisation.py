import numpy as np 
import scipy as sp
import math

#As computed on the database; These are the results for overall normalisation and not block normalisation
#This is not what is the most optimum but still better than nothing
# M0 = 74.98999708879379
# V0 = 2011395.439145634



def Normalisation (test_image):
	M0 = 74.98999708879379
	V0 = 2011395.439145634

	Mean = np.mean(test_image)
	V_i = np.var(test_image)

	output_image = np.zeros(test_image.shape)
	Size = test_image.shape()
	

	"""
	for i in range (0, Size[0]):
		for j in range (0, Size[1]):
			if (test_image[i][j] > Mean):
				output_image[i][j] = M_0 + math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i)
			else:
				output_image[i][j] = M_0 - math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i) 
	"""

	for i,j in np.argwhere(test_image > Mean):
		output_image[i][j] = M_0 + math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i)

	for i,j in np.argwhere(test_image <= Mean):
		output_image[i][j] = M_0 - math.sqrt(V_0*math.pow((test_image[i][j]-Mean),2)/V_i)


	return output_image



