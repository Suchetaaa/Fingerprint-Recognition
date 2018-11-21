import cv2
import numpy as np
from math import pi




#Read the image
#I = cv2.imread("/home/neharika/Desktop/DIP/Project/dummy4.tif");



def coherenceOrientationField(I):
	w1 = 16
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	#print(I.dtype)
	#print(I.shape)
	#I = np.ones((100,100))
	#Gaussian Filter

	G = cv2.GaussianBlur(I,(5,5),0)

	#Sobel Operator
	Gx = cv2.Sobel(G,cv2.CV_64F,1,0,ksize=3)
	Gy = cv2.Sobel(G,cv2.CV_64F,0,1,ksize=3)

	J1 = 2*(np.multiply(Gx, Gy))
	J2 = np.multiply(Gx, Gx) - np.multiply(Gy, Gy)
	J3 = np.multiply(Gx, Gx) + np.multiply(Gy, Gy)

	Gradient = np.sqrt(J3)
	cv2.imshow('gradient square', I)
	cv2.waitKey(0)


	#anisotropy orientation estimation is done using convoltion with a 16*16 ones filter

	anisotropy_filter = np.ones((16,16))
	sigma_J1 = cv2.filter2D(J1,-1,anisotropy_filter)
	sigma_J2 = cv2.filter2D(J2,-1,anisotropy_filter)
	sigma_J3 = cv2.filter2D(J3,-1,anisotropy_filter)


	theta_bar = 0.5 * np.arctan(np.divide(sigma_J1, sigma_J2))
	for i,j in np.argwhere(sigma_J2 == 0):
		if (sigma_J1[i,j] > 0):
			theta_bar[i,j] = 0.25*pi
		elif (sigma_J1[i,j] < 0):
			theta_bar[i,j] = (-0.25)*pi
		else:
			theta_bar[i,j] = 0

	theta = (pi/2) + theta_bar
	gt = 0.10
	Grad_max = np.amax(Gradient)
	Grad_min = np.amin(Gradient)

	Gth = gt * (abs(Grad_max) - abs(Grad_min)) + abs(Grad_min)
	## threshold matrix

	block_coherence = np.sqrt(np.multiply(sigma_J1, sigma_J1) + np.multiply(sigma_J2, sigma_J2))
	block_coherence = np.divide(block_coherence, sigma_J3)
	for i,j in np.argwhere(sigma_J3 < Gth*Gth*(w1*w1)):
		block_coherence[i,j] = (-1)

	cv2.imshow('gradient square', block_coherence)
	cv2.waitKey(0)

	print(block_coherence)
	print(np.argwhere(block_coherence > 0))
	"""
	a = np.zeros(sigma_J1.shape)
	for i,j in np.argwhere(block_coherence > 0):
		a[i,j] = 1
	"""
	#cv2.imshow('gradient square', a)
	#cv2.waitKey(0)

	#block_coherence = np.zeros(sigma_J1.shape)
	###############try to optimise this part of the code
	# compare_matrix = sigma_J3 / (16*16) < np.multiply(Gth, Gth)

	# for i in range(0, compare_matrix.shape[0]):
	# 	for j in range(0, compare_matrix.shape[1]):
	# 		if compare_matrix[i][j] == 1:
	# 			block_coherence[i][j] = -1

	# ############################
	# connectivity = 8
	# block_coherence = block_coherence.astype(np.int8)
	# output = cv2.connectedComponentsWithStats(block_coherence, connectivity, cv2.CV_8S)
	# labels = output[1]
	# stats = output[2]
	# foreground_label = np.argmax(stats, axis =0)[4]

	block_coherence_dash = (-1)*np.ones(block_coherence.shape)
	foreground_mask = np.zeros(I.shape)

	#for i,j in np.argwhere(labels == foreground_label):
	for i,j in np.argwhere(block_coherence > 0):
		block_coherence_dash[i,j] = block_coherence[i,j]
		foreground_mask[i,j] = 1

	kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
	foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ELLIPSE, kernel_closing)
	foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ELLIPSE, kernel_closing)
	foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ELLIPSE, kernel_closing)
	foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ELLIPSE, kernel_closing)
	foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ELLIPSE, kernel_closing)

	cv2.imshow('foreground_mask', foreground_mask)
	cv2.waitKey(0)


	"""

	for i in range(0, block_coherence.shape[0]):
		for j in range(0, block_coherence.shape[1]):
			if labels[i][j] == foreground_label:
				block_coherence_dash = block_coherence[i][j]
	"""			

	return foreground_mask


#foreground_mask =  coherenceOrientationField(I)
#print(foreground_mask)