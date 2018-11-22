import cv2
import numpy as np
from math import pi
import scipy as sp
import math
import numpy as np 
from scipy import stats
from direction_field import *

def Singular_point_detection(img_2):
	#Read the image

	img_3 = img_2
	img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	I = cv2.dilate(img_2,kernel,iterations = 1)

	## Gaussian Blurring to remove noise
	G = cv2.GaussianBlur(I,(5,5),0)

	#Sobel Operator
	Gx = cv2.Sobel(G,cv2.CV_64F,1,0,ksize=3)
	Gy = cv2.Sobel(G,cv2.CV_64F,0,1,ksize=3)


	# derivative calculation Gradient
	J1 = 2*(np.multiply(Gx, Gy))
	J2 = np.multiply(Gx, Gx) - np.multiply(Gy, Gy)
	J3 = np.multiply(Gx, Gx) + np.multiply(Gy, Gy)
	Gradient = np.sqrt(J3)

	# gradient smoothening
	w1 = 16
	anisotropy_filter = np.ones((w1,w1))
	sigma_J1 = cv2.filter2D(J1,-1,anisotropy_filter)
	sigma_J2 = cv2.filter2D(J2,-1,anisotropy_filter)
	sigma_J3 = cv2.filter2D(J3,-1,anisotropy_filter)

	# less smoothened  Gradient
	w2 = 10
	anisotropy_filter_0 = np.ones((w2,w2))
	sigma_J3_0 = cv2.filter2D(J3,-1,anisotropy_filter_0)

	# Direction Calculation
	theta_bar = 0.5 * np.arctan(np.divide(sigma_J1, sigma_J2))
	for i,j in np.argwhere(sigma_J2 == 0):
		if (sigma_J1[i,j] > 0):
			theta_bar[i,j] = 0.25*pi
		elif (sigma_J1[i,j] < 0):
			theta_bar[i,j] = (-0.25)*pi
		else:
			theta_bar[i,j] = 0

	theta_dash = (pi/2) + theta_bar


	## threshold matrix on coherence
	gt = 0.10
	Grad_max = np.amax(Gradient)
	Grad_min = np.amin(Gradient)
	Gth = gt * (abs(Grad_max) - abs(Grad_min)) + abs(Grad_min)

	## calculation of block coherence
	block_coherence = np.sqrt(np.multiply(sigma_J1, sigma_J1) + np.multiply(sigma_J2, sigma_J2))
	block_coherence = np.divide(block_coherence, sigma_J3)
	for i,j in np.argwhere(sigma_J3 < Gth*Gth*(w1*w1)):
		block_coherence[i,j] = (-1)

	#Mask calculation!!	
	a = np.zeros(sigma_J1.shape)
	for i,j in np.argwhere(block_coherence > 0):
		a[i,j] = 1
	for i,j in np.argwhere(block_coherence < 0):
		a[i,j] = -1

	#### 8 field direction calculation
	theta = Eight_field_Direction(img_3)

	poincare_image = np.zeros(a.shape)
	for i,j in np.argwhere(a != -1):
		store = np.zeros([3,3])
		if ((i-1 >= 0) and (i+1 <= a.shape[0]-1) and (j-1 >= 0) and (j+1 <= a.shape[1]-1)):
			store[0,0] = theta[i-1,j] - theta[i-1, j-1]
			store[0,1] = theta[i-1, j+1] - theta[i-1, j]
			store[0,2] = theta[i,j+1] - theta[i-1, j+1]
			store[1,2] = theta[i+1, j+1] - theta[i, j+1]
			store[2,2] = theta[i+1,j] - theta[i+1, j+1]
			store[2,1] = theta[i+1, j-1] - theta[i+1, j]
			store[2,0] = theta[i, j-1] - theta[i+1, j-1]
			store[1,0] = theta[i-1, j-1] - theta[i, j-1]
			
			for x,y in np.argwhere(store > pi/2):
				store[x,y] = pi - store[x,y]
				
			for x,y in np.argwhere(store < (-0.5)*pi):
				store[x,y] = pi + store[x,y]


			poincare_image[i,j] = store.sum()/(2*pi)
	f = 0
	delta_image = np.zeros(img_2.shape)
	#### Detecting all the singularities
	# for x,y in np.argwhere(poincare_image == 0.5):
	# 	img_3[x,y] = [0,0,255]
	# 	img_3[x+1, y] = [0,0,255]
	# 	img_3[x, y+1] = [0,0,255]
	# 	img_3[x+1, y+1] = [0,0,255]

	# Detecting the maximum gradient 
	# CORE is the point of maximum ridge

	index_1, index_2 = [-1,-1]
	max_gradient = 0	
	for x,y in np.argwhere(poincare_image == 0.5):
		if(sigma_J3_0[x,y] > max_gradient):
			index_1,index_2 = x,y
			max_gradient = sigma_J3_0[x,y]


	singular_point_Index = []
	singular_point_Index.append(index_1)
	singular_point_Index.append(index_2)

	return singular_point_Index

