import cv2
import numpy as np
from math import pi

def directionEstimationLSE(I):
	w = 16 ###############Choose the Block size over which you want to do sigma J
	#I = cv2.imread("dummy.tif");
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

	Gx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=3)
	Gy = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=3)

	J1 = 2*(np.multiply(Gx, Gy))
	J2 = np.multiply(Gx, Gx) - np.multiply(Gy, Gy)

	anisotropy_filter = np.ones((w,w))
	sigma_J1 = cv2.filter2D(J1,-1,anisotropy_filter)
	sigma_J2 = cv2.filter2D(J2,-1,anisotropy_filter)


	##theta_bar == o
	theta_bar = 0.5 * np.arctan(np.divide(sigma_J1, sigma_J2))
	for i,j in np.argwhere(sigma_J2 == 0):
		if (sigma_J1[i,j] > 0):
			theta_bar[i,j] = 0.25*pi
		elif (sigma_J1[i,j] < 0):
			theta_bar[i,j] = (-0.25)*pi
		else:
			theta_bar[i,j] = 0

	#cv2.imshow('gradient square', theta_bar)
	#cv2.waitKey(0)

	phi_x = np.cos(theta_bar)
	phi_y = np.sin(theta_bar)

	####Size of Window to smooth over
	sw = 5

	filtered_phi_x = cv2.blur(phi_x, (sw,sw))
	filtered_phi_y = cv2.blur(phi_y, (sw,sw))

	# cv2.imshow('y', filtered_phi_y)
	# cv2.waitKey(0)
	# cv2.imshow('x', filtered_phi_x)
	# cv2.waitKey(0)

	o_dash = 0.5 * np.arctan(np.divide(filtered_phi_y, filtered_phi_x))
	for i,j in np.argwhere(filtered_phi_x == 0):
		if (filtered_phi_y[i,j] > 0):
			o_dash[i,j] = 0.25*pi
		elif (filtered_phi_y[i,j] < 0):
			o_dash[i,j] = (-0.25)*pi
		else:
			o_dash[i,j] = 0

	# cv2.imshow('gradient square', o_dash)
	# cv2.waitKey(0)
	
	return o_dash

