import cv2
import numpy as np
from math import pi


#Read the image
I = cv2.imread("dummy.tif");
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
print(I.dtype)
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

#anisotropy orientation estimation is done using convoltion with a 16*16 ones filter

anisotropy_filter = np.ones((16,16))
sigma_J1 = cv2.filter2D(J1,-1,anisotropy_filter)
sigma_J2 = cv2.filter2D(J2,-1,anisotropy_filter)
sigma_J3 = cv2.filter2D(J2,-1,anisotropy_filter)

theta_bar = 0.5 * np.arctan(np.divide(sigma_J1, sigma_J2))
#theta_bar = np.zeros(sigma_J1.shape)
theta = (pi/2) + theta_bar

gt = 0.15
Grad_max = np.amax(Gradient)
Grad_min = np.amin(Gradient)

Gth = gt * (abs(Grad_max) - abs(Grad_min)) + abs(Grad_min)


block_coherence = np.sqrt(np.multiply(sigma_J1, sigma_J1) + np.multiply(sigma_J2, sigma_J2))
block_coherence = np.divide(block_coherence, sigma_J3)
#block_coherence = np.zeros(sigma_J1.shape)
###############try to optimise this part of the code
compare_matrix = sigma_J3 / (16*16) < np.multiply(Gth, Gth)

for i in range(0, compare_matrix.shape[0]):
	for j in range(0, compare_matrix.shape[1]):
		if compare_matrix[i][j] == 1:
			block_coherence[i][j] = -1

############################
connectivity = 8
block_coherence = block_coherence.astype(np.int8)
output = cv2.connectedComponentsWithStats(block_coherence, connectivity, cv2.CV_8S)
labels = output[1]
stats = output[2]
foreground_label = np.argmax(stats, axis =0)[4]

block_coherence_dash = (-1)*np.ones(block_coherence.shape)

for i in range(0, block_coherence.shape[0]):
	for j in range(0, block_coherence.shape[1]):
		if labels[i][j] == foreground_label:
			block_coherence_dash = block_coherence[i][j]


