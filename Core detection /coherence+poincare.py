import cv2
import numpy as np
from math import pi
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from Normalisation import *

import pandas as pd
from openpyxl.workbook import Workbook
w1 = 16

#Read the image
I = cv2.imread("103_3.tif");
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
# cv2.imshow('gradient square', I)
# cv2.waitKey(0)


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

# cv2.imshow('gradient square', block_coherence)
# cv2.waitKey(0)

# print(block_coherence)
# print(np.argwhere(block_coherence > 0))
a = np.zeros(sigma_J1.shape)
for i,j in np.argwhere(block_coherence > 0):
	a[i,j] = 1

for i,j in np.argwhere(block_coherence < 0):
	a[i,j] = -1

normalised_image = Normalisation(I, 160, 5900, a)
# cv2.imshow('gradient square', a)
# cv2.waitKey(0)
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
# output = cv2.connectedComponentsWithStats(block_coherence, connectivity , cv2.CV_8S)
# labels = output[1]
# stats = output[2]
# foreground_label = np.argmax(stats, axis =0)[4]

# block_coherence_dash = (-1)*np.ones(block_coherence.shape)

# for i in range(0, block_coherence.shape[0]):
# 	for j in range(0, block_coherence.shape[1]):
# 		if labels[i][j] == foreground_label:
# 			block_coherence_dash = block_coherence[i][j]

# for i,j in np.argwhere(labels == foreground_label):
# 	block_coherence_dash[i,j] = block_coherence[i,j]

print (theta)
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
			print('1')

		for x,y in np.argwhere(store < (-0.5)*pi):
			store[x,y] = pi + store[x,y]


		poincare_image[i,j] = store.sum()/(2*pi)

for g,h in np.argwhere(poincare_image != 0):
	print(poincare_image[g,h])
df = pd.DataFrame(poincare_image)
filepath = 'my_excel_file.xlsx'
df.to_excel(filepath, index=False)
cv2.imshow('i',poincare_image)
cv2.waitKey(0)

		
