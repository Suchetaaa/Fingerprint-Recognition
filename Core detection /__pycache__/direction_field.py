import math
import numpy as np
import cv2 
from math import pi
from scipy import stats

def Eight_field_Direction(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_2 = np.pad(img, 4, mode='constant')
	Size = img_2.shape
	G = np.zeros(9)
	G_diff = np.zeros(5)
	FloatImage = np.float32(img_2)
	im3 = np.zeros(img.shape) 
	im_out = np.zeros(img.shape)
	im_out_2 = np.zeros(img.shape) 
	for x in range (4, Size[0]-4):
		for y in range (4,Size[1]-4):
			for i in range (1,9):

				if(i == 1):
					G[i] = (FloatImage[x+2][y]+ FloatImage[x-2][y]+ FloatImage[x+4][y]+ FloatImage[x-4][y])/4.0
					
				elif(i == 2):
					G[i] = (FloatImage[x-1][y+2]+ FloatImage[x-2][y+4]+ FloatImage[x+1][y-2]+ FloatImage[x+2][y-4])/4.0
				elif(i == 3):
					G[i] = (FloatImage[x-2][y+2]+ FloatImage[x+2][y-2]+ FloatImage[x-4][y+4]+ FloatImage[x+4][y-4])/4.0
				elif(i == 4):
					G[i] = (FloatImage[x-2][y+1]+ FloatImage[x-4][y+2]+ FloatImage[x+2][y-1]+ FloatImage[x+4][y-2])/4.0	
				elif(i == 5):
					G[i] = (FloatImage[x-2][y]+ FloatImage[x-4][y]+ FloatImage[x+2][y]+ FloatImage[x+4][y])/4.0
				elif(i == 6):
					G[i] = (FloatImage[x-2][y-1]+ FloatImage[x+2][y+1]+ FloatImage[x-4][y-2]+ FloatImage[x+4][y+2])/4.0
				elif(i == 7):
					G[i] = (FloatImage[x-2][y-2]+ FloatImage[x-4][y-4]+ FloatImage[x+2][y+2]+ FloatImage[x-4][y-4])/4.0
				else:
					G[i] = (FloatImage[x+1][y+2]+ FloatImage[x+2][y+4]+ FloatImage[x-1][y-2]+ FloatImage[x-2][y-4])/4.0
			for k in range (1,5):
				G_diff[k] = abs(G[k] - G[k+4])
			i_max = np.argmax(G_diff)
			if (abs(FloatImage[x][y]-G[i_max]) < abs(FloatImage[x][y]-G[i_max+4])):
				im3[x-4][y-4] = i_max
			else:
				im3[x-4][y-4] = i_max+4
	w1 = 4
	im_current = im3
	im_next = im3
	
	for x in range (0, img.shape[0]):
		for y in range (0, img.shape[1]):
			x_start = max (0, x-w1 ) 
			x_end = min(img.shape[0]-1, x+w1)
			y_start = max(0, y-w1 ) 
			y_end = min(img.shape[1]-1, y+w1)

			sub_img = im_next[x_start:x_end:1, y_start:y_end:1] 
			z = stats.mode(sub_img, axis=None)
			im_out_2[x][y] = z[0][0]
			
	im_out_2 = (im_out_2 - np.ones(im_out_2.shape))*pi/8.0
	return im_out_2