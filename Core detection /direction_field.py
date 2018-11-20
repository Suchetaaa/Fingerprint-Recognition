import numpy as np
import cv2 
img = cv2.imread('fingerprint_01.png',0)
from matplotlib import pyplot as plt


img_2 = np.pad(img, 4, mode='constant')
Size = img_2.shape
G = np.zeros(9)
G_diff = np.zeros(5)
FloatImage = np.float32(img_2)
Out_Image = np.zeros(img.shape) 

for x in range (4, Size[0]-4):
	for y in range (4,Size[1]-4):
		for i in range (1,9):

			if(i == 1):
				G[i] = (FloatImage[x+2][y]+ FloatImage[x-2][y]+ FloatImage[x+4][y]+ FloatImage[x-4][y])/4.0
				
			elif(i == 2):
				G[i] = (FloatImage[x+2][y+1]+ FloatImage[x-2][y-1]+ FloatImage[x+4][y+2]+ FloatImage[x-4][y-2])/4.0
			elif(i == 3):
				G[i] = (FloatImage[x+2][y+2]+ FloatImage[x-2][y-2]+ FloatImage[x+4][y+4]+ FloatImage[x-4][y-4])/4.0
			elif(i == 4):
				G[i] = (FloatImage[x+1][y+2]+ FloatImage[x-1][y-2]+ FloatImage[x+2][y+4]+ FloatImage[x-2][y-4])/4.0	
			elif(i == 5):
				G[i] = (FloatImage[x][y+2]+ FloatImage[x][y-2]+ FloatImage[x][y+4]+ FloatImage[x][y-4])/4.0
			elif(i == 6):
				G[i] = (FloatImage[x-1][y+2]+ FloatImage[x-2][y+4]+ FloatImage[x+1][y-2]+ FloatImage[x+2][y-4])/4.0
			elif(i == 7):
				G[i] = (FloatImage[x+2][y]+ FloatImage[x-2][y]+ FloatImage[x+4][y]+ FloatImage[x-4][y])/4.0
			else:
				G[i] = (FloatImage[x+2][y]+ FloatImage[x-2][y]+ FloatImage[x+4][y]+ FloatImage[x-4][y])/4.0

		for k in range (1,5):
			G_diff[k] = abs(G[k] - G[k+4])

		i_max = np.argmax(G_diff)
		if (abs(FloatImage[x][y]-G[i_max]) < abs(FloatImage[x][y]-G[i_max+4])):
			Out_Image[x-4][y-4] = i_max
		else:
			Out_Image[x-4][y-4] = i_max+4
print(Out_Image)
print(Out_Image.size)
print(img.size)

