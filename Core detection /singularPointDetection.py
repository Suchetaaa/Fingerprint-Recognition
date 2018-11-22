import cv2
import numpy as np
from math import pi
from coherenceOrientationField import *
import os
from tqdm import tqdm

def singularPointDetection(path):	
	#returns the singular points
	image = cv2.imread(path)
	return Singular_point_detection(image)

i = 0
j = 0
k = 0
l = 0

image_names1 = os.listdir('/home/neharika/Desktop/DIP/Project/DB1_B')
# image_names2 = os.listdir('DB2_B/')
# image_names3 = os.listdir('DB3_B/')
# image_names4 = os.listdir('DB4_B/')

singularPoint_1 = (-1) * np.ones((80, 2))
singularPoint_2 = (-1) * np.ones((80, 2))
singularPoint_3 = (-1) * np.ones((80, 2))
singularPoint_4 = (-1) * np.ones((80, 2))

for images in tqdm(image_names1):
	(a, b) = singularPointDetection(images)
	singularPoint_1[i,0] = a
	singularPoint_1[i, 1] = b
	i = i+1

	
# for images in tqdm(image_names2):
# 	(a, b) = singularPointDetection(images)
# 	singularPoint_2[j,0] = a
# 	singularPoint_2[j, 1] = b
# 	j = j+1

# for images in tqdm(image_names3):
# 	(a, b) = singularPointDetection(images)
# 	singularPoint_3[k,0] = a
# 	singularPoint_3[k, 1] = b
# 	k = k+1
	
# for images in tqdm(image_names4):
# 	(a, b) = singularPointDetection(images)
# 	singularPoint_4[l,0] = a
# 	singularPoint_4[l, 1] = b
# 	l = l+1
	

np.save('singularPoint_1.npy', singularPoint_1)
# np.save('singularPoint_2.npy', singularPoint_2)
# np.save('singularPoint_3.npy', singularPoint_3)
# np.save('singularPoint_4.npy', singularPoint_4)
