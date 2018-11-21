import cv2
import math
import numpy as np
import os
from tqdm import tqdm

M0_all_images = np.zeros((2,320))
V0_all_images = np.zeros((2,320))

i = 0

image_names1 = os.listdir('DB1_B/')
image_names2 = os.listdir('DB2_B/')
image_names3 = os.listdir('DB3_B/')
image_names4 = os.listdir('DB4_B/')

# I = cv2.imread("/home/neharika/Desktop/DIP/Project/dummy4.tif");
# mean = np.mean(I)
# print(mean)



for images in tqdm(image_names1):
	I = cv2.imread('DB1_B/' +images)
	M0_all_images[0][i] = np.mean(I)
	V0_all_images[0][i] = np.var(I)
	i = i+1

for images in tqdm(image_names2):
	I = cv2.imread('DB2_B/' +images)
	M0_all_images[0][i] = np.mean(I)
	V0_all_images[0][i] = np.var(I)
	i = i+1

for images in tqdm(image_names3):
	I = cv2.imread('DB3_B/' +images)
	M0_all_images[0][i] = np.mean(I)
	V0_all_images[0][i] = np.var(I)
	i = i+1
	
for images in tqdm(image_names4):
	I = cv2.imread('DB4_B/' +images)
	M0_all_images[0][i] = np.mean(I)
	V0_all_images[0][i] = np.var(I)
	i = i+1

M0 = np.mean(M0_all_images)
V0 = np.var(V0_all_images)

print (M0)
print (V0)

