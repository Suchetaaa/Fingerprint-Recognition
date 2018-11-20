import numpy as np 
import cv2 
import math 

num_rotations = 80
max_blpoc = -1000
initial_rot = -40
best_rot = initial_rot

for x in xrange(0,num_rotations+1) :
	rot = initial_rot + x 
	transformation = cv2.getRotationMatrix2D((reg_img_coord[0], reg_img_coord[1]), rot, 1)
	rotated_img = cv2.warpAffine(displaced_image, transformation, (rows, cols))
	a = BLPOC(rotated_img, ref_img)
	if (max_blpoc < a) :
		max_blpoc = a
		best_rot = rot

transformation = cv2.getRotationMatrix2D((reg_img_coord[0], reg_img_coord[1]), best_rot, 1)
final_rot_img = cv2.warpAffine(displaced_image, transformation, (rows, cols))



	


