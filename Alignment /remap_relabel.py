import numpy as np 
import os 
import cv2

def remap_relabel(rotated_image, rotation, center_coord_0, center_coord_1):

	M_inv = cv2.getRotationMatrix2D((center_coord_0, center_coord_1), -1*rotation, 1)
	print M_inv
	rows, cols = rotated_image.shape
	print rows
	for x in range(0,rows):
		for y in range(0,cols):
			new_coord_col = y - center_coord_1 
			new_coord_row = x - center_coord_0
			point = [x, y]
			print point
			point = np.asarray(point)
			point = np.reshape(point, (2, 1))
			original_point = cv2.transform(point, M_inv)
			print original_point
			if (original_point[0] < 0 or original_point[0] > rows or original_point[1] < 0 or original_point[1] > cols) :
				rotated_image[x, y] = -1
	return rotated_image

			
		
	

