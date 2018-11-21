import numpy as np 
import cv2 
import math 

input_img_path = To be written
reg_img_path = To be written 
input_img = cv2.imread(input_img_path)
ref_img = cv2.imread(ref_img_path)

rows, cols = input_img.size
max_height_reduction = 50
max_width_reduction = 50

best_norm = -99999
best_height_reduction = 50
best_width_reduction = 50

for x in xrange(0,max_height_reduction):
	for y in yrange(0,max_width_reduction):
		input_dummy = np.zeroes((rows - 2*x, cols - 2*y))
		reg_dummy = np.zeros((rows - 2*x, cols - 2*y))
		input_dummy = input_img[x : (rows - x - 1), y : (cols - y - 1)]
		reg_dummy = ref_img[x : (rows - x - 1), y : (cols - y - 1)]

		input_row_sum = input_dummy.sum(axis = 1)
		input_col_sum = input_dummy.sum(axis = 0)
		reg_row_sum = ref_dummy.sum(axis = 1)
		reg_col_sum = ref_dummy.sum(axis = 0)

		input_array = [input_row_sum, input_col_sum]
		reg_array = [reg_row_sum, reg_col_sum]
		dist = np.linalg.norm(input_array - reg_array)

		if (dist < best_norm) : 
			best_norm = dist 
			best_height_reduction = x
			best_width_reduction = y

input_final = np.zeroes((rows - 2*best_height_reduction, cols - 2*best_width_reduction))
reg_final = np.zeroes((rows - 2*best_height_reduction, cols - 2*best_width_reduction))

input_final = input_img[best_height_reduction : (rows - best_height_reduction - 1), best_width_reduction : (cols - best_width_reduction - 1)]
reg_final = reg_img[best_height_reduction : (rows - best_height_reduction - 1), best_width_reduction : (cols - best_width_reduction - 1)]

