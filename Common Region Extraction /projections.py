import numpy as np 
import cv2 
import math 
from PIL import Image

input_img = cv2.imread("Input Image.tif", 0)
reg_img = cv2.imread("Registered Image.tif", 0)

rows, cols = input_img.shape
max_height_reduction = 20
max_width_reduction = 20

best_norm = -99999
best_height_reduction_top = 50
best_height_reduction_bottom = 50
best_width_reduction_left = 50
best_width_reduction_right = 50

# i - top, j - bottom
# k - left, l - right

for i in range(0,max_height_reduction) :
	for j in range(0,max_height_reduction) :
		for k in range(0,max_width_reduction) :
			for l in range(0,max_width_reduction) :
				input_dummy = np.zeros((rows - i - j, cols - k - l))
				reg_dummy = np.zeros((rows - i - j, cols - k - l))
				input_dummy = input_img[i : (rows - i - j - 1), k : (cols - k - l - 1)]
				reg_dummy = reg_img[i : (rows - i - j - 1), k : (cols - k - l - 1)]

				input_row_sum = input_dummy.sum(axis = 1)
				input_col_sum = input_dummy.sum(axis = 0)
				reg_row_sum = reg_dummy.sum(axis = 1)
				reg_col_sum = reg_dummy.sum(axis = 0)

				input_row_sum = np.reshape(input_row_sum, (input_row_sum.shape[0], 1))
				input_col_sum = np.reshape(input_col_sum, (input_col_sum.shape[0], 1))
				reg_row_sum = np.reshape(reg_row_sum, (reg_row_sum.shape[0], 1))
				reg_col_sum = np.reshape(reg_col_sum, (reg_col_sum.shape[0], 1))

				input_array = np.concatenate((input_row_sum, input_col_sum), axis = 0)
				reg_array = np.concatenate((reg_row_sum, reg_col_sum), axis = 0)
				input_array = np.asarray(input_array)
				reg_array = np.asarray(reg_array)
				norm_arr = np.subtract(input_array, reg_array)
				dist = np.linalg.norm(norm_arr)

				if (dist < best_norm) : 
					best_norm = dist 
					best_height_reduction_top = i
					best_height_reduction_bottom = j
					best_width_reduction_left = k
					best_width_reduction_right = l

input_final = np.zeros((rows - best_height_reduction_top - best_height_reduction_bottom, cols - best_width_reduction_left - best_width_reduction_right))
reg_final = np.zeros((rows - best_height_reduction_top - best_height_reduction_bottom, cols - best_width_reduction_left - best_width_reduction_right))

input_final = input_img[best_height_reduction_top : (rows - best_height_reduction_top - best_height_reduction_bottom - 1), best_width_reduction_left : (cols - best_width_reduction_left - best_width_reduction_right - 1)]
reg_final = reg_img[best_height_reduction_top : (rows - best_height_reduction_top - best_height_reduction_bottom - 1), best_width_reduction_left : (cols - best_width_reduction_left - best_width_reduction_right - 1)]

input_img_final = Image.fromarray(input_final)
input_img_final.show()
reg_img_final = Image.fromarray(reg_final)
reg_img_final.show()

