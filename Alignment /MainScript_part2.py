import numpy as np 
import os 
from alignment_cores import*
# from alignment_poc import align_poc
from BLPOC import*
from direction_matching import*
from segmentation import*

input_core_bool = 1
reg_core_bool = 1
reg_coord = [100, 100]
input_coord = [105, 108]
num_rotations = 2
scale = 0.3
k1 = 100
k2 = 100

input_img_path = "Input Image.tif"
reg_img_path = "Registered Image.tif"
input_img = cv2.imread(input_img_path, 0)

dataset_path = "DB1_B"
for i in os.listdir(dataset_path) : 
	print i 


if (input_core_bool == 1 and reg_core_bool == 1) :
	displaced_reg = align_cores(input_coord, reg_coord, input_img_path, reg_img_path)
# 	disp = "displaced_reg.png"
# 	displaced_reg = os.path.join(present_path, disp)
	rot_reg_img = match_dir(num_rotations, input_img_path, displaced_reg, input_coord, k1, k2)
# 	reg_corrected = "Rotated Image.tif"
# 	reg_corrected_path = os.path.join(present_path, reg_corrected)
	top_row, bottom_row, left_col, right_col = segment(rot_reg_img, scale)
	new_reg = np.zeros((bottom_row - top_row + 1, right_col - left_col + 1))
	new_input = np.zeros((bottom_row - top_row + 1, right_col - left_col + 1))
	new_reg = rot_reg_img[top_row:bottom_row, left_col:right_col]
	new_input = input_img[top_row:bottom_row, left_col:right_col]
	reg_segmented = Image.fromarray(new_reg)
	input_segmented = Image.fromarray(new_input)
	reg_segmented.show()
	input_segmented.show()

	blpoc_val = BLPOC(new_input, new_reg, k1, k2)
	print blpoc_val









