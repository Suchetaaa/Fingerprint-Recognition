import numpy as np 
import os 
from alignment_cores import*
# from alignment_poc import align_poc
from BLPOC import*
from direction_matching import*
from segmentation import*

database = np.load('new_arr.npy')

num_rotations = 100
scale = 0.2
k1 = 150
k2 = 150
k1_1 = 100
k2_1 = 100
num_test = 2
num_subjects = 10
num_img_sub = 8

iter_var = 0
k = 0

data_path = "DB1_B"
test_img = "101_7.tif"
test_iter = 6
images_all = os.listdir(data_path)

best_blpoc = -100
best_img = 0

path_train = list()
train_cores = np.zeros(((num_img_sub-num_test)*num_subjects, 2))
for x in xrange(0,num_subjects):
	for y in range(0,(num_img_sub-num_test)) :
		# print x*(num_img_sub-num_test) + y + x*num_test
		train_cores[iter_var][0] = database[x*(num_img_sub-num_test) + y + x*num_test][0]
		train_cores[iter_var][1] = database[x*(num_img_sub-num_test) + y + x*num_test][1]
		path_train.append(images_all[x*(num_img_sub-num_test) + y + x*num_test])
		iter_var = iter_var + 1
path_train = np.asarray(path_train)

for x in path_train:
	print k
	reg_img_path = os.path.join(data_path, x)
	input_img_path = os.path.join(data_path, test_img)
	reg_coord = np.zeros((2, 1))
	input_coord = np.zeros((2, 1))
	reg_coord = train_cores[k, :]
	input_coord = database[test_iter, :]
	k = k + 1
	input_img = cv2.imread(input_img_path, 0)
	displaced_reg = align_cores(input_coord, reg_coord, input_img_path, reg_img_path)
	displaced_reg_img = Image.fromarray(displaced_reg)
	displaced_reg_img.show()
	rot_reg_img = match_dir(num_rotations, input_img_path, displaced_reg, input_coord, k1, k2)
	rot_img = Image.fromarray(rot_reg_img)
	rot_img.show()
	top_row, bottom_row, left_col, right_col = segment(rot_reg_img, scale)
	new_reg = np.zeros((bottom_row - top_row + 1, right_col - left_col + 1))
	new_input = np.zeros((bottom_row - top_row + 1, right_col - left_col + 1))
	new_reg = rot_reg_img[top_row:bottom_row, left_col:right_col]
	new_input = input_img[top_row:bottom_row, left_col:right_col]
	reg_segmented = Image.fromarray(new_reg)
	input_segmented = Image.fromarray(new_input)
	reg_segmented.show()
	input_segmented.show()

	print new_input.shape
	print new_reg.shape


	blpoc_val = BLPOC(new_input, new_reg, k1_1, k2_1)
	if (blpoc_val > best_blpoc) :
		best_blpoc = blpoc_val
		best_img = x
	print blpoc_val
	print best_img
	break

	# print reg_img_path








