import numpy as np 
import cv2 
from PIL import Image

def align_cores(input_core_coord, reg_img_coord, input_img_path, reg_img_path):

	input_img = cv2.imread(input_img_path, 0)
	reg_img = cv2.imread(reg_img_path, 0)
	# reg_img_arr = np.asarray(reg_img)
	# input_arr = np.asarray(input_img)

	disp_x =  input_core_coord[0] - reg_img_coord[0] 
	disp_y = input_core_coord[1] - reg_img_coord[1] 
	rows, cols = reg_img.shape
	transformation = np.float32([[1, 0, disp_x], [0, 1, disp_y]])
	displaced_reg = np.zeros((rows, cols))
	displaced_reg = cv2.warpAffine(reg_img, transformation, (cols, rows))
	print displaced_reg.shape
	img = Image.fromarray(displaced_reg)
	img.show()
	print "alignment cores done"
	return displaced_reg
	





			
	 	
