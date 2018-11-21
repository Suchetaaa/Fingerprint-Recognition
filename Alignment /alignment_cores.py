import numpy as np 
import cv2 
from PIL import Image

image_width = 300
image_height = 300

input_img = cv2.imread("Input Image.tif", 0)
reg_img = cv2.imread("Registered Image.tif", 0)
reg_img_arr = np.asarray(reg_img)
input_arr = np.asarray(input_img)

input_core_bool = 1
reg_core_bool = 1
input_core_coord = [200, 205]
reg_img_coord = [200, 200]

if (input_core_bool == 1 and reg_core_bool == 1) :
	disp_x =  input_core_coord[0] - reg_img_coord[0] 
	disp_y = input_core_coord[1] - reg_img_coord[1] 
	rows, cols = reg_img.shape
	transformation = np.float32([[1, 0, disp_x], [0, 1, disp_y]])
	displaced_reg = np.zeros((rows, cols))
	displaced_reg = cv2.warpAffine(reg_img, transformation, (rows, cols))
	img = Image.fromarray(displaced_reg)
	img.save('displaced_reg.png')
	img.show()
	





			
	 	
