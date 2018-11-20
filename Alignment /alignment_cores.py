import numpy as np 
import cv2 

image_width = 300
image_height = 300

input_img = cv2.imread("Input Image.tiff")
reg_img = cv2.imread("Registered Image.tiff")
reg_img_arr = np.asarray(reg_img);

# input_core_bool
# reg_core_bool 
# input_core_coord
# reg_img_coord

if (input_core_bool == 1 and reg_core_bool == 1) :
	disp_x =  input_core_coord[0] - reg_img_coord[0] 
	disp_y = input_core_coord[1] - reg_img_coord[1] 
	rows, cols = reg_img.size
	transformation = np.float32([[1, 0, disp_x], [0, 1, disp_y]])
	displaced_input = cv2.warAffine(reg_img, transformation, (rows, cols))
	
	np.save(displaced_image, displaced_image)





			
	 	
