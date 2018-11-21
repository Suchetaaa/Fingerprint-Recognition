import numpy as np 
import cv2 
import math 
from PIL import Image
from BLPOC import BLPOC

num_rotations = 80
max_blpoc = -1000
initial_rot = -40
best_rot = initial_rot
displaced_image = cv2.imread("displaced_reg.png", 0)
rows,cols = displaced_image.shape
reg_img_coord = [150, 150]
input_img = cv2.imread("Input Image.tif", 0)

for x in xrange(0,num_rotations+1) :
	rot = initial_rot + x 
	transformation = cv2.getRotationMatrix2D((reg_img_coord[0], reg_img_coord[1]), rot, 1)
	rotated_img = cv2.warpAffine(displaced_image, transformation, (rows, cols))
	img = Image.fromarray(rotated_img)
	# img.save("Rotated Image")
	# img.show()

	a = BLPOC(input_img, rotated_img)
	# print a 
	if (max_blpoc < a) :
		max_blpoc = a
		best_rot = rot
		print "Best BLPOC value "
		print max_blpoc

transformation = cv2.getRotationMatrix2D((reg_img_coord[0], reg_img_coord[1]), best_rot, 1)
final_rot_img = cv2.warpAffine(displaced_image, transformation, (rows, cols))
img = Image.fromarray(final_rot_img)
img.save("Rotated Image.tif")
img.show()



	


