import numpy as np 
import cv2 
import math 
from PIL import Image
from BLPOC import BLPOC
from remap_relabel import remap_relabel

def match_dir(num_rotations, input_img_path, displaced_reg, input_core_coord, k1, k2):
	
	max_blpoc = -1000
	initial_rot = -40
	best_rot = initial_rot
	rows,cols = displaced_reg.shape
	input_img = cv2.imread(input_img_path, 0)
	print input_img.shape
	print displaced_reg.shape

	for x in xrange(0,num_rotations+1) :
		rot = initial_rot + x 
		transformation = cv2.getRotationMatrix2D((input_core_coord[0], input_core_coord[1]), rot, 1)
		rotated_img = cv2.warpAffine(displaced_reg, transformation, (cols, rows))
		img = Image.fromarray(rotated_img)
		print rotated_img.shape
		# img.show()

		a = BLPOC(input_img, rotated_img, k1, k2)
	# 	# print a 
		if (max_blpoc < a) :
			max_blpoc = a
			best_rot = rot
			print "Best BLPOC value "
			print max_blpoc

	transformation = cv2.getRotationMatrix2D((input_core_coord[0], input_core_coord[1]), best_rot, 1)
	final_rot_img = cv2.warpAffine(displaced_reg, transformation, (cols, rows))
	print final_rot_img.shape
	img = Image.fromarray(final_rot_img)
	# img.save("Rotated Image.tif")
	img.show()
	print "direction estimation done"
	return final_rot_img

def main() :
	input_img_path = "Input Image.tif"
	reg_img_path = "Registered Image.tif"
	match_dir(20, input_img_path, reg_img_path)

if __name__ == '__main__':
	main()



	


