import numpy as np 
from PIL import Image 
import cv2
import os 
import scipy as sp
import math

def gray2binfun(input_image) :
	img = cv2.imread(input_image, 0)
	dummy, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# img_return = Image.fromarray(img_threshold)
	return img_threshold

def Normalisation (input_image_path):
	M0 = 74.98999708879379
	V0 = 2011395.439145634

	test_image = cv2.imread(input_image_path, 0)
	test_image = cv2.GaussianBlur(test_image, (3,3), 0)

	Mean = np.mean(test_image)
	V_i = np.var(test_image)

	output_image = np.zeros(test_image.shape)
	Size = test_image.shape
	# print test_image.shape
	print(output_image.shape)
	print(test_image.shape)
	for i,j in np.argwhere(test_image > Mean):
		output_image[i, j] = M0 + math.sqrt(V0*math.pow((test_image[i, j]-Mean),2)/V_i)

	for i,j in np.argwhere(test_image <= Mean):
		output_image[i, j] = M0 - math.sqrt(V0*math.pow((test_image[i, j]-Mean),2)/V_i)

	output_image = Image.fromarray(output_image)
	return output_image

def main() :
	lightning_conditions = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
	dataset_path = "/Users/suchetaaa/Desktop/Academics @IITB/Semester V/DIP/Project/Fingerprint-Recognition/FVC2002"
	binary_lighting = ["DB1_B_B", "DB2_B_B", "DB3_B_B", "DB4_B_B"]

	for x in range(0, len(lightning_conditions)):
		path_lighting_condition = os.path.join(dataset_path, lightning_conditions[x])
		image_files = os.listdir(path_lighting_condition)

		# binary_lighting_path = os.path.join(dataset_path, binary_lighting)

		for y in range(1,len(image_files)):
			input_image_path = os.path.join(path_lighting_condition, image_files[y])
			normalized_image = Normalisation(input_image_path)
			binary_image = gray2binfun(normalized_image)
			# image_path = os.path.join(binary_lighting_path, image_files[y])
			# binary_image.save(image_path)
			binary_image.show()

if __name__ == '__main__':
	dataset_path = "/Users/suchetaaa/Desktop/Academics @IITB/Semester V/DIP/Project/Fingerprint-Recognition/FVC2002/DB3_B/104_8.tif"
	normalized_img = Normalisation(dataset_path)
	normalized_img.show()
	input_image = cv2.imread(dataset_path)
	otsu = gray2binfun(dataset_path)
	otsu = Image.fromarray(otsu)
	otsu.show()


	# main()

	
