import numpy as np 
from PIL import Image 
import cv2
import os 

path = "/Users/suchetaaa/Desktop/Academics @IITB/Semester V/DIP/Project/Fingerprint-Recognition/Binary conversion/Binary"
images = os.listdir("/Users/suchetaaa/Desktop/Academics @IITB/Semester V/DIP/Project/Fingerprint-Recognition/Binary conversion/Binary")
print len(images)
print images
threshold = 128
for x in xrange(0,len(images)):
	print images[x]
	img_path = os.path.join(path, images[x])
	img = cv2.imread(img_path, 0)
	blur = cv2.GaussianBlur(img,(3,3),0)
	dummy, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# img_arr = np.asarray(img)
	img_show = Image.fromarray(img_threshold)
	img_show.show()

	
