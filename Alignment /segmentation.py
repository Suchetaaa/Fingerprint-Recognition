import numpy as np 
import os 
import cv2 
from PIL import Image

def segment(rotated_img, scale):
	rotated_img = (rotated_img - np.amin(rotated_img))/(np.amax(rotated_img) - np.amin(rotated_img))
	rotated_row_sum = rotated_img.sum(axis = 1)
	rotated_col_sum = rotated_img.sum(axis = 0)
	rotated_row_sum = np.reshape(rotated_row_sum, (rotated_row_sum.shape[0], 1))
	rotated_col_sum = np.reshape(rotated_col_sum, (rotated_col_sum.shape[0], 1))
	print rotated_col_sum.shape
	print rotated_row_sum.shape
	rows, cols = rotated_img.shape
	top_row = 0
	bottom_row = 0
	left_col = 0
	right_col = 0
	for x in xrange(0,rows):
		if (rotated_row_sum[x][0] > cols*scale) :
			top_row = x
			break
	for x in xrange(1, rows-1):
		if (rotated_row_sum[rows-x][0] > cols*scale) :
			bottom_row = rows - x 
			break
	for x in xrange(0,cols):
		if (rotated_col_sum[x][0] > rows*scale):
			left_col = x
			break
	for x in xrange(1,cols-1):
		if (rotated_col_sum[cols-x][0] > rows*scale) :
			right_col = cols-x
			break

	return top_row, bottom_row, left_col, right_col

def main():
	rotated_img = "Rotated Image.tif"
	rotated_img = cv2.imread(rotated_img, 0)
	top_row, bottom_row, left_col, right_col = segment(rotated_img)
	new_image = np.zeros((bottom_row-top_row+1, right_col-left_col+1))
	new_image = rotated_img[top_row:bottom_row, left_col:right_col]
	new_image = Image.fromarray(new_image)
	new_image.show()

if __name__ == '__main__':
	main()
