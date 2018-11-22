import numpy as np
import cv2 

def BLPOC(input_image, reg_image_corrected, k1, k2):
	l1 = 2*k1 + 1
	l2 = 2*k2 + 1

	print input_image.shape
	print reg_image_corrected.shape
	fft_input = np.fft.fft2(input_image)
	fft_shift_input = np.fft.fftshift(fft_input)
	fft_reg = np.fft.fft2(reg_image_corrected)
	fft_shift_reg = np.fft.fftshift(fft_reg)
	rows, cols = fft_input.shape
	print fft_shift_input.shape

	fft_input_conj = np.matrix.conjugate(fft_shift_input)
	print fft_shift_reg.shape
	print fft_input_conj.shape
	cross_phase_spec = np.divide(np.multiply(fft_shift_reg, fft_input_conj), (np.absolute(np.multiply(fft_shift_reg, fft_input_conj))))

	m1 = rows/2
	m2 = cols/2
	diff_row = m1 - k1
	diff_col = m2 - k2
	left_extreme = m2 - k2 
	right_extreme = 2*m2 - (diff_row) 
	top_extreme = m1 - k1 
	bottom_extreme = 2*m1 - (diff_col)
	eff_rows = (bottom_extreme - top_extreme) + 1
	eff_cols = (right_extreme - left_extreme) + 1
	blpoc_fft = np.zeros((eff_rows, eff_cols), dtype = complex)

	for x in range(0,eff_rows):
		for y in range(0,eff_cols):
			blpoc_fft[x][y] = cross_phase_spec[top_extreme + x, left_extreme + y]

	blpoc_fft_shifted = np.fft.ifftshift(blpoc_fft)
	blpoc_ifft = np.fft.ifft2(blpoc_fft_shifted)
	blpoc_ifft = np.absolute(blpoc_ifft)
	# print blpoc_ifft
	blpoc_peak_value = np.amax(blpoc_ifft)

	return blpoc_peak_value

# if __name__ == '__main__' :
# 	input_image = cv2.imread('Input Image.tif', 0)
# 	reg_image_corrected = cv2.imread('Registered Image.tif', 0)
# 	a = BLPOC(input_image, reg_image_corrected)
# 	print a


			


		
		



