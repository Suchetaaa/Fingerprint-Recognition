import numpy as np 

def BLPOC(input_image_corrected, reg_image_corrected):
	k1 = 100
	k2 = 200
	l1 = 2*k1 + 1
	l2 = 2*k2 + 1

	fft_input = np.fft.fft2(input_image_corrected)
	fft_shift_input = np.fft.fftshift(fft_input)
	fft_reg = np.fft.fft2(reg_image_corrected)
	fft_shift_reg = np.fft.fftshift(fft_reg)
	rows, cols = fft_input.size

	fft_input_conj = np.matrix.conjugate(fft_shift_input)
	cross_phase_spec = (fft_shift_reg.*fft_input_conj)./(np.absolute(fft_shift_reg.*fft_input_conj))

	m1 = rows/2
	m2 = cols/2
	diff_row = m1 - k1
	diff_col = m2 - k2
	left_extreme = m2 - k2 - 1
	right_extreme = 2*m2 - (diff_row) 
	top_extreme = m1 - k1 - 1
	bottom_extreme = 2*m1 - (diff_col)
	eff_rows = (bottom_extreme - top_extreme) + 1
	eff_cols = (right_extreme - left_extreme) + 1
	blpoc_fft = np.zeros((eff_rows, eff_cols))

	for x in range(0,eff_rows):
		for y in range(0,eff_cols):
			blpoc_fft[x][y] = cross_phase_spec[left_extreme + x][top_extreme + y]

	blpoc_fft_shifted = np.fft.ifftshift(blpoc_fft)
	blpoc_ifft = np.fft.ifft2(blpoc_fft_shifted)
	blpoc_peak_value = np.amax(blpoc_ifft)

	return blpoc_peak_value

			


		
		



