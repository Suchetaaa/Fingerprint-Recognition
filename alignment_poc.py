import numpy as np 
import cv2 

Input images = rotated_img, ref_img

rotated_img = cv2.imread("Rotated Image.tiff")
reg_img = cv2.imread("Registered Image.tiff")
reg_img_arr = np.asarray(reg_img);

rows, cols = reg_img.size

# Computation of POC function
rotated_img_fft = np.fft.fft2(rotated_img)
rotated_img_fft_shift = np.fft.fftshift(rotated_img_fft)

reg_img_fft = np.fft.fft2(reg_img)
reg_img_fft_shift = np.fft.fftshift(reg_img_fft)

input_conj_fft = np.matrix.conjugate(rotated_img_fft_shift)

cross_phase_spec = (input_conj_fft.*reg_img_fft_shift)./(np.absolute(input_conj_fft.*reg_img_fft_shift))
shifted_cross_phase_spec = np.fft.ifftshift(cross_phase_spec)
poc = np.fft.ifft2(shifted_cross_phase_spec)

i, j = np.unravel_index(a.argmax(), a.shape)
disp_x = j - cols/2
disp_y = i - rows/2
transformation = np.float32([[1, 0, disp_x], [0, 1, disp_y]])
displaced_input = cv2.warAffine(reg_img, transformation, (rows, cols))




