import numpy as np 
import cv2 

input_img = cv2.imread("Input Image.tif", 0)
reg_rotated_img = cv2.imread("Rotated Image.tif", 0)
reg_img_arr = np.asarray(reg_rotated_img);

rows, cols = reg_rotated_img.shape

# Computation of POC function
rotated_img_fft = np.fft.fft2(reg_rotated_img)
rotated_img_fft_shift = np.fft.fftshift(rotated_img_fft)

input_img_fft = np.fft.fft2(input_img)
input_img_fft_shift = np.fft.fftshift(input_img_fft)

input_conj_fft = np.matrix.conjugate(input_img_fft_shift)

cross_phase_spec = np.divide(np.multiply(input_conj_fft, rotated_img_fft_shift), np.absolute(np.multiply(input_conj_fft, rotated_img_fft_shift)))
shifted_cross_phase_spec = np.fft.ifftshift(cross_phase_spec)
poc = np.fft.ifft2(shifted_cross_phase_spec)
poc = np.absolute(poc)

i, j = np.unravel_index(poc.argmax(), poc.shape)
disp_x = j - cols/2
disp_y = i - rows/2
print "The translational displacement and max value are  "
print disp_x 
print disp_y 
p = np.amax(poc)
print p

transformation = np.float32([[1, 0, disp_x], [0, 1, disp_y]])
displaced_input = cv2.warpAffine(reg_rotated_img, transformation, (rows, cols))




