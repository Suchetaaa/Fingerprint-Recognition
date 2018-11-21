import cv2
import numpy as np
from math import pi

# import BLPOC
# import Normalisation
# import Singular_index
# import alignment_cores
# import alignmnet_poc
import coherenceOrientationField
import direction_field
# import direction_matching
# import poincare_index

##Read the image
I = cv2.imread("/home/neharika/Desktop/DIP/Project/dummy4.tif");

foreground_mask = coherenceOrientationField.coherenceOrientationField(I)

# cv2.imshow('foreground_mask', foreground_mask)
# cv2.waitKey(0)

