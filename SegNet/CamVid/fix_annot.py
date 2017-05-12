
# https://github.com/alexgkendall/caffe-segnet/issues/101

#purpose is to change pixel range from 0 to 1

import os
import cv2
import numpy as np

for filename in os.listdir('.'):
	img = cv2.imread(filename, 0)
	a_img = np.array(img, np.double)
	normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
	cv2.imwrite(filename, normalized)
	#testing
	if False:
		for i in range(len(img)):
			for j in range(len(img[0])):
				print img[i,j], normalized[i,j]
