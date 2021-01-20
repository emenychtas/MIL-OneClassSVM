from glob import glob
import cv2 as cv
import os

for fn in glob('/media/sf_Shared_Folder/ucsb_bcc/*'):
	original = cv.imread(fn)
	
	# Convert from cv2 standard of BGR to our convention of RGB.
	resized = cv.resize(original, (400,342))

	# Perform stain normalization
	out_fn = os.path.basename(os.path.splitext(fn)[0])
	cv.imwrite('output2/'+out_fn+'.png',resized,[int(cv.IMWRITE_PNG_COMPRESSION),0])
