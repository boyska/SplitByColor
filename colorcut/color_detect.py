# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
def get_parser()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image")
    ap.add_argument("-m", "--mask", help = "path to the mask")
    return vars(ap.parse_args())
 
def set_boundaries():
'''
at moment you need to insert by hand right colors, 
it should be done by terminal
'''
    boundaries = [
	([00, 00, 00], [255,50,50], "dx"),
	([00, 00, 00], [50,50,255],"sx")
    ]
    return boundaries


# loop over the boundaries
#image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
def masker():
    boundaries=set_boundaries()
    image = cv2.imread(args["image"])
    for (lower, upper,side) in boundaries:
	    # create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	    # find the colors within the specified boundaries and apply
	    # the mask
	mask = cv2.inRange(image, lower, upper)
	np.save(args['mask']+"/"+side+".npy",arr=mask)
	output = cv2.bitwise_and(image, image, mask = mask)
     
	    # show the images
	#output = cv2.resize(output, (0,0), fx=0.5, fy=0.5)
	#cv2.imshow("images", np.hstack([image, output]))
	#cv2.waitKey(0)
    return mask       


