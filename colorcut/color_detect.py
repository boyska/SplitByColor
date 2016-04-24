# import the necessary packages
import numpy as np
import cv2
import csv
from PIL import Image

# construct the argument parse and parse the arguments
def get_parser():
    import argparse
    ap = argparse.ArgumentParser("find_color")
    ap.add_argument("-i", "--image", help = "path to the image")
    ap.add_argument("-m", "--mask", help = "path to the mask")
    ap.add_argument('--blur', metavar='RADIUS', type=int,
                      help='It must be an odd number. If omitted, no blurring '
                      'will occur')
    ap.add_argument('--debug-steps', action='store_true', default=False,
                   help='Display image at each step')
    return ap
 
def set_boundaries():
	'''at moment you need to insert by hand right colors, it should be done by terminal
	'''
    	boundaries = [
	([70, 110, 75], [180,255,205], "sx"),
	([160, 60, 40], [255,175,205],"dx")
    	]	
	return boundaries

def import_boundaries():
    '''write detection parameters in a csv file
    where, first row is left page, and first 3 column the minimum values
    '''
    colors=np.genfromtxt(args.mask+'/colors.csv', dtype=int)
    left=list(colors[0])
    right=list(colors[1])
    return [(left[0:3], left[3:6],'sx'),(right[0:3],right[3:6],'rx')]
    
# loop over the boundaries
def masker(image,printout=False):
    boundaries=import_boundaries()
    print boundaries
    mask=[]
    for (lower, upper, side) in boundaries:
	    # create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	    # find the colors within the specified boundaries and apply
	    # the mask
	mask.append(cv2.inRange(image, lower, upper))
	    # show the images
	if printout:
		output = cv2.bitwise_and(image, image, mask = mask)
		cv2.imshow("images", np.hstack([image, output]))
		cv2.waitKey(0)
    return np.stack([mask[0],mask[1]])   

def find_countours(mask,side):
    mask=mask<255
    first_row=np.min(np.nonzero(mask)[0])
    last_row=np.max(np.nonzero(mask)[0])
    first_column=np.min(np.nonzero(mask)[1])
    last_column=np.max(np.nonzero(mask)[1])
    with open('mask/countours.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter='\t')
	writer.writerow([first_row, last_row, first_column, last_column])
   
def show_exit(im):
    cv2.imshow('test', im)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
    args=get_parser().parse_args()
    image = cv2.imread(args.image)

    if args.blur is not None:
        image = cv2.GaussianBlur(image, (args.blur, args.blur), 0)
        if args.debug_steps:
            cv2.imshow('after blurring', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    mask=masker(image)
    find_countours(mask[0],"left")
    find_countours(mask[1],"right")
    if args.debug_steps:
	show_exit(mask[0])
	show_exit(mask[1])
	
    np.save(args.mask+"/mask.npy",arr=mask)
    print("colordetect has correctly runned")
    print("some info: red page on the left and left as first mask")
