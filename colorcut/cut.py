from __future__ import print_function
import os.path,csv
from collections import namedtuple
import numpy as np
import cv2

Point = namedtuple('Point', ('x', 'y'))


def show_exit(im):
    cv2.imshow('test', im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def mask_import(mask_path):
    mask=np.load(mask_path+"/mask.npy")
#    sx_mask=np.load(mask_path+"/sx.npy")
    return mask
        
def cut_by_mask(mask,image):
    only_left = cv2.bitwise_and(image, image, mask = mask[0,:,:])
    only_right= cv2.bitwise_and(image, image, mask = mask[1,:,:])
    height, width, depth = image.shape
    only_left = crop(only_left, 0, width)
    only_right = crop(only_right,0, width)
    return only_left, only_right
    
def split_image(coord_buf, img):
    data = coord_buf.read()
    p1 = Point(*map(int, data.split('\n')[0].split('\t')))
    p2 = Point(*map(int, data.split('\n')[1].split('\t')))
    height, width, depth = img.shape
    # note that axes are inverted: x = alpha * y
    alpha, b, c = get_cut_params(p1, p2, height)

    left_poly = [(0, 0), (b, 0), (c, height), (0, height)]
    right_poly = [(b, 0), (width, 0), (width, height), (c, height)]
    only_right = blackened(img, left_poly)
    only_left = blackened(img, right_poly)

    only_left = crop(only_left, 0, max(b, c))
    only_right = crop(only_right, min(b, c), width)
    return only_left, only_right

def crop_by_countours(image,side,mask_folder):
    csvfile=open(mask_folder+"/countours"+side+'.csv', 'rb')
    my_reader = csv.reader(csvfile, delimiter='\t')
    my_list=list(my_reader)[0]
    #for row in my_reader:
    print (my_list)
    cv2.imshow('mask',image)
    cv2.waitKey()
    image= image[int(my_list[0]):int(my_list[1]),\
    int(my_list[2]):int(my_list[3])]
    print(image.shape)
    return image
    
def crop(img, from_, to):
    '''crop by width'''
    return img[0:img.shape[0], from_:to]


def  get_cut_params(p1, p2, img_height):
    alpha = float(p1.x-p2.x)/(p1.y-p2.y)
    b = p1.x - alpha * p1.y
    c = alpha * img_height + b
    print(p1, p2)
    print(alpha, b, c)
    return (alpha, b, c)


def blackened(im, poly_to_blacken):
    bg = [0] * 3
    mask = np.full(im.shape, 255, dtype=np.uint8)
    roi = np.array([poly_to_blacken], dtype=np.int32)
    cv2.fillPoly(mask, roi, bg)
    return cv2.bitwise_and(im, mask)


def get_parser():
    import argparse
    p = argparse.ArgumentParser('taglia taglia')
    #p.add_argument('cut_coordinates_filename', type=argparse.FileType('r'))
    p.add_argument('img_filename')
    p.add_argument('--outfile')
    p.add_argument('--show', action='store_true', default=False)
    p.add_argument("-m", "--mask", help = "path to the mask")
    p.add_argument("-c", "--color", help = "path to the mask")
    return p


def main():
    args = get_parser().parse_args()
    img = cv2.imread(args.img_filename)
    #cut by color or spot
    if not args.color:
        left, right = split_image(args.cut_coordinates_filename, img)
    else:
        mask=mask_import(args.mask)
        left, right = cut_by_mask(mask,img)
        left= crop_by_countours(left,"right",args.mask)
        right= crop_by_countours(right,"left",args.mask)
        
    if args.outfile is not None:
        l_fname = '%s.L%s' % os.path.splitext(args.outfile)
        r_fname = '%s.R%s' % os.path.splitext(args.outfile)
        cv2.imwrite(l_fname, left)
        cv2.imwrite(r_fname, right)
    if args.show is True:
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
