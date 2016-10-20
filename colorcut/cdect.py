import cv2
import numpy as np


def get_centers(img, args):
    copied = img.copy()
    contours, hierarchy = cv2.findContours(copied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 2:
        raise Exception('Error looking for contours')
    moments = [cv2.moments(c) for c in contours]
    centers = [(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) for M in moments]
    if args.debug_steps:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.imshow('contours', img)
        cv2.waitKey(0)
    return centers


def highlight(img, centers, args):
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, channels = img.shape
#  deltaX , delta Y
    delta = [centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]]
    t = (np.array([0, h])-centers[0][1])/delta[1]  # ([immagine] - y1) / deltaY
    x1, x2 = centers[0][0] + delta[0]*t  # x1 + deltaX*t
    for c in centers:
        cv2.circle(img, c, 3, (0, 255, 255), -1)
    cv2.line(img, (x1, 0), (x2, h), (0, 255, 0), 1)
    cv2.line(img, centers[0], centers[1], (0, 255, 0), 1)
    return img


def main(args=None):
    if args is None:
        args = get_args()
    img = cv2.imread(args.image)
    height, width, channels = img.shape
    img = img[args.cut_top:(height - args.cut_bottom),
              args.cut_left:(width - args.cut_right)]
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if args.debug_steps:
        cv2.imshow('original (hsv)', cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.blur is not None:
        cimg = cv2.GaussianBlur(cimg, (args.blur, args.blur), 0)
        if args.debug_steps:
            cv2.imshow('after blurring', cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    lower = np.array([args.hue - args.hue_radius,
                      args.min_saturation, args.min_value])
    upper = np.array([args.hue + args.hue_radius, 255, args.max_value])
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(cimg, lower, upper)
    cimg = cv2.bitwise_and(cimg, cimg, mask=mask)
    if args.debug_steps:
        cv2.imshow('after HSV threshold',
                   cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR),
                        cv2.COLOR_BGR2GRAY)
    if args.debug_steps:
        cv2.imshow('to gray', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    centers = get_centers(gray, args)
    if args.coordinates is not None:
        for c in centers:
            p1, p2 = c
            args.coordinates.write('%d\t%d\n' % (p1+int(args.cut_left),
                                                 p2+int(args.cut_top)))

    if args.show_result:
        highlight(img, centers, args)
        cv2.imshow('detected circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_parser():
    import argparse
    p = argparse.ArgumentParser(
        description='recognize special points in an image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('image')

    proc = p.add_argument_group('preprocessing')
    proc.add_argument('--cut-left', metavar='PX', default='0')
    proc.add_argument('--cut-right', metavar='PX', default='0')
    proc.add_argument('--cut-top', metavar='PX', default='0')
    proc.add_argument('--cut-bottom', metavar='PX', default='0')
    proc.add_argument('--blur', metavar='RADIUS', type=int,
                      help='It must be an odd number. If omitted, no blurring '
                      'will occur')

    col = p.add_argument_group('color filtering',
                               description="HSV is on the 0-255 range;"
                               " 120 is blue, 180 is red")
    col.add_argument('--hue', metavar='HUECENTER', default=60, type=int,
                     help="Default: 60 (green)")
    col.add_argument('--hue-radius', metavar='HUERADIUS', default=10, type=int,
                     help=u"Will accept colors with "
                     u"hue=HUECENTER\u00B1HUERADIUS".encode('utf-8'))
    col.add_argument('--min-saturation', metavar='S', default=150, type=int,
                     help="Minimum accepted saturation")
    col.add_argument('--min-value', metavar='V', default=150, type=int,
                     help="Minimum accepted value")
    col.add_argument('--max-value', metavar='V', default=255, type=int,
                     help="Maximum accepted value")

    det = p.add_argument_group('circle detection')
    det.add_argument('--param2', default=20, type=int,
                     help='If it is lower, it will have more false positives')
    det.add_argument('--min-dist', metavar='PX', default='100',
                     help='Minimum distance between circles')
    det.add_argument('--min-radius', metavar='PX', default='0')
    det.add_argument('--max-radius', metavar='PX', default='0')

    p.add_argument('--coordinates', metavar='FILE',
                   type=argparse.FileType('w'),
                   help='Write found coordinates to FILE')
    p.add_argument('--show-result', action='store_true', default=False,
                   help='Show image with found circles')
    p.add_argument('--debug-steps', action='store_true', default=False,
                   help='Display image at each step')

    return p


def get_args():
    args = get_parser().parse_args()
    img = cv2.imread(args.image)
    height, width, channels = img.shape
    for attr in ('min_dist', 'min_radius', 'max_radius',
                 'cut_left', 'cut_right', 'cut_top', 'cut_bottom'):
        value = getattr(args, attr)
        if value.endswith('%'):
            value = width * float(value[:-1]) / 100.0
        else:
            value = float(value)
        setattr(args, attr, int(value))
    return args

if __name__ == '__main__':
    main()
