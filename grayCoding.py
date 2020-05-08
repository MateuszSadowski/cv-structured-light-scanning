import cv2
import numpy as np

import helper

# TODO: import images in gray scale
# TODO: rectify images
# TODO: do thresholding ?
# TODO: create mask
# TODO: obtain code for each point on the object
# vertical plane index = 2^(n-i) + (binary code of the point) * 2^(n-i+1),
# where: n - number of all patterns, i - index of pattern
    # TODO: figure out code for each image plane ??
# TODO: match each point in a row to a corresponding point in the same row in the other camera by code (maybe throw out the last row?)
# TODO: triangulate