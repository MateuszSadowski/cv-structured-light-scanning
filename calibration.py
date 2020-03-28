import cv2
import numpy as np

PATTERN_WIDTH = 22
PATTERN_HEIGHT = 13
SQUARE_SIZE = 15#mm

cornerPoints = np.mgrid[0:PATTERN_WIDTH*SQUARE_SIZE:SQUARE_SIZE,0:PATTERN_HEIGHT*SQUARE_SIZE:SQUARE_SIZE].T.reshape(-1,2)

img = cv2.imread('../calib_images_rect/frame0_0.png')
cv2.imshow('tatus', img)
cv2.waitKey()

retval, corners = cv2.findChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT))
if retval == 0:
    print('Something went wrong with finding the chessboard corners')
else:
    newImg = cv2.drawChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT), corners, retval)
    cv2.imshow('Image with corners detected', newImg)
    cv2.waitKey()

cv2.destroyAllWindows()
