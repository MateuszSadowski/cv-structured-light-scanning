import cv2
import numpy as np
import glob

PATTERN_WIDTH = 22
PATTERN_HEIGHT = 13
SQUARE_SIZE = 15.0#mm
PATH_TO_CALIBRATION_IMAGES = '../calib_images_rect/'

cornerCoords = np.zeros((PATTERN_HEIGHT*PATTERN_WIDTH, 3), np.float32)
cornerCoords[:,:2] = np.mgrid[0:PATTERN_WIDTH*SQUARE_SIZE:SQUARE_SIZE,0:PATTERN_HEIGHT*SQUARE_SIZE:SQUARE_SIZE].T.reshape(-1,2) # Used to fill worldCoords
imageSize = None

imagesCam1 = glob.glob(PATH_TO_CALIBRATION_IMAGES + 'frame0_*.png')
imagesCam2 = glob.glob(PATH_TO_CALIBRATION_IMAGES + 'frame1_*.png')

objectPoints = [] # 3D coords of checkerboard corners, we assume that it's planar and placed in the origin and that we rotate the camera when taking pictures
imagePointsCam1 = [] # 2D coords of projected points in respect to camera 1
imagePointsCam2 = [] # 2D coords of projected points in respect to camera 1

# === Find checkerboard corners ===
for image in imagesCam1:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if imageSize == None:
        imageSize = img.shape[::-1]

    retval, corners = cv2.findChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT))
    if retval == True:
        # newImg = cv2.drawChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT), corners, retval)
        # cv2.imshow('Image with corners detected', newImg)
        # cv2.waitKey()
        objectPoints.append(cornerCoords)
        imagePointsCam1.append(corners)

for image in imagesCam2:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    retval, corners = cv2.findChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT))
    if retval == True:
        # newImg = cv2.drawChessboardCorners(img, (PATTERN_WIDTH, PATTERN_HEIGHT), corners, retval)
        # cv2.imshow('Image with corners detected', newImg)
        # cv2.waitKey()
        imagePointsCam2.append(corners)

# === Calibrate each camera separately for higher accuracy ===
retval, cameraMatrix1, distCoeffs1, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePointsCam1, imageSize, None, None)
retval, cameraMatrix2, distCoeffs2, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePointsCam2, imageSize, None, None)

# === Stereo calibrate cameras ===
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, imagePointsCam1, imagePointsCam2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize)

cv2.destroyAllWindows()