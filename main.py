import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

PATH_TO_IMAGES = '../angel_rect/'
LOW_INTENSITY_CUTOFF = 0.1 * 255
HIGH_INTENSITY_CUTOFF = 0.9 * 255

def loadImagesInGrayscale(images):
    gray = []
    for image in images:
        gray.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
    return gray

def getMask(image):
    imageSize = image.shape[::-1]
    mask = np.ones((imageSize[1], imageSize[0]))
    for x in range(imageSize[1]):
        for y in range(imageSize[0]):
            intensity = image[x, y]
            if (intensity < LOW_INTENSITY_CUTOFF) or (HIGH_INTENSITY_CUTOFF < intensity):
                mask[x, y] = 0
    return mask

imagesRefCam1Fn = glob.glob(PATH_TO_IMAGES + 'frames0_[0-1].png')
imagesPrimaryPhaseCam1Fn = glob.glob(PATH_TO_IMAGES + 'frames0_[2-9].png')
imagesSecondaryPhaseCam1Fn = glob.glob(PATH_TO_IMAGES + 'frames0_[10-17].png')

imagesRefCam2Fn = glob.glob(PATH_TO_IMAGES + 'frames1_[0-1].png')
imagesPrimaryPhaseCam2Fn = glob.glob(PATH_TO_IMAGES + 'frames1_[2-9].png')
imagesSecondaryPhaseCam2Fn = glob.glob(PATH_TO_IMAGES + 'frames1_[10-17].png')

imagesRefCam1 = loadImagesInGrayscale(imagesRefCam1Fn)
imagesPrimaryPhaseCam1 = loadImagesInGrayscale(imagesPrimaryPhaseCam1Fn)
imagesSecondaryPhaseCam1 = loadImagesInGrayscale(imagesSecondaryPhaseCam1Fn)

imagesRefCam2 = loadImagesInGrayscale(imagesRefCam2Fn)
imagesPrimaryPhaseCam2 = loadImagesInGrayscale(imagesPrimaryPhaseCam2Fn)
imagesSecondaryPhaseCam2 = loadImagesInGrayscale(imagesSecondaryPhaseCam2Fn)

imageMaskCam1 = getMask(imagesRefCam1[0])
imageMaskCam2 = getMask(imagesRefCam2[0])
cv2.imshow('cam1', imageMaskCam1)
cv2.waitKey()
cv2.imshow('cam2', imageMaskCam2)
cv2.waitKey()

cv2.destroyAllWindows()