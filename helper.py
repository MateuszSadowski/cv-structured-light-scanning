import cv2
import numpy as np

def loadImagesInGrayscale(images):
    print('loadImagesInGrayscale')
    gray = []
    for image in images:
        gray.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
    return gray

def maskImage(img, mask):
    print('maskImage')
    imgHeight = len(img)
    imgWidth = len(img[0]) if imgHeight > 0 else 0
    maskHeight = len(mask)
    maskWidth = len(mask[0]) if maskHeight > 0 else 0

    if imgHeight != maskHeight or imgWidth != maskWidth:
        print('Image and mask have to be of the same size')
        return -1

    for x in range(imgHeight):
        for y in range (imgWidth):
            img[x][y] = img[x][y] * mask[x][y]
    
    return img