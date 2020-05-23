import cv2
import numpy as np
from colormath.color_objects import sRGBColor
from colormath.color_diff import delta_e_cie1976
import csv

import helper

LOW_INTENSITY_CUTOFF = 0.1 * 255
HIGH_INTENSITY_CUTOFF = 0.9 * 255

def getDisplacement(mask1, mask2):
    if len(mask1) != len(mask2):
        print("Masks are not the same size")
        return -1
    
    cam1HeightVal = 0
    cam2HeightVal = 0

    for i in range(len(mask1)):
        if cam1HeightVal != 0 and cam2HeightVal != 0:
            break
        if mask1[i].max() == 1 and cam1HeightVal == 0:
            cam1HeightVal = i
        if mask2[i].max() == 1 and cam2HeightVal == 0:
            cam2HeightVal = i
    
    return cam1HeightVal - cam2HeightVal

def getMask(image):
    print('getMask')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageSize = gray.shape[::-1]
    mask = np.ones((imageSize[1], imageSize[0]))
    for x in range(imageSize[1]):
        for y in range(imageSize[0]):
            intensity = gray[x, y]
            if (intensity < LOW_INTENSITY_CUTOFF) or (HIGH_INTENSITY_CUTOFF < intensity):
                mask[x, y] = 0
    return mask

def getColorDifference(color1, color2):
    # rgb1 = sRGBColor(color1[0], color1[1], color1[2])
    # rgb2 = sRGBColor(color2[0], color2[1], color2[2])

    # TODO: convert to int
    return (color1[0] - color2[0])*(color1[0] - color2[0]) + (color1[1] - color2[1])*(color1[1] - color2[1]) + (color1[2] - color2[2])*(color1[2] - color2[2])

def registerColor(img1, img2, mask1, mask2, d=0):
    imgHeight = len(img1)
    imgWidth = len(img1[0]) if imgHeight > 0 else 0
    q1 = []
    q2 = []
    
    # TODO: pass start row from getDisplacement here
    # TODO: print out row, col to know progress 
    for row in range(imgHeight):
        if row - d < 0:
            continue
        elif row - d >= imgHeight:
            break
        else:
            row2 = row - d
        for col in range(imgWidth):
            if mask1[row][col] != 0:
                color1 = img1[row][col]
                minDiff = float("inf")
                minPixel = [-1, -1]
                for col2 in range(imgHeight):
                    if mask2[row2][col2] != 0:
                        color2 = img2[row2][col2]
                        diff = getColorDifference(color1, color2)
                        if diff < minDiff:
                            minDiff = diff
                            minPixel = [row2, col2] # TODO: should it be row2 or row for triangulation?
                q1.append([row, col])
                q2.append(minPixel)
    return q1, q2

# TODO: import rectified images in color
cam1 = cv2.imread("../Other objects/Cam 1 (left)/rect-cup-color.png")
cam2 = cv2.imread("../Other objects/Cam 2 (right)/rect-cup-color.png")
# TODO: look for matches of the same color in the corresponding part of the image
mask1 = getMask(cam1)
mask2 = getMask(cam2)

maskedCam1 = helper.maskImage(cam1, mask1)
# maskedCam1 = cv2.bitwise_and(cam1, cam1, mask=mask1)
maskedCam2 = helper.maskImage(cam2, mask2)
# maskedCam2 = cv2.bitwise_and(cam2, cam2, mask=mask2)

cv2.imshow('cam1', maskedCam1)
cv2.waitKey()
cv2.imshow('cam2', maskedCam2)
cv2.waitKey()

d = getDisplacement(mask1, mask2)

q1, q2 = registerColor(cam1, cam2, mask1, mask2, d)

fName = 'q1q2.csv'
with open(fName, 'w+', newline='') as file:
    writer = csv.writer(file, quoting = csv.QUOTE_NONNUMERIC)
    for i in range(len(q1)):
        writer.writerow([q1[i][0], q1[i][1], "", q2[i][0], q2[i][1]])