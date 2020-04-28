import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os.path

PATH_TO_IMAGES = '../angel_rect/'
PATH_TO_IMAGES_SEC_CAM1 = '../angel_rect/secondary_cam1/'
PATH_TO_IMAGES_SEC_CAM2 = '../angel_rect/secondary_cam2/'
LOW_INTENSITY_CUTOFF = 0.1 * 255
HIGH_INTENSITY_CUTOFF = 0.9 * 255

def loadImagesInGrayscale(images):
    print('loadImagesInGrayscale')
    gray = []
    for image in images:
        gray.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
    return gray

def getMask(image):
    print('getMask')
    imageSize = image.shape[::-1]
    mask = np.ones((imageSize[1], imageSize[0]))
    for x in range(imageSize[1]):
        for y in range(imageSize[0]):
            intensity = image[x, y]
            if (intensity < LOW_INTENSITY_CUTOFF) or (HIGH_INTENSITY_CUTOFF < intensity):
                mask[x, y] = 0
    return mask

def decodePhase(primary, secondary):
    print('decodePhase')
    numOfImages = len(primary)
    height = len(primary[0]) if numOfImages > 0 else 0
    width = len(primary[0][0]) if height > 0 else 0
    heterodynePhase = np.zeros((height, width))
    wrappedPrimaryImg = np.zeros((height, width))

    # 2D array of lists
    tmpPhasePrimary = [
        [[] for y in range( width )]
            for x in range( height )
    ]

    tmpPhaseSecondary = [
        [[] for y in range( width )]
            for x in range( height )
    ] 

    # List in [x, y] holds the temporal intensity of pixel [x, y] between images
    for x in range(height):
        for y in range(width):
            for n in range(numOfImages):
                tmpPhasePrimary[x][y].append(primary[n][x][y])
                tmpPhaseSecondary[x][y].append(secondary[n][x][y])
    fftPrimary = np.fft.fft(tmpPhasePrimary) # FFT across time dimension
    fftSecondary = np.fft.fft(tmpPhaseSecondary) # FFT across time dimension
    for x in range(height):
        for y in range(width):
            wrappedPrimary = np.angle(fftPrimary[x][y][1]) # Get phase from the second channel
            wrappedSecondary = np.angle(fftSecondary[x][y][1]) # Get phase from the second channel

            wrappedPrimaryImg[x][y] = wrappedPrimary
            heterodynePhase[x][y] = (wrappedPrimary - wrappedSecondary) % (2 * np.pi)

    return heterodynePhase

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

def registerContinous(code1, code2, mask1, mask2):
    imgWidth = len(code1)
    imgHeight = len(code1[0]) if imgWidth > 0 else 0
    q1 = []
    q2 = []
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            if mask1[x][y] != 0:
                phase1 = code1[x][y]
                minDiff = float("inf")
                minPixel = [-1, -1]
                for y2 in range(imgWidth):
                    if mask2[x][y2] != 0:
                        diff = abs(phase1 - code2[x][y2])
                        if diff < minDiff:
                            minDiff = diff
                            minPixel = [x, y2]
                q1.append([x, y])
                q2.append(minPixel)
    return q1, q2

# === Create image paths ===
imagesRefCam1Fn = []
imagesPrimaryPhaseCam1Fn = []
imagesSecondaryPhaseCam1Fn = []
for index in [0, 1]:
    imagesRefCam1Fn.append(PATH_TO_IMAGES + 'frames0_' + str(index) + '.png')
for index in range(2, 10):
    imagesPrimaryPhaseCam1Fn.append(PATH_TO_IMAGES + 'frames0_' + str(index) + '.png')
for index in range(10, 18):
    imagesSecondaryPhaseCam1Fn.append(PATH_TO_IMAGES_SEC_CAM1 + 'frames0_' + str(index) + '.png')

imagesRefCam2Fn = []
imagesPrimaryPhaseCam2Fn = []
imagesSecondaryPhaseCam2Fn = []
for index in [0, 1]:
    imagesRefCam2Fn.append(PATH_TO_IMAGES + 'frames1_' + str(index) + '.png')
for index in range(2, 10):
    imagesPrimaryPhaseCam2Fn.append(PATH_TO_IMAGES + 'frames1_' + str(index) + '.png')
for index in range(10, 18):
    imagesSecondaryPhaseCam2Fn.append(PATH_TO_IMAGES_SEC_CAM2 + 'frames1_' + str(index) + '.png')

if os.path.isfile('q1.xlsx') and os.path.isfile('q2.xlsx'):
    print('Loading precomputed point matches')
    q1 = pd.read_excel('q1.xlsx')
    q1.to_numpy()
    q1 = q1.T
    q2 = pd.read_excel('q2.xlsx')
    q2.to_numpy()
    q2 = q2.T
else:
    # === Load images and create masks ===
    imagesRefCam1 = loadImagesInGrayscale(imagesRefCam1Fn)
    imagesPrimaryPhaseCam1 = loadImagesInGrayscale(imagesPrimaryPhaseCam1Fn)
    imagesSecondaryPhaseCam1 = loadImagesInGrayscale(imagesSecondaryPhaseCam1Fn)

    imagesRefCam2 = loadImagesInGrayscale(imagesRefCam2Fn)
    imagesPrimaryPhaseCam2 = loadImagesInGrayscale(imagesPrimaryPhaseCam2Fn)
    imagesSecondaryPhaseCam2 = loadImagesInGrayscale(imagesSecondaryPhaseCam2Fn)

    imageMaskCam1 = getMask(imagesRefCam1[0])
    imageMaskCam2 = getMask(imagesRefCam2[0])
    # cv2.imshow('cam1', imageMaskCam1)
    # cv2.waitKey()
    # cv2.imshow('cam2', imageMaskCam2)
    # cv2.waitKey()
    
    # === Decode phase ===
    unwrappedPhaseCam1 = decodePhase(imagesPrimaryPhaseCam1, imagesSecondaryPhaseCam1)
    maskedCam1 = maskImage(unwrappedPhaseCam1, imageMaskCam1)
    # plt.imshow(maskedCam1, cmap="gray") 
    # plt.show()
    unwrappedPhaseCam2 = decodePhase(imagesPrimaryPhaseCam2, imagesSecondaryPhaseCam2)
    maskedCam2 = maskImage(unwrappedPhaseCam2, imageMaskCam2)

    # === Find point matches ===
    q1, q2 = registerContinous(maskedCam1, maskedCam2, imageMaskCam1, imageMaskCam2)
    # TODO: Fix writing to file
    q1.to_numpy()
    q1 = q1.T
    q2.to_numpy()
    q2 = q2.T
    df = pd.DataFrame(q1).T
    df.to_excel(excel_writer = "q1.xlsx")
    df = pd.DataFrame(q2).T
    df.to_excel(excel_writer = "q2.xlsx")

# === Plot point matches ===
"""
    #Takes a long time
y = [q1.iloc[:,0], q2.iloc[:,0]]
x = [q1.iloc[:,1], q2.iloc[:,1]]
plt.plot(x,y)
plt.show()
"""

cv2.destroyAllWindows()
