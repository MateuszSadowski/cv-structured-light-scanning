import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

PATH_TO_IMAGES = '../angel_rect/'
PATH_TO_IMAGES_SEC_CAM1 = '../angel_rect/secondary_cam1/'
PATH_TO_IMAGES_SEC_CAM2 = '../angel_rect/secondary_cam2/'
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

def decodePhase(primary, secondary):
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
            # fftPrimary = np.fft.fft(tmpPhasePrimary[x][y]) # FFT across time dimension
            # wrappedPrimary = np.angle(fftPrimary[1]) # Get phase from the second channel
            # fftSecondary = np.fft.fft(tmpPhaseSecondary[x][y]) # FFT across time dimension
            # wrappedSecondary = np.angle(fftSecondary[1]) # Get phase from the second channel

            wrappedPrimaryImg[x][y] = wrappedPrimary
            heterodynePhase[x][y] = (wrappedPrimary - wrappedSecondary) % (2 * np.pi)
    # heterodynePhase = (wrappedPrimary - wrappedSecondary) % (2 * np.pi)

    cv2.imshow('wrapped phase1', wrappedPrimaryImg)
    cv2.waitKey()
    return heterodynePhase

def maskImage(img, mask):
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

# === Load images and create masks ===
imagesRefCam1Fn = glob.glob(PATH_TO_IMAGES + 'frames0_[0-1].png')
imagesPrimaryPhaseCam1Fn = glob.glob(PATH_TO_IMAGES + 'frames0_[2-9].png')
imagesSecondaryPhaseCam1Fn = glob.glob(PATH_TO_IMAGES_SEC_CAM1 + '*.png')

imagesRefCam2Fn = glob.glob(PATH_TO_IMAGES + 'frames1_[0-1].png')
imagesPrimaryPhaseCam2Fn = glob.glob(PATH_TO_IMAGES + 'frames1_[2-9].png')
imagesSecondaryPhaseCam2Fn = glob.glob(PATH_TO_IMAGES_SEC_CAM2 + '*.png')

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
# maskedCam1 = maskImage(unwrappedPhaseCam1, imageMaskCam1)
cv2.imshow('phase1', unwrappedPhaseCam1)
cv2.waitKey()

cv2.destroyAllWindows()