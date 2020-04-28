#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:55:29 2020

@author: s143818
"""

import cv2
import numpy as np

LOW_I_CUTOFF = 0.1 * 255
HIGH_I_CUTOFF = 0.9 * 255


def calcProjectionM(R, t, f, A):
    A = np.c_[A,[0,0,0]]
    P = A*np.c_[R,t]
    #P = A*np.c_[[0,0,0]]
    return P

def loadImInGrayscale(images):
    gray = []
    for image in images:
        gray.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
    #cv2.imshow('Window',gray(0))
    #cv2.waitKey(0)
    return gray

def getMask(image):
    imageSize = image.shape[::-1]
    mask = np.ones((imageSize[1], imageSize[0]))
    for x in range(imageSize[1]):
        for y in range(imageSize[0]):
            intensity = image[x, y]
            if (intensity < LOW_I_CUTOFF) or (HIGH_I_CUTOFF < intensity):
                mask[x, y] = 0
    return mask

#XNOT WORKING YET!
def decodePhase(imPrim, imSeq):
    F_primary = np.fft.fft(imPrim)
    F_seq = np.fft.fft(imSeq);
 
        
    channel = 1
    imPrim = np.angle(F_primary[channel])
    imSeq = np.angle(F_seq[channel])
    
    phase = imPrim - imSeq % (2 * np.pi)

    return phase
