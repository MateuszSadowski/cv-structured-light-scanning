#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:44:14 2020

@author: s143818
"""


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import methods as meth


PATTERN_W = 22
PATTERN_H = 13
SQUARE_SIZE = 15.0
PATH_TO_CAL_IMAGES = 'calib_images_rect/'
PATH_TO_IMAGES = 'angel_rect/'


cornerCoords = np.zeros((PATTERN_H*PATTERN_W, 3), np.float32)
cornerCoords[:,:2] = np.mgrid[0:PATTERN_W*SQUARE_SIZE:SQUARE_SIZE,0:PATTERN_H*SQUARE_SIZE:SQUARE_SIZE].T.reshape(-1,2)

imCam1 = glob.glob(PATH_TO_CAL_IMAGES + 'frame0_*.png')
imCam2 = glob.glob(PATH_TO_CAL_IMAGES + 'frame1_*.png')

objectPoints = []
projPointsCam1 = []
projPointsCam2 = []

img4size = cv2.imread(imCam1[0], cv2.IMREAD_GRAYSCALE)
imSize = img4size.shape[::-1]

print('Hello checkerboard');
#checkerboard
for index, (image1, image2) in enumerate(zip(imCam1, imCam2)):
    #First camera
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    rv, corners = cv2.findChessboardCorners(img1, (PATTERN_W, PATTERN_H))
    if rv:
        objectPoints.append(cornerCoords)
        projPointsCam1.append(corners)
        
    #Second camera
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)
    rv, corners = cv2.findChessboardCorners(img2, (PATTERN_W, PATTERN_H))
    if rv:
        projPointsCam2.append(corners)



#Calibrate each camera
retval, cameraM1, distCoeffs1, rvecs, tvecs = cv2.calibrateCamera(objectPoints, projPointsCam1, imSize, None, None)
retval, cameraM2, distCoeffs2, rvecs, tvecs = cv2.calibrateCamera(objectPoints, projPointsCam2, imSize, None, None)

#Stereo calibrate cameras
retval, cameraM1, distCoeffs1, cameraM2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, projPointsCam1, projPointsCam2, cameraM1, distCoeffs1, cameraM2, distCoeffs2, imSize)

#retval
#CameraM1 (A1)
#distCoeffs1
#cameraM2 (A2)
#distCoeffs3
#R Rotation matrix
#T translation
#E Essential matrix
#F  Fundamental matrix

P1 = meth.calcProjectionM(R, T, F, cameraM1)
P2 = meth.calcProjectionM(R, T, F, cameraM2)

#def calcProjectionM(R, t, f, beta, alpha, deltax, deltay):
 #   A = np.matrix([[f, f*beta, deltax],[0, alpha*f, deltay],[0,0,1]])
 #   P = A*np.c_[R,t]
 #   return P



q1 = pd.read_excel('q1.xlsx')
q1.as_matrix()
q1 = q1.T
q2 = pd.read_excel('q2.xlsx')
q2.as_matrix()
q2 = q2.T



y = [q1.iloc[:,0], q2.iloc[:,0]]
x = [q1.iloc[:,1], q2.iloc[:,1]]

#plt.plot(x,y, 'g-')
#plt.show()


q1_1 = np.c_[q1,[1]*16383]
q2_1 = np.c_[q2,[1]*16383]


points = [[q1.iloc[:,0], q1.iloc[:,1]],[q2.iloc[:,0], q2.iloc[:,1]]]
ps = [P1,P2]

#x1 = [P1[0,:]*q1_1,]

#hope = cv2.triangulatePoints(points, ps)



#function point3d = triangulateOnePoint(point1, point2, P1, P2)

#P1 = np.matrix(P1)
#P2 = np.matrix(P2)
all3dp = np.empty([len(q1_1), 3])

for i, (q11, q22) in enumerate(zip(q1_1, q2_1)):
    #print(q11)
    point1 = q11;
    point2 = q22;
    
    # do the triangulation
    B1 = point1[0] * P1[2] - [P1[0]]
    B2 = point1[1] * P1[2] - [P1[1]]
    B3 = point2[0] * P2[2] - [P2[0]]
    B4 = point2[1] * P2[2] - [P2[1]]
    
    A = np.vstack((B1,B2,B3, B4))
    
    u, s, vh = np.linalg.svd(A)
    
    #temp = vh[:,3]
    temp = vh[3]
    
    point = temp/temp[3]
    point = [point[0],point[1],point[2]]
    all3dp[i] = point

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.view_init(60, 100)
ax.scatter3D(all3dp[:,0], all3dp[:,1], all3dp[:,2], cmap='Greens', s=.1)



#import plotly.graph_objects as go
#import numpy as np

# Helix equation
#t = np.linspace(0, 10, 50)
#x, y, z = np.cos(t), np.sin(t), t

#fig = go.Figure(data=[go.Scatter3d(x=all3dp[:,0], y=all3dp[:,1], z=all3dp[:,2],
                #                   mode='markers')])
#fig.show()



#point1 = q1.iloc[0];
#point2 = q2.iloc[0];

# do the triangulation
#A = (4,4)
#np.zeros(A)
#B1 = point1[0] * P1[2] - [P1[0]]
#B2 = point1[1] * P1[2] - [P1[1]]
#B3 = point2[0] * P2[2] - [P2[0]]
#B4 = point2[1] * P2[2] - [P2[1]]

#A = np.vstack((B1,B2,B3, B4))

#u, s, vh = np.linalg.svd(A)

#temp = vh[:,3]

#point = temp/temp[3]
#point = [point[0],point[1],point[2]]


#X = V.iloc[:,-1]
#X = X/X(-1)

#point3d = X([0,1,2])