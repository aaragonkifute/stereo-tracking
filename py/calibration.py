import numpy as np
import cv2 as cv
import glob

chessboardSize = (9,6)
frameSize = (1280,720)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
        

size_of_chessboard_squares_mm = 1
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('images/stereoRight/*.png'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(333)


cv.destroyAllWindows()

# Calibration

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

rvecL = rvecsL[len(rvecsL)-1]
rvecR = rvecsR[len(rvecsR)-1]
tvecL = tvecsL[len(tvecsL)-1]
tvecR = tvecsR[len(tvecsR)-1]
rotML = cv.Rodrigues(rvecL)[0]
rotMR = cv.Rodrigues(rvecR)[0]


# Save parameters

cv_file = cv.FileStorage('calibration.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('cameraMatrixL',cameraMatrixL)
cv_file.write('cameraMatrixR',cameraMatrixR)
cv_file.write('newCameraMatrixL',newCameraMatrixL)
cv_file.write('newCameraMatrixR',newCameraMatrixR)
cv_file.write('distL',distL)
cv_file.write('distR',distR)
cv_file.write('rvecL',rvecL)
cv_file.write('rvecR',rvecR)
cv_file.write('tvecL',tvecL)
cv_file.write('tvecR',tvecR)
cv_file.write('rotML',rotML)
cv_file.write('rotMR',rotMR)
cv_file.write('posL',-np.matrix(rotML).T * np.matrix(tvecL))
cv_file.write('posR',-np.matrix(rotMR).T * np.matrix(tvecR))
cv_file.write('inverseNewCameraMatrixL', np.linalg.inv(newCameraMatrixL))
cv_file.write('inverseNewCameraMatrixR', np.linalg.inv(newCameraMatrixR))


cv_file.release()

print('\x1b[6;30;42m' + 'Parameters saved!' + '\x1b[0m')
