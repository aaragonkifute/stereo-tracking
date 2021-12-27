import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num = 10

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == 32:
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("Images saved!")
        num += 1
        
    resized_img = cv2.resize(img,(720,480),interpolation = cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2,(720,480),interpolation = cv2.INTER_AREA)
    preview = np.concatenate((resized_img,resized_img2), axis=1)
    cv2.imshow('Preview', preview)
    

cv2.destroyAllWindows()