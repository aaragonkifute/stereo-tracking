import numpy as np
import cv2
from pythonosc import udp_client
import argparse
from stereotracking import StereoTracking

st = StereoTracking()

# Setup cameras
capL = cv2.VideoCapture(2, cv2.CAP_DSHOW)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

capR = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#Set up tracking parameters and window to modify them dinamically
lower_threshold = 235
min_area = 90
max_area = 3000
min_circ = 0.75

def nothing(x):
    pass

cv2.namedWindow('Markers tracking',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Lower threshold','Markers tracking' ,lower_threshold,255, nothing)
cv2.createTrackbar('Min area', 'Markers tracking', min_area,200, nothing)
cv2.createTrackbar('Max area', 'Markers tracking', max_area,3000, nothing)
cv2.createTrackbar('Min circularity', 'Markers tracking', int(min_circ * 100),100, nothing)
cv2.createTrackbar('Send cameras positions', 'Markers tracking',0,1,nothing)

# Setup UDP connection with Unity via OSC
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5005)
args = parser.parse_args()
osc_client = udp_client.SimpleUDPClient(args.ip, args.port)

osc_client.send_message("/POS", st.parse_cameras_pos_rot())

while(True):
    
    # Retrieve frames from capturers
    _, imgL =  capL.retrieve()
    _, imgR =  capR.retrieve()

    
    # Retrieve values from window tracker
    lower_threshold = cv2.getTrackbarPos('Lower threshold','Markers tracking')
    min_area = cv2.getTrackbarPos('Min area','Markers tracking')
    max_area = cv2.getTrackbarPos('Max area','Markers tracking')
    min_circ = cv2.getTrackbarPos('Min circularity','Markers tracking') / 100
    s = cv2.getTrackbarPos('Send cameras positions','Markers tracking')
    
    # Switch to send cameras positions via OSC
    if s == 0:
        pass
    else:
        osc_client.send_message("/POS", st.parse_cameras_pos_rot())
        cv2.setTrackbarPos('Send cameras positions','Markers tracking',0)
    
    
    # Detect markers centers from the images
    markers_centers, thresh_frames = st.detect_markers_centers(
        imgL, imgR, lower_threshold, min_area, max_area, min_circ)
    
    resized_thresh_left = cv2.resize(thresh_frames[0],(720,480),interpolation = cv2.INTER_AREA)
    resized_thresh_right = cv2.resize(thresh_frames[1],(720,480),interpolation = cv2.INTER_AREA)
    
    resized_thresh = np.concatenate((resized_thresh_left, resized_thresh_right), axis=1)
    
    cv2.imshow('Markers tracking', resized_thresh)

    # If at least one frame is detected in each frame...
    if (len(markers_centers[0]) > 0 and len(markers_centers[1]) > 0):

        #Caculate markers 3D position
        markers_pos = st.calculate_markers_position(markers_centers)

        # Send positions to Unity through OSC
        XYZ = []
        for pos in markers_pos:
            for c in pos:
                XYZ.append(c)
                
        XYZ.insert(0, len(XYZ))
        osc_client.send_message("/XYZ", XYZ) 
        
    else:
        osc_client.send_message("/XYZ", "0")
        

    k = cv2.waitKey(33)
    if k==27:
        break
    
capL.release()
capR.release()
cv2.destroyAllWindows()

