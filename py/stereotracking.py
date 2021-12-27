import cv2
import numpy as np
import math

def unprojectPoints(points, intrinsic, dist, newIntrinsic,
                    inverseNewIntrinsic, tvec, rotM):
    
    """ 
    Given makers centers (ponints), detected in one capturer image,
    unproject them and get the temporal 3D point without depth
    """
    
    undistorted_points = cv2.undistortPoints(np.array(points),
                                              intrinsic,
                                              dist,
                                              P=newIntrinsic);   
  
    result = []
    for point in undistorted_points:
        
        u,v = point[0,0], point[0,1]
        uv_1=np.array([[u,v,1]], dtype=np.float32)
        uv_1=uv_1.T
        xyz_c=inverseNewIntrinsic.dot(uv_1)
        xyz_c=xyz_c-tvec
        xyz=rotM.T.dot(xyz_c)

        
        result.append(xyz.ravel())
        
    return result

def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    """ 
    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    return the closest points on each segment and their distance
    """
    

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)

class StereoTracking(object):
    """
    This class tracks markers
    It obtains markers posticion from camera images, match rays if
    multiple markers are detected and send them to Unity
    """

    def __init__(self):
        
        """
            Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
        """
        
        cv_file = cv2.FileStorage()
        cv_file.open('calibration.xml', cv2.FileStorage_READ)
        
        self.cameraMatrixL = cv_file.getNode('cameraMatrixL').mat()
        self.cameraMatrixR = cv_file.getNode('cameraMatrixR').mat()
        self.newCameraMatrixL = cv_file.getNode('newCameraMatrixL').mat()
        self.newCameraMatrixR = cv_file.getNode('newCameraMatrixR').mat()
        self.distL = cv_file.getNode('distL').mat()
        self.distR = cv_file.getNode('distR').mat()
        self.rvecL = cv_file.getNode('rvecL').mat()
        self.rvecR = cv_file.getNode('rvecR').mat()
        self.tvecL = cv_file.getNode('tvecL').mat()
        self.tvecR = cv_file.getNode('tvecR').mat()
        self.rotML = cv_file.getNode('rotML').mat()
        self.rotMR = cv_file.getNode('rotMR').mat()
        self.posL = cv_file.getNode('posL').mat()
        self.posR = cv_file.getNode('posR').mat()
        self.inverseNewCameraMatrixL = cv_file.getNode('inverseNewCameraMatrixL').mat()
        self.inverseNewCameraMatrixR = cv_file.getNode('inverseNewCameraMatrixR').mat()
    
        
    def parse_cameras_pos_rot(self):
        """Returns array containing models positions"""
        
        cameras_pos = []
        for c in np.concatenate((self.posL
                                 , self.posR)):
            cameras_pos.append(c[0])
            
           
        return cameras_pos
    
    def detect_markers_centers(self, img1, img2, lower_threshold, min_area, max_area, min_circ):
        """
        Process frames, detect blobs, classify them depending
        on specified parameters and return blobs centers array
        """
        
        markers_centers = []
        thresh_frames = []
        
        for frame in [img1, img2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            #Erode??
            close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
            ret,thresh = cv2.threshold(close,lower_threshold,255,cv2.THRESH_BINARY)
            rgb_tresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
            
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            popup = []
            
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                if (area < min_area or area > max_area):
                    popup.append(i)
                else:
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4*np.pi*(area/(perimeter*perimeter))
                    
                    if circularity < min_circ:
                        popup.append(i)
                                
            blobs = []
            
            for i, cnt in enumerate(contours):
                if i not in popup:
                    
                    cv2.drawContours(rgb_tresh, cnt, -1, (0, 255, 0), 3)
                    
                    
                    center, _ = cv2.minEnclosingCircle(
                        cv2.approxPolyDP(cnt,3, True))
                    blobs.append(center)
                    
            thresh_frames.append(rgb_tresh)
            
            markers_centers.append(blobs)
            
        return markers_centers, thresh_frames
    
    def calculate_markers_position(self, markers_centers):
        """
        Calculate markers 3D position given its centers 
        detected on the image
        """
        
        markers_pos = []
        
        upL = unprojectPoints(markers_centers[0], self.cameraMatrixL, 
                              self.distL, self.newCameraMatrixL,
                              self.inverseNewCameraMatrixL, self.tvecL,
                              self.rotML)
        
        upR = unprojectPoints(markers_centers[1], self.cameraMatrixR, 
                              self.distR, self.newCameraMatrixR,
                              self.inverseNewCameraMatrixR, self.tvecR,
                              self.rotMR)    
        
        while len(upL) > 0 and len(upR) > 0:
            distances = np.zeros(len(upR))
            pointL = upL.pop()
            
            for i in range(len(distances)):
                _, _, distance = closestDistanceBetweenLines(self.posL.ravel(), pointL,
                                                      self.posR.ravel(), upR[i])
                distances[i] = distance
            
            pointR = upR.pop(np.argmin(distances))
            
            pA, pB, _ = closestDistanceBetweenLines(self.posL.ravel(), pointL,
                                                  self.posR.ravel(), pointR)
            
            markers_pos.append((pA+pB)/2)
            
        return markers_pos


        
        

    
        




