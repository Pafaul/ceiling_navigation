import cv2
import numpy as np

def match_keypoints(kp_1, des_1, kp_2, des_2, matcher):
    matches = matcher.knnMatch(des_1,des_2,k=2)
    
    pts1 = []
    pts2 = []
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts1.append(kp_1[m.queryIdx].pt)
            pts2.append(kp_2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2