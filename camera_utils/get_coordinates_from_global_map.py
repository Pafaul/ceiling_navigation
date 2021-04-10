import numpy as np
import cv2

def get_coordinated_from_global_map(gm_kp, gm_des, img_kp, img_des, matcher, img_shape, required_matches=10):
    matches = matcher.knnMatch(img_des, gm_des, k=2)
    if len(matches) < required_matches:
        return None

    pts1, pts2 = [], []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts1.append(img_kp[m.queryIdx].pt) #src
            pts2.append(gm_kp[m.trainIdx].pt)  #dst

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img_shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    return dst