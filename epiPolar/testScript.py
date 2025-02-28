import cv2
import numpy as np
from calibrations import getCameraCalibration



def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return x, y, z

# Load images
img1 = cv2.imread('static/3.jpg', 0)
img2 = cv2.imread('static/2.jpg', 0)

# Detect features and match
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Estimate the Fundamental Matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Compute Essential Matrix (if K is known)
c = getCameraCalibration("legion_720p")
K = c["camera_matrix"]
#K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix

E = K.T @ F @ K

# Decompose Essential Matrix to get R and t
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation:\n", R)
print("Translation:\n", t)

print(np.rad2deg(rotation_matrix_to_euler_angles(R)))