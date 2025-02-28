import cv2
import numpy as np
from calibrations import getCameraCalibration
from camera import Camera


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


class Odometry:
    def __init__(self):
        self.lastImage = None
        c = getCameraCalibration("legion_720p")
        self.K = c["camera_matrix"]
        self.camera = Camera(0, (1280, 720), 0, 0)
        self.camera.register_observer(self.newImage)
        self.rotCumSum = np.array([0, 0, 0]).astype("float64")
        self.traCumSum = np.array([0, 0, 0]).astype("float64")
        self.run = 1

    def newImage(self, image):
        if(type(self.lastImage) != type(None)):
            try:
                img1 = self.lastImage
                img2 = image
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
                E = self.K.T @ F @ self.K

                # Decompose Essential Matrix to get R and t
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                rr = np.array(np.rad2deg(rotation_matrix_to_euler_angles(R)))
                self.rotCumSum += rr
                #print(t)
                self.traCumSum += t[0]
                print(self.traCumSum)
            except (KeyboardInterrupt):
                self.run = 0


        self.lastImage = image.copy()


if __name__ == "__main__":
    import time
    o = Odometry()
    while(o.run):
        time.sleep(1)
