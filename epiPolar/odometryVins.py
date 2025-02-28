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
        
        
        self.rotCumSum = np.array([0, 0, 0]).astype("float64")
        self.traCumSum = np.array([0, 0, 0]).astype("float64")
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=200, 
                      qualityLevel=0.3, 
                      minDistance=7, 
                      blockSize=7)
        self.lk_params = dict(winSize=(15, 15), 
                maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))






        self.camera = Camera(0, (1280, 720), 0, 0)
        self.camera.register_observer(self.newImage)
        self.run = 1




    def newImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        draw = image.copy()
        if(type(self.lastImage) != type(None)):
            try:
                if self.p0 is not None and len(self.p0) > 0:
                    self.p1, self.st, err = cv2.calcOpticalFlowPyrLK(self.lastImage, image, self.p0, None, **self.lk_params)

                    # Select good points
                    if self.p1 is not None:
                        self.good_new = self.p1[self.st == 1]
                        self.good_old = self.p0[self.st == 1]

                        # Draw the tracks
                        
                        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            draw = cv2.circle(draw, (int(a), int(b)), 5, (0, 0, 255), -1)

                        # Update previous frame and points
                        self.lastImage = image.copy()
                        self.p0 = self.good_new.reshape(-1, 1, 2)

                # If too few points remain, detect new features
                if self.p0 is None or len(self.p0) < 10:
                    self.p0 = cv2.goodFeaturesToTrack(image, mask=None, **self.feature_params)
                    self.mask = np.zeros_like(self.lastImage)  # Reset mask
                

                # Estimate the Fundamental Matrix
                F, mask = cv2.findFundamentalMat(self.good_old, self.good_new, cv2.FM_RANSAC)
                E = self.K.T @ F @ self.K

                # Decompose Essential Matrix to get R and t
                _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.K)
                rr = np.array(np.rad2deg(rotation_matrix_to_euler_angles(R)))
                self.rotCumSum += rr
                #print(t)
                self.traCumSum += t[0]
                print(self.traCumSum)
            except (KeyboardInterrupt):
                self.run = 0
        else:
            self.p0 = cv2.goodFeaturesToTrack(image, mask=None, **self.feature_params)
            self.mask = np.zeros_like(image)



        self.lastImage = image.copy()
        cv2.imshow("main", draw)
        cv2.waitKey(10)
        


if __name__ == "__main__":
    import time
    o = Odometry()
    while(o.run):
        time.sleep(1)
