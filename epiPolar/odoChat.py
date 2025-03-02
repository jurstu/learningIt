import cv2
import numpy as np
from camera import Camera

import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, intrinsic_matrix, min_features=100, max_features=500):
        """ 
        Initialize visual odometry with camera intrinsic matrix K.
        
        :param intrinsic_matrix: Camera intrinsic matrix (3x3 NumPy array)
        :param min_features: Minimum number of tracked features before detecting new ones
        :param max_features: Maximum number of features to track
        """
        self.K = intrinsic_matrix
        self.min_features = min_features
        self.max_features = max_features
        self.prev_gray = None
        self.prev_pts = None
        self.pose = np.eye(4)  # Camera pose (4x4 transformation matrix)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def add_frame(self, image):
        """ 
        Process an image, detect motion, and return (R, t) transformation.
        
        :param image: Input frame (BGR or grayscale)
        :return: Rotation matrix R (3x3), translation vector t (3x1), and motion vectors [(x1, y1, dx, dy), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Initialize feature points if first frame or lost tracking
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_gray, self.prev_pts = gray, self._detect_features(gray)
            return np.eye(3), np.zeros((3, 1)), []

        # Track features using Lucas-Kanade optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        # Validate tracked points
        if new_pts is None or status is None or np.count_nonzero(status) < self.min_features:
            self.prev_gray, self.prev_pts = gray, self._detect_features(gray)
            return np.eye(3), np.zeros((3, 1)), []

        # Extract valid points and reshape to (N, 2)
        good_new = new_pts[status.flatten() == 1].reshape(-1, 2)
        good_old = self.prev_pts[status.flatten() == 1].reshape(-1, 2)

        # Compute Essential Matrix and recover camera motion
        R, t = self._estimate_motion(good_old, good_new)

        # Compute motion vectors
        vectors = [(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) 
                   for (x1, y1), (x2, y2) in zip(good_old, good_new)]

        # Re-detect features if tracking drops below threshold
        if len(good_new) < self.min_features:
            new_features = self._detect_features(gray)
            if new_features is not None and len(new_features) > 0:
                good_new = np.vstack((good_new, new_features.reshape(-1, 2)))

        # Limit the number of tracked features
        if len(good_new) > self.max_features:
            good_new = good_new[:self.max_features]

        # Update state
        self.prev_gray, self.prev_pts = gray, good_new.reshape(-1, 1, 2)

        return R, t, vectors

    def _estimate_motion(self, points1, points2):
        """ Estimates motion (R, t) using the Essential Matrix. """
        E, _ = cv2.findEssentialMat(points2, points1, self.K, method=cv2.RANSAC, threshold=1.0)

        if E is None:
            return np.eye(3), np.zeros((3, 1))  # No motion detected, return identity

        _, R, t, _ = cv2.recoverPose(E, points2, points1, self.K)
        return R, t

    def _detect_features(self, gray):
        """ Detects new feature points if needed. """
        features = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, 
                                            qualityLevel=0.01, minDistance=10)
        return features if features is not None else np.empty((0, 1, 2))



if __name__ == "__main__":
    from calibrations import getCameraCalibration
    
    c = getCameraCalibration("legion_720p")
    K = c["camera_matrix"]
    motion_estimator = VisualOdometry(K)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        R, t, vectors = motion_estimator.add_frame(frame)
        
        #for (x, y, dx, dy) in vectors:
        #    cv2.arrowedLine(frame, (int(x), int(y)), (int(x + dx), int(y + dy)), (0, 255, 0), 2)

        cv2.imshow("Motion Vectors", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()