
import numpy as np



class Rotation:
    def __init__(self):
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def rotatePoints(self, points, rotation):
        OUT = [[0,0,0]] * len(points)
        for i, p in enumerate(points):
            rotatedPoint = np.dot(rotation, p)
            OUT[i] = rotatedPoint
        return OUT

    def getRotationMatrixInverted(self):
        R = self.getRotationMatrix()
        return np.linalg.inv(R)

    def getRotationMatrix(self):
        # against Z axis
        rm = np.array([
            [np.cos(self.roll), -np.sin(self.roll),                   0],
            [np.sin(self.roll),  np.cos(self.roll),                   0],
            [               0,                 0,                   1]
        ])


        # against X axis
        pm = np.array([
            [                 1,                   0,                   0],
            [                 0,  np.cos(self.pitch), -np.sin(self.pitch)],
            [                 0,  np.sin(self.pitch),  np.cos(self.pitch)]
        ])


        # against Y axis
        ym = np.array([
            [ np.cos(self.yaw), 0, np.sin(self.yaw)],
            [                 0, 1,                 0],
            [-np.sin(self.yaw), 0, np.cos(self.yaw)]
        ])

        FullRot = np.dot(np.dot(ym, pm), rm)
        return FullRot
    

