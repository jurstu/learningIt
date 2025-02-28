
import numpy as np



class Rotation:
    def __init__(self):
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def getRotationMatrix(self):
        # against Y axis
        rm = np.array([
            [ np.cos(self.roll), 0, np.sin(self.roll)],
            [                 0, 1,                 0],
            [-np.sin(self.roll), 0, np.cos(self.roll)]
        ])


        # against X axis
        pm = np.array([
            [                 1,                   0,                   0],
            [                 0,  np.cos(self.pitch), -np.sin(self.pitch)],
            [                 0,  np.sin(self.pitch),  np.cos(self.pitch)]
        ])


        # against Z axis
        ym = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw),                   0],
            [np.sin(self.yaw),  np.cos(self.yaw),                   0],
            [               0,                 0,                   1]
        ])

        FullRot = np.dot(np.dot(ym, pm), rm)
        return FullRot
    

