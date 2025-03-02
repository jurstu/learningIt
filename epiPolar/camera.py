import cv2
import threading, time
import numpy as np

class Camera:
    def __init__(self, camera_id, resolution, camera_matrix, distortion_coeffs):
        self.camera_id = camera_id
        self.resolution = resolution
        self.distortion_coeffs = distortion_coeffs
        self.camera_matrix = camera_matrix
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if(len(resolution) == 3):
            self.cap.set(cv2.CAP_PROP_FPS, resolution[2])

        self.latest_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.observers = []
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.latest_frame = frame
            self.notify_observers()

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer(self.latest_frame)

    def release(self):
        self.cap.release()

# Example usage
if __name__ == "__main__":
    
    def observer(frame):
        pass

    camera = Camera(camera_id=0, resolution=(640, 480, 30), camera_matrix=None, distortion_coeffs=None)
    camera.register_observer(observer)


    while True:
        cv2.imshow('Camera', camera.latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


