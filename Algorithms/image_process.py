import cv2
import numpy as np
from Algorithms.particle_filter import ParticleFilter


class VideoProcessor:
    def __init__(self, apply=False):
        self.apply = apply

    def toggle(self):
        self.apply = not self.apply

    def process(self, frame):
        pass


class ClaheProcessor(VideoProcessor):
    def __init__(self, apply=False):
        super().__init__(apply)
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 6))

    def process(self, frame):
        if self.apply:
            return self.clahe.apply(frame)
        return frame


class ParticleFilterProcessor(VideoProcessor):
    def __init__(self, dimensions, n_particles, apply=False):
        super().__init__(apply)
        self.particle_filter = ParticleFilter(n_particles, dimensions)

    def process(self, frame):
        if self.apply:
            _, binary_frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary_frame)
            if moments['m00'] > 0:
                target = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
                self.particle_filter.update(target)
            self.particle_filter.predict()
        return frame
