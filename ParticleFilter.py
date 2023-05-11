import cv2
import numpy as np


# Particle filter class
class ParticleFilter:
    def __init__(self, num_particles, frame_shape):
        self.num_particles = num_particles
        self.particles = np.random.rand(self.num_particles, 2) * frame_shape

    def predict(self):
        self.particles += np.random.randn(self.num_particles, 2) * 20

    def update(self, target):
        distances = np.linalg.norm(self.particles - target, axis=1)
        weights = np.exp(-distances)
        weights /= np.sum(weights)
        self.resample(weights)

    def resample(self, weights):
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=weights)
        self.particles = self.particles[indices]

    def draw(self, frame):
        for particle in self.particles:
            cv2.circle(frame, tuple(particle.astype(int)), 1, (0, 0, 255), -1)
