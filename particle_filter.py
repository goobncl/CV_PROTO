import cv2
import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, frame_shape):
        # 클래스 초기화. 입자의 개수와 프레임의 형태를 입력으로 받는다.
        self.num_particles = num_particles  # 사용할 입자의 수를 저장한다.
        self.particles = np.random.rand(self.num_particles, 2) * frame_shape  # 무작위 위치에 입자를 초기화. 각 입자는 프레임 내의 (x, y) 위치를 나타낸다.

    def predict(self):
        # 입자의 위치를 예측. 이 경우에는 간단하게 가우시안 노이즈를 추가하여 입자를 이동.
        self.particles += np.random.randn(self.num_particles, 2) * 20  # 각 입자에 대해 평균 0, 표준편차 20의 가우시안 노이즈를 추가합.

    def update(self, target):
        # 목표 위치에 따라 입자의 가중치를 업데이트.
        distances = np.linalg.norm(self.particles - target, axis=1)  # 각 입자와 목표 사이의 거리를 계산.
        weights = np.exp(-distances)  # 거리에 음의 지수를 적용하여 가중치를 계산. 목표에 가까운 입자가 더 높은 가중치를 가짐.
        weights /= np.sum(weights)  # 가중치를 정규화하여 합계가 1이 되도록 함.
        self.resample(weights)  # 가중치에 따라 입자를 재샘플링.

    def resample(self, weights):
        # 가중치에 따라 입자를 재샘플링. 높은 가중치를 가진 입자는 새로운 샘플 집합에서 더 자주 나타남.
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles,p=weights)  # 가중치에 따라 입자 인덱스를 선택. 높은 가중치를 가진 입자가 더 자주 선택됨.
        self.particles = self.particles[indices]  # 선택된 인덱스에 해당하는 입자로 새로운 입자 집합을 구성.

    def draw(self, frame):
        # 각 입자의 위치에 원을 그려 프레임에 입자를 그림.
        for particle in self.particles:
            cv2.circle(frame, tuple(particle.astype(int)), 1, (0, 0, 255), -1)  # 각 입자의 위치에 반지름이 1이고 색이 빨간색인 원을 그림.
