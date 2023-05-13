import cv2
import sys
import time
import atexit
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QAbstractButton
from PySide6.QtCore import QTimer, QRect
from PySide6.QtGui import QImage, QPixmap, QFont, Qt, QPen, QPainter, QColor
from PySide6.QtUiTools import QUiLoader
from particle_filter import ParticleFilter


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui()
        self.setup_camera()
        self.setup_clahe()
        self.setup_particle_filter()
        self.setup_font()
        self.setup_frame_update_timer()
        self.setup_fps_counter()

    def set_button_state(self, state):
        button_list = self.ui.findChildren(QAbstractButton)
        for button in button_list:
            button.setEnabled(state)

    def setup_ui(self):
        loader = QUiLoader()
        self.ui = loader.load('./Form Files/COMPUTER_VISION.ui', None)
        self.ui.setFixedSize(self.ui.size())
        self.ui.setWindowFlags(self.ui.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.ui.show()
        self.statusbar = self.ui.statusBar()
        self.set_button_state(False)
        self.ui.claheBtn.clicked.connect(self.toggle_clahe)
        self.ui.pftrkBtn.clicked.connect(self.toggle_particle_filter)

    def setup_camera(self):
        self.setting_label = QLabel(self.ui.videoLabel)
        self.setting_label.setGeometry(QRect(0, 0, self.ui.videoLabel.width(), self.ui.videoLabel.height()))
        self.setting_label.setText("Camera Initialize...")
        self.setting_label.setAlignment(Qt.AlignCenter)
        self.setting_label.setFrameStyle(QFrame.NoFrame)
        self.setting_label.show()
        QApplication.processEvents()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        atexit.register(self.cap.release)
        self.setting_label.hide()
        self.set_button_state(True)

    def setup_clahe(self):
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 6))
        self.apply_clahe = False

    def setup_particle_filter(self):
        self.apply_particle_filter = False
        self.particle_filter = ParticleFilter(3000, (self.ui.videoLabel.width(), self.ui.videoLabel.height()))

    def setup_font(self):
        self.font_bold = QFont()
        self.font_bold.setBold(True)
        self.font_normal = QFont()
        self.font_normal.setBold(False)

    def setup_frame_update_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def setup_fps_counter(self):
        self.frames = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.timer_fps = QTimer()
        self.timer_fps.timeout.connect(self.update_fps)
        self.timer_fps.start(10)

    def toggle_clahe(self):
        self.apply_clahe = not self.apply_clahe

        if self.apply_clahe:
            font = self.font_bold
        else:
            font = self.font_normal

        self.ui.claheBtn.setFont(font)

    def toggle_particle_filter(self):
        self.apply_particle_filter = not self.apply_particle_filter

        if self.apply_particle_filter:
            font = self.font_bold
        else:
            font = self.font_normal

        self.ui.pftrkBtn.setFont(font)

    def calculate_fps(self):
        now = time.time()
        duration = now - self.last_time
        if duration >= 1.0:
            self.fps = self.frames / duration
            self.frames = 0
            self.last_time = now

    def update_fps(self):
        self.statusbar.showMessage(f"FPS: {self.fps:.8f}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frames += 1
        self.calculate_fps()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.apply_clahe:
            gray_frame = self.clahe.apply(gray_frame)

        if self.apply_particle_filter:
            _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary_frame)
            if moments['m00'] > 0:
                target = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
                self.particle_filter.update(target)

            self.particle_filter.predict()

        self.display_image(gray_frame)

    def display_image(self, image):
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        if self.apply_particle_filter:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0, 255), 1)
            painter.setPen(pen)

            for particle in self.particle_filter.particles:
                x, y = particle.astype(int)
                painter.drawEllipse(x, y, 1, 1)

            painter.end()

        self.ui.videoLabel.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
