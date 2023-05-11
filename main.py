import sys
import time
import atexit
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QFrame
from PySide6.QtCore import QTimer, QRect
from PySide6.QtGui import QImage, QPixmap, QFont, Qt
from PySide6.QtUiTools import QUiLoader
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui()
        self.setup_camera()
        self.setup_clahe()
        self.setup_font()
        self.setup_frame_update_timer()
        self.setup_fps_counter()

    def setup_ui(self):
        loader = QUiLoader()
        self.ui = loader.load('./Form Files/COMPUTER_VISION.ui', None)
        self.ui.setFixedSize(self.ui.size())
        self.ui.setWindowFlags(self.ui.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.ui.show()
        self.statusbar = self.ui.statusBar()
        self.ui.claheBtn.clicked.connect(self.toggle_clahe)

    def setup_camera(self):
        self.setting_label = QLabel(self.ui.videoLabel)
        self.setting_label.setGeometry(QRect(0, 0, self.ui.videoLabel.width(), self.ui.videoLabel.height()))  # Adjust position and size of the QLabel
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

    def setup_clahe(self):
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 6))
        self.apply_clahe = False

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
        self.ui.claheBtn.setFont(self.font_bold if self.apply_clahe else self.font_normal)

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

        self.display_image(gray_frame)

    def display_image(self, image):
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.ui.videoLabel.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
