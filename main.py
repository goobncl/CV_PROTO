import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QFont
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load('COMPUTER_VISION.ui', None)
        self.ui.show()

        self.cap = cv2.VideoCapture(0)
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 6))
        self.apply_clahe = False

        self.font_bold = QFont()
        self.font_bold.setBold(True)
        self.font_normal = QFont()
        self.font_normal.setBold(False)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.ui.claheBtn.clicked.connect(self.toggle_clahe)

    def toggle_clahe(self):
        self.apply_clahe = not self.apply_clahe
        if self.apply_clahe:
            self.ui.claheBtn.setFont(self.font_bold)
        else:
            self.ui.claheBtn.setFont(self.font_normal)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.apply_clahe:
            frame_gray = self.clahe.apply(frame_gray)

        height, width = frame_gray.shape
        bytes_per_line = width
        qimage = QImage(frame_gray.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.ui.videoLabel.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
