import cv2
import numpy as np
from ParticleFilter import ParticleFilter


def main():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame_shape = frame.shape[1::-1]
    pf = ParticleFilter(1000, frame_shape)

    while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_bin = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_BINARY)

        moments = cv2.moments(frame_bin)
        if moments['m00'] > 0:
            target = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
            pf.update(target)

        pf.predict()
        pf.draw(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

