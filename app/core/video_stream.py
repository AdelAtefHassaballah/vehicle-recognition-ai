import cv2

class VideoStream:
    def __init__(self, source: str | int):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

    def read_frame(self):
        success, frame = self.cap.read()
        return success, frame

    def release(self):
        if self.cap:
            self.cap.release()