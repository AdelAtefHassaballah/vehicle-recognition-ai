import easyocr
import cv2
from app.utils import clean_plate_text

class OCRReader:
    def __init__(self, languages=None):
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=False)

    def read_plate(self, plate_image):
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)

        best_text = None
        best_conf = 0.0

        for _, text, conf in results:
            cleaned = clean_plate_text(text)
            if cleaned and conf > best_conf:
                best_text = cleaned
                best_conf = float(conf)

        return best_text, best_conf