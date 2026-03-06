import re
import cv2
from datetime import datetime
from pathlib import Path

def clean_plate_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def save_snapshot(frame, folder: Path, prefix: str = "vehicle") -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = folder / filename
    cv2.imwrite(str(path), frame)
    return str(path)