from ultralytics import YOLO
from app.config import ALLOWED_VEHICLE_CLASSES
from app.schemas import DetectionResult
import torch


class VehicleDetector:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame):
        results = self.model(frame, device=self.device, conf=0.25, imgsz=640)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id in ALLOWED_VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detections.append(
                        DetectionResult(
                            class_name=ALLOWED_VEHICLE_CLASSES[cls_id],
                            confidence=conf,
                            bbox=(x1, y1, x2, y2)
                        )
                    )

        return detections