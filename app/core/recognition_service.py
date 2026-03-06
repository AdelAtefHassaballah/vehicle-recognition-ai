import cv2
from app.database import SessionLocal
from app.models import VehicleLog
from app.utils import save_snapshot
from app.config import SNAPSHOT_DIR

class VehicleRecognitionService:
    def __init__(self, detector, plate_detector, ocr_reader, tracker, camera_id: str):
        self.detector = detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.tracker = tracker
        self.camera_id = camera_id

    def process_frame(self, frame):
        detections = self.detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                continue

            plate_bbox, _ = self.plate_detector.detect_plate(vehicle_crop)
            px1, py1, px2, py2 = plate_bbox

            plate_crop = vehicle_crop[py1:py2, px1:px2]
            plate_text = None
            ocr_conf = 0.0

            if plate_crop.size != 0:
                plate_text, ocr_conf = self.ocr_reader.read_plate(plate_crop)

            if self.tracker.should_log(plate_text, det.class_name):
                snapshot_path = save_snapshot(vehicle_crop, SNAPSHOT_DIR, prefix=det.class_name)
                self._save_to_db(
                    plate_number=plate_text,
                    vehicle_type=det.class_name,
                    snapshot_path=snapshot_path
                )

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} | Plate: {plate_text if plate_text else 'N/A'}"
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # draw plate area relative to original frame
            gx1, gy1 = x1 + px1, y1 + py1
            gx2, gy2 = x1 + px2, y1 + py2
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

        return frame

    def _save_to_db(self, plate_number, vehicle_type, snapshot_path):
        session = SessionLocal()
        try:
            log = VehicleLog(
                plate_number=plate_number,
                vehicle_type=vehicle_type,
                camera_id=self.camera_id,
                snapshot_path=snapshot_path
            )
            session.add(log)
            session.commit()
        finally:
            session.close()