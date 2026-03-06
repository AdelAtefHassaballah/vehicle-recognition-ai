from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.model = None

    def detect_plate(self, vehicle_crop):
        h, w = vehicle_crop.shape[:2]

        if self.model is not None:
            results = self.model(vehicle_crop, verbose=False)
            best_box = None
            best_conf = 0.0

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if conf > best_conf:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2)

            if best_box:
                return best_box, best_conf

        # fallback heuristic
        plate_x1 = int(w * 0.25)
        plate_y1 = int(h * 0.65)
        plate_x2 = int(w * 0.75)
        plate_y2 = int(h * 0.90)

        return (plate_x1, plate_y1, plate_x2, plate_y2), 0.30