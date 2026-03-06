from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SNAPSHOT_DIR = BASE_DIR / "app" / "data" / "snapshots"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "sqlite:///vehicle_logs.db"

VEHICLE_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
PLATE_MODEL_PATH = str(MODELS_DIR / "license_plate_detector.pt")

CAMERA_ID = "CAM_01"

# COCO class IDs from YOLO
# car=2, motorcycle=3, bus=5, truck=7
ALLOWED_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}