import cv2
from pathlib import Path
from app.config import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, CAMERA_ID
from app.database import Base, engine
from app.core.video_stream import VideoStream
from app.core.detector import VehicleDetector
from app.core.plate_detector import PlateDetector
from app.core.ocr_reader import OCRReader
from app.core.tracker import SimplePlateTracker
from app.core.recognition_service import VehicleRecognitionService


def run(source=0, save_output=True):
    Base.metadata.create_all(bind=engine)

    video_stream = VideoStream(source)
    detector = VehicleDetector(VEHICLE_MODEL_PATH)
    plate_detector = PlateDetector(PLATE_MODEL_PATH)
    ocr_reader = OCRReader(['en'])
    tracker = SimplePlateTracker(cooldown_seconds=10)

    service = VehicleRecognitionService(
        detector=detector,
        plate_detector=plate_detector,
        ocr_reader=ocr_reader,
        tracker=tracker,
        camera_id=CAMERA_ID
    )

    writer = None
    output_path = Path("output_result.mp4")

    try:
        while True:
            success, frame = video_stream.read_frame()
            if not success:
                print("No more frames or camera not available.")
                break

            processed = service.process_frame(frame)

            if save_output:
                if writer is None:
                    h, w = processed.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

                writer.write(processed)

        print(f"Processing finished. Output saved to: {output_path}")

    finally:
        video_stream.release()
        if writer is not None:
            writer.release()