from datetime import datetime, timedelta

class SimplePlateTracker:
    def __init__(self, cooldown_seconds: int = 10):
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.last_seen = {}

    def should_log(self, plate_text: str | None, vehicle_type: str) -> bool:
        key = plate_text if plate_text else f"unknown_{vehicle_type}"
        now = datetime.utcnow()

        if key not in self.last_seen:
            self.last_seen[key] = now
            return True

        if now - self.last_seen[key] > self.cooldown:
            self.last_seen[key] = now
            return True

        return False