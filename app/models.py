from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.database import Base

class VehicleLog(Base):
    __tablename__ = "vehicle_logs"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, nullable=True)
    vehicle_type = Column(String, nullable=False)
    camera_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    snapshot_path = Column(String, nullable=True)