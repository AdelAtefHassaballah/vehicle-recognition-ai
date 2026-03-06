from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

@dataclass
class PlateResult:
    plate_text: Optional[str]
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]