from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    class_id: int | None = None
    track_id: int | None = None


@dataclass
class FrameContext:
    frame: Any
    timestamp: float
    is_static: bool
    detections: list[Detection] = field(default_factory=list)