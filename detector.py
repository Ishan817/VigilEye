"""
detector.py
-----------
Thin wrapper around Ultralytics YOLO for detection + multi-object tracking.
Returns clean Detection objects for the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List, Set

from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int
    class_id: int
    label: str
    confidence: float
    # xyxy format
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def is_person(self) -> bool:
        return self.class_id == 0


class VigilEyeDetector:
    """Loads a YOLO model once and exposes a simple `track(frame)` call."""

    # COCO id → name
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        weights: str,
        confidence: float,
        iou: float,
        target_classes: List[int],
        tracker_cfg: str,
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.target_classes: Set[int] = set(target_classes)
        self.tracker_cfg = tracker_cfg

    def track(self, frame) -> List[Detection]:
        """Run detection + tracking on a single BGR frame."""
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            persist=True,
            tracker=self.tracker_cfg,
            classes=list(self.target_classes),
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return detections

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in self.target_classes:
                continue

            track_id = int(boxes.id[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            label = self.CLASS_NAMES.get(cls_id, str(cls_id))

            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def reset(self):
        """Reset internal tracker state (call when switching video sources)."""
        self.model = YOLO(self.model.ckpt_path if hasattr(self.model, "ckpt_path") else self.model.model_name)
