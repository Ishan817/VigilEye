"""
visualize.py
------------
Draws bounding boxes, track IDs, and zone overlays onto frames.
"""

from typing import List, Set

import cv2
import numpy as np

from .detector import Detection

ZONE_COLORS = {
    "restricted": (0, 0, 255),   # red
    "monitor": (255, 191, 0),    # amber
}

BOX_COLOR = (0, 200, 0)
ALERT_BOX_COLOR = (0, 0, 255)


def draw_zones(frame: np.ndarray, zones: List) -> np.ndarray:
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone.polygon.exterior.coords, dtype=np.int32)
        color = ZONE_COLORS.get(zone.zone_type, (200, 200, 200))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(pts[0][0]), int(pts[0][1])
        cv2.putText(
            frame,
            zone.name,
            (x + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    flagged_ids: Set[int] = None,
) -> np.ndarray:
    flagged_ids = flagged_ids or set()
    for det in detections:
        color = ALERT_BOX_COLOR if det.track_id in flagged_ids else BOX_COLOR
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{det.track_id} {det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_stats(
    frame: np.ndarray,
    people: int,
    vehicles: int,
    active_tracks: int,
    alert_count: int,
) -> np.ndarray:
    lines = [
        f"People: {people}",
        f"Vehicles: {vehicles}",
        f"Tracks: {active_tracks}",
        f"Alerts: {alert_count}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, y0 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame
