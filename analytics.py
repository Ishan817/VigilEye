"""
analytics.py
------------
Core intelligence of VigilEye — built on top of raw detections:

  - Zone definitions (polygons) + point-in-zone checks
  - Restricted-zone intrusion detection
  - Crowd density / people-count alerting
  - Loitering detection (object dwelling in a zone past a time threshold)

Outputs a list of Alert objects for the dashboard + database layer.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from shapely.geometry import Point, Polygon

from .detector import Detection


@dataclass
class Zone:
    name: str
    polygon: Polygon
    zone_type: str  # "restricted" or "monitor"


@dataclass
class Alert:
    kind: str          # "intrusion" | "crowd" | "loitering"
    message: str
    zone: str
    track_id: int = -1
    timestamp: float = field(default_factory=time.time)


class AnalyticsEngine:
    """Stateful engine — tracks how long each object has been in each zone."""

    def __init__(
        self,
        zones_cfg: List[dict],
        frame_w: int,
        frame_h: int,
        loitering_seconds: float,
        crowd_alert_threshold: int,
        intrusion_classes: List[int],
    ):
        self.zones: List[Zone] = []
        for z in zones_cfg:
            pts = [(px * frame_w, py * frame_h) for px, py in z["points"]]
            self.zones.append(
                Zone(name=z["name"], polygon=Polygon(pts), zone_type=z["type"])
            )

        self.loitering_seconds = loitering_seconds
        self.crowd_alert_threshold = crowd_alert_threshold
        self.intrusion_classes = set(intrusion_classes)

        # track_id → {zone_name: first_seen_timestamp}
        self._zone_entry_time: Dict[int, Dict[str, float]] = {}
        self._loitering_alerted: Set[Tuple[int, str]] = set()
        self._intrusion_alerted: Set[Tuple[int, str]] = set()
        self._crowd_alerted = False
        self._last_crowd_time = 0.0

    def update(self, detections: List[Detection]) -> List[Alert]:
        """Process one frame of detections and return any new alerts."""
        now = time.time()
        alerts: List[Alert] = []
        people_count = sum(1 for d in detections if d.is_person)

        # ---------- Crowd alert ----------
        if people_count >= self.crowd_alert_threshold:
            # Cooldown of 10 s to avoid spam
            if not self._crowd_alerted or (now - self._last_crowd_time) > 10:
                alerts.append(
                    Alert(
                        kind="crowd",
                        message=f"Crowd detected: {people_count} people in frame",
                        zone="global",
                    )
                )
                self._crowd_alerted = True
                self._last_crowd_time = now
        else:
            self._crowd_alerted = False

        # ---------- Zone-based logic ----------
        active_tracks_in_zones: Dict[int, List[str]] = {}

        for det in detections:
            pt = Point(det.cx, det.cy)
            for zone in self.zones:
                if zone.polygon.contains(pt):
                    active_tracks_in_zones.setdefault(det.track_id, []).append(zone.name)

                    # Record entry time
                    if det.track_id not in self._zone_entry_time:
                        self._zone_entry_time[det.track_id] = {}
                    if zone.name not in self._zone_entry_time[det.track_id]:
                        self._zone_entry_time[det.track_id][zone.name] = now

                    # Intrusion (restricted zones)
                    if (
                        zone.zone_type == "restricted"
                        and det.class_id in self.intrusion_classes
                    ):
                        key = (det.track_id, zone.name)
                        if key not in self._intrusion_alerted:
                            self._intrusion_alerted.add(key)
                            alerts.append(
                                Alert(
                                    kind="intrusion",
                                    message=f"{det.label} (ID {det.track_id}) entered '{zone.name}'",
                                    zone=zone.name,
                                    track_id=det.track_id,
                                )
                            )

                    # Loitering
                    entry_t = self._zone_entry_time[det.track_id].get(zone.name, now)
                    dwell = now - entry_t
                    key = (det.track_id, zone.name)
                    if dwell >= self.loitering_seconds and key not in self._loitering_alerted:
                        self._loitering_alerted.add(key)
                        alerts.append(
                            Alert(
                                kind="loitering",
                                message=(
                                    f"{det.label} (ID {det.track_id}) loitering in "
                                    f"'{zone.name}' for {dwell:.0f}s"
                                ),
                                zone=zone.name,
                                track_id=det.track_id,
                            )
                        )

        # Cleanup tracks that left all zones
        current_ids = {d.track_id for d in detections}
        for tid in list(self._zone_entry_time.keys()):
            if tid not in current_ids:
                del self._zone_entry_time[tid]

        return alerts

    def reset(self):
        self._zone_entry_time.clear()
        self._loitering_alerted.clear()
        self._intrusion_alerted.clear()
        self._crowd_alerted = False
