"""
VigilEye — Smart CCTV Analytics Platform
============================================
Real-time detection + tracking + zone intrusion / crowd / loitering analytics.
Supports Webcam, Video Upload, and RTSP/IP cameras.
Storage: PostgreSQL (preferred) with SQLite fallback.

Run:  streamlit run app.py
"""

import os
import tempfile
import time
from collections import Counter
from typing import Optional

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import AnalyticsEngine
from src.config_loader import load_config
from src.database import get_database
from src.detector import VigilEyeDetector
from src.visualize import draw_detections, draw_zones, draw_stats

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="VigilEye | Smart CCTV Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

CONFIG = load_config("config/config.yaml")


# ──────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model...")
def get_detector():
    m = CONFIG["model"]
    return VigilEyeDetector(
        weights=m["weights"],
        confidence=m["confidence"],
        iou=m["iou"],
        target_classes=m["target_classes"],
        tracker_cfg=CONFIG["tracker"]["type"],
    )


@st.cache_resource(show_spinner=False)
def get_db():
    return get_database(CONFIG)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown("## 👁️ VigilEye")
    st.sidebar.caption("Smart CCTV Analytics Platform")
    st.sidebar.divider()

    source_type = st.sidebar.radio(
        "Video source",
        ["Upload video", "Webcam", "RTSP / IP Camera"],
        index=0,
    )

    uploaded_file = None
    rtsp_url = None

    if source_type == "Upload video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"]
        )
    elif source_type == "RTSP / IP Camera":
        rtsp_url = st.sidebar.text_input(
            "RTSP URL",
            placeholder="rtsp://user:pass@192.168.1.10:554/stream",
        )

    st.sidebar.divider()
    st.sidebar.subheader("Live thresholds")
    loiter_s = st.sidebar.slider(
        "Loitering threshold (s)", 5, 60, CONFIG["analytics"]["loitering_seconds"]
    )
    crowd_n = st.sidebar.slider(
        "Crowd alert threshold", 2, 30, CONFIG["analytics"]["crowd_alert_threshold"]
    )
    CONFIG["analytics"]["loitering_seconds"] = loiter_s
    CONFIG["analytics"]["crowd_alert_threshold"] = crowd_n

    st.sidebar.divider()
    conf = st.sidebar.slider("Confidence", 0.1, 0.9, float(CONFIG["model"]["confidence"]), 0.05)
    CONFIG["model"]["confidence"] = conf

    st.sidebar.divider()
    start = st.sidebar.button("▶ Start Analysis", type="primary", use_container_width=True)
    stop = st.sidebar.button("⏹ Stop", use_container_width=True)

    st.sidebar.divider()
    db = get_db()
    backend = CONFIG["db"]["backend"]
    if hasattr(db, "conn") and db.conn is not None:
        st.sidebar.success(f"DB: PostgreSQL")
    else:
        st.sidebar.info(f"DB: SQLite (fallback)")

    st.sidebar.caption("VigilEye · 2026")
    return source_type, uploaded_file, rtsp_url, start, stop


# ──────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────
def run_pipeline(cap, source_name: str):
    detector = get_detector()
    detector.confidence = CONFIG["model"]["confidence"]
    db = get_db()

    # Read first frame to know resolution for zones
    ret, frame = cap.read()
    if not ret:
        st.error("Could not read from source.")
        return
    h, w = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    analytics = AnalyticsEngine(
        zones_cfg=CONFIG["zones"],
        frame_w=w,
        frame_h=h,
        loitering_seconds=CONFIG["analytics"]["loitering_seconds"],
        crowd_alert_threshold=CONFIG["analytics"]["crowd_alert_threshold"],
        intrusion_classes=CONFIG["analytics"]["intrusion_classes"],
    )

    video_ph = st.empty()
    status_ph = st.empty()
    metrics_ph = st.empty()
    chart_ph = st.empty()
    table_ph = st.empty()

    all_alerts = []
    frame_idx = 0
    active_track_ids = set()
    status_ph.info(f"🔴 Live · {source_name}")

    try:
        while st.session_state.get("running", True) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            detections = detector.track(frame)
            alerts = analytics.update(detections)

            if alerts:
                db.insert_many(alerts)
                all_alerts.extend(alerts)

            flagged = {a.track_id for a in alerts if a.track_id >= 0}
            people = sum(1 for d in detections if d.is_person)
            vehicles = len(detections) - people
            active_track_ids.update(d.track_id for d in detections)

            # Draw
            frame = draw_zones(frame, analytics.zones)
            frame = draw_detections(frame, detections, flagged)
            frame = draw_stats(frame, people, vehicles, len(active_track_ids), len(all_alerts))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_ph.image(rgb, channels="RGB", use_container_width=True)

            # Metrics
            c1, c2, c3, c4 = metrics_ph.columns(4)
            c1.metric("People", people)
            c2.metric("Vehicles", vehicles)
            c3.metric("Unique Tracks", len(active_track_ids))
            c4.metric("Total Alerts", len(all_alerts))

            # Alert breakdown chart
            if all_alerts:
                kinds = Counter(a.kind for a in all_alerts)
                df_chart = pd.DataFrame(
                    {"Kind": list(kinds.keys()), "Count": list(kinds.values())}
                )
                fig = px.bar(
                    df_chart, x="Kind", y="Count", color="Kind",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    height=220, margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
                )
                chart_ph.plotly_chart(fig, use_container_width=True, key=f"alerts_chart_{frame_idx}")

            # Recent alerts table
            recent = all_alerts[-12:][::-1]
            if recent:
                table_ph.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Type": a.kind,
                                "Message": a.message,
                                "Zone": a.zone,
                                "Track": a.track_id if a.track_id >= 0 else "—",
                            }
                            for a in recent
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            time.sleep(0.01)

    finally:
        cap.release()
        status_ph.success(f"✅ Session finished · {frame_idx} frames · {len(all_alerts)} alerts")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    st.markdown('<p class="main-title">👁️ VigilEye</p>', unsafe_allow_html=True)
    st.caption("Smart CCTV Analytics — Detection · Tracking · Intrusion · Crowd · Loitering")

    source_type, uploaded_file, rtsp_url, start, stop = sidebar()

    if "running" not in st.session_state:
        st.session_state.running = False

    if stop:
        st.session_state.running = False

    if start:
        st.session_state.running = True

        if source_type == "Upload video":
            if uploaded_file is None:
                st.warning("Please upload a video first.")
                return
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            run_pipeline(cap, uploaded_file.name)
            os.unlink(tfile.name)

        elif source_type == "Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
                return
            run_pipeline(cap, "Webcam-0")

        elif source_type == "RTSP / IP Camera":
            if not rtsp_url:
                st.warning("Enter a valid RTSP URL.")
                return
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.error("Cannot connect to RTSP stream.")
                return
            run_pipeline(cap, "RTSP Stream")

    else:
        st.info("👈 Configure source & thresholds in the sidebar, then click **Start Analysis**.")
        with st.expander("Features"):
            st.markdown(
                """
- **YOLOv8 + ByteTrack** real-time detection & multi-object tracking  
- **Zone intrusion** alerts (restricted polygons)  
- **Crowd density** alerts  
- **Loitering** detection  
- **PostgreSQL** storage (SQLite fallback)  
- Webcam · Video upload · RTSP / IP cameras  
                """
            )


if __name__ == "__main__":
    main()
