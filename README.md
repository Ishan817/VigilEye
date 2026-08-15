# 👁️ VigilEye — Smart CCTV Analytics Platform

Real-time multi-object detection & tracking with **intelligent analytics**:
zone intrusion, crowd density, and loitering detection.

**Stack:** Python · YOLOv8 (Ultralytics) · ByteTrack · Streamlit · PostgreSQL / SQLite · Shapely

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue?logo=postgresql)

---

## ✨ Features

| Feature                    | Description                                              |
|---------------------------|----------------------------------------------------------|
| Real-time Detection       | YOLOv8 + ByteTrack for people & vehicles                 |
| Zone Intrusion            | Restricted polygon zones → instant intrusion alerts      |
| Crowd Density             | Configurable people-count threshold alerts               |
| Loitering Detection       | Flags objects dwelling in a zone beyond time threshold   |
| Multi-source Input        | Webcam · Video upload · RTSP / IP cameras                |
| Dual Database             | PostgreSQL (production) + SQLite fallback                |
| Live Dashboard            | Annotated feed, metrics, alert charts & table            |

---

## 🏗️ Project Structure

```
VigilEye/
├── app.py                  # Streamlit entry point
├── config/
│   └── config.yaml         # Model, zones, thresholds
├── src/
│   ├── detector.py         # YOLO + tracking wrapper
│   ├── analytics.py        # Intrusion / crowd / loitering logic
│   ├── database.py         # PostgreSQL + SQLite layer
│   ├── visualize.py        # Drawing helpers
│   └── config_loader.py
├── logs/                   # Runtime DB + snapshots
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone / extract
cd VigilEye

# 2. Virtualenv + install
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) PostgreSQL
createdb vigileye
cp .env.example .env
# Edit .env with your DB credentials

# 4. Run
streamlit run app.py
```

> If PostgreSQL is not available the app automatically falls back to SQLite.

---

## ⚙️ Configuration

All tunables live in `config/config.yaml`:

- Model confidence / IoU / target classes
- Restricted & monitored zone polygons (normalized 0-1 coordinates)
- Loitering time threshold & crowd-count threshold

Database settings are controlled via `.env` (`DB_BACKEND=postgresql` or `sqlite`).

---

## 📄 Resume Bullet (ready to use)

```
VigilEye: Smart CCTV Analytics Platform | Python, YOLO, Streamlit, PostgreSQL | 2026

• Built a real-time multi-object detection & tracking platform (people + vehicles) 
  using YOLOv8 + ByteTrack with an interactive Streamlit dashboard.
• Implemented intelligent analytics: zone-based intrusion detection, crowd density 
  alerts, and loitering detection using Shapely geometry.
• Designed dual database layer (PostgreSQL primary + SQLite fallback) for scalable 
  alert & event storage; supported Webcam, video upload and RTSP camera feeds.
```

---

## 🛠️ Tech Stack

- **Detection / Tracking:** Ultralytics YOLOv8 + ByteTrack
- **Geometry / Zones:** Shapely
- **Dashboard:** Streamlit + Plotly
- **Storage:** PostgreSQL (psycopg2) + SQLite
- **CV:** OpenCV

---

## License

MIT
