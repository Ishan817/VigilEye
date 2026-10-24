"""
database.py
-----------
Unified persistence layer for VigilEye.
Supports both PostgreSQL (production) and SQLite (fallback / offline).
Stores alerts + optional detection events.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from .analytics import Alert


class BaseDatabase(ABC):
    @abstractmethod
    def insert_alert(self, alert: Alert) -> None: ...

    @abstractmethod
    def insert_many(self, alerts: List[Alert]) -> None: ...

    @abstractmethod
    def fetch_recent(self, limit: int = 50) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def alert_counts_by_kind(self) -> Dict[str, int]: ...

    @abstractmethod
    def close(self) -> None: ...


# ───────────────────────────── SQLite ─────────────────────────────

class SQLiteDatabase(BaseDatabase):
    def __init__(self, db_path: str):
        import sqlite3
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._sqlite3 = sqlite3
        self._init_schema()

    def _connect(self):
        return self._sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    zone TEXT,
                    track_id INTEGER,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def insert_alert(self, alert: Alert):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alerts (kind, message, zone, track_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    alert.kind,
                    alert.message,
                    alert.zone,
                    alert.track_id,
                    datetime.fromtimestamp(alert.timestamp).isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

    def insert_many(self, alerts: List[Alert]):
        for a in alerts:
            self.insert_alert(a)

    def fetch_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = self._sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def alert_counts_by_kind(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) as cnt FROM alerts GROUP BY kind"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def close(self):
        pass


# ───────────────────────────── PostgreSQL ─────────────────────────────

class PostgresDatabase(BaseDatabase):
    def __init__(self, host: str, port: int, name: str, user: str, password: str):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        self._psycopg2 = psycopg2
        self._RealDictCursor = RealDictCursor
        self.conn = None
        try:
            self.conn = psycopg2.connect(
                host=host, port=port, dbname=name, user=user, password=password
            )
            self.conn.autocommit = True
            self._init_schema()
            print("[DB] Connected to PostgreSQL")
        except Exception as e:
            print(f"[DB] PostgreSQL connection failed: {e}")
            print("[DB] Falling back to SQLite is recommended via DB_BACKEND=sqlite")
            raise

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    kind VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    zone VARCHAR(100),
                    track_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER,
                    frame_number INTEGER,
                    track_id INTEGER,
                    class_id INTEGER,
                    class_name VARCHAR(50),
                    confidence REAL,
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    is_person BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_kind ON alerts(kind);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
            """)

    def insert_alert(self, alert: Alert):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alerts (kind, message, zone, track_id, timestamp)
                VALUES (%s, %s, %s, %s, to_timestamp(%s))
                """,
                (
                    alert.kind,
                    alert.message,
                    alert.zone,
                    alert.track_id,
                    alert.timestamp,
                ),
            )

    def insert_many(self, alerts: List[Alert]):
        for a in alerts:
            self.insert_alert(a)

    def fetch_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT %s", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    def alert_counts_by_kind(self) -> Dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT kind, COUNT(*) FROM alerts GROUP BY kind")
            return {row[0]: row[1] for row in cur.fetchall()}

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# ───────────────────────────── Factory ─────────────────────────────

def get_database(cfg: dict) -> BaseDatabase:
    """
    Return the appropriate database backend.
    Falls back to SQLite if PostgreSQL is requested but unavailable.
    """
    db_cfg = cfg.get("db", {})
    backend = db_cfg.get("backend", "postgresql").lower()

    if backend == "postgresql":
        try:
            return PostgresDatabase(
                host=db_cfg["host"],
                port=db_cfg["port"],
                name=db_cfg["name"],
                user=db_cfg["user"],
                password=db_cfg["password"],
            )
        except Exception:
            print("[DB] Switching to SQLite fallback")
            return SQLiteDatabase(db_cfg.get("sqlite_path", "logs/vigileye.db"))

    return SQLiteDatabase(db_cfg.get("sqlite_path", "logs/vigileye.db"))
