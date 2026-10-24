"""Load YAML config + environment overrides."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides for database
    cfg["db"] = {
        "backend": os.getenv("DB_BACKEND", "postgresql").lower(),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "name": os.getenv("DB_NAME", "vigileye"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "sqlite_path": cfg.get("storage", {}).get("sqlite_path", "logs/vigileye.db"),
        "snapshot_dir": cfg.get("storage", {}).get("snapshot_dir", "logs/snapshots"),
    }
    return cfg
