"""Local work database (SQLite): photos seen, tags, faces with embeddings, people (clusters).

Face embeddings are biometric data. Keep SPT_DATA_DIR private (mode 700) and local.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos(
    id INTEGER PRIMARY KEY, filename TEXT, folder TEXT, analyzed INT DEFAULT 0, written INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS tags(id_unit INT, tag TEXT, score REAL, PRIMARY KEY(id_unit, tag));
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY AUTOINCREMENT, id_unit INT, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    landmarks TEXT, score REAL, embedding BLOB, cluster INT, written INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS faces_unit ON faces(id_unit);
CREATE TABLE IF NOT EXISTS cluster_names(cluster INT PRIMARY KEY, name TEXT, votes INT, matched INT);
"""


def connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_dir.chmod(0o700)
    except OSError:
        pass
    db = sqlite3.connect(data_dir / "work.sqlite", timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db
