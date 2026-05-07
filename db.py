"""SQLite helpers for auth and per-user history."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def _conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    conn = _conn(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                text_input TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_user(username: str, password_hash: str, db_path: str = DB_PATH) -> bool:
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username: str, db_path: str = DB_PATH) -> Optional[dict]:
    conn = _conn(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_history(username: str, text_input: str, prediction_label: str, confidence_score: float, db_path: str = DB_PATH) -> None:
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO history (username, text_input, prediction_label, confidence_score, timestamp) VALUES (?, ?, ?, ?, ?)",
            (username, text_input, prediction_label, confidence_score, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(username: str, db_path: str = DB_PATH, limit: int = 100) -> list[dict]:
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, username, text_input, prediction_label, confidence_score, timestamp FROM history WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
