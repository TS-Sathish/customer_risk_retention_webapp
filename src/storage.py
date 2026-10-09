"""SQLite persistence for local prediction history."""

import json
import sqlite3
from pathlib import Path

from .config import DB_PATH


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT NOT NULL,
        probability REAL NOT NULL,
        risk TEXT NOT NULL,
        segment INTEGER NOT NULL,
        explanation TEXT NOT NULL,
        narrative TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    return connection


def save_prediction(customer_id: str, probability: float, risk: str, segment: int,
                    explanation: dict, narrative: str, db_path: str | Path = DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO predictions(customer_id, probability, risk, segment, explanation, narrative) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (customer_id, probability, risk, segment, json.dumps(explanation), narrative),
        )


def recent_predictions(limit: int = 50, db_path: str | Path = DB_PATH) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["explanation"] = json.loads(item["explanation"])
        results.append(item)
    return results
