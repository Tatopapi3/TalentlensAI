import sqlite3
import uuid
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'predictions.db')


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id         TEXT PRIMARY KEY,
        created_at TEXT,
        model      TEXT,
        inputs     TEXT,
        prediction REAL,
        actual     REAL,
        label      TEXT
    )''')
    c.commit()
    return c


def log_prediction(model: str, inputs: dict, prediction: float,
                   label: str = "", actual: Optional[float] = None) -> str:
    pred_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)', (
            pred_id, datetime.utcnow().isoformat(), model,
            json.dumps(inputs), prediction, actual, label,
        ))
    return pred_id


def log_outcome(pred_id: str, actual: float):
    with _conn() as c:
        c.execute('UPDATE predictions SET actual=? WHERE id=?', (actual, pred_id))


def get_predictions(model: Optional[str] = None, limit: int = 200) -> list:
    with _conn() as c:
        if model:
            rows = c.execute(
                'SELECT * FROM predictions WHERE model=? ORDER BY created_at DESC LIMIT ?',
                (model, limit),
            ).fetchall()
        else:
            rows = c.execute(
                'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?', (limit,)
            ).fetchall()

    cols = ['id', 'created_at', 'model', 'inputs', 'prediction', 'actual', 'label']
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d['inputs'] = json.loads(d['inputs'])
        result.append(d)
    return result
