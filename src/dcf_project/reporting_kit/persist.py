"""Persist a built ReportModel next to its outputs so `export pdf|ppt` reuses the exact model.

A folder can hold more than one model slot (`stem`): the default slot is the book's primary
product (e.g. the India pitch package), while `.report_model_dashboard.pkl` holds the live PM
dashboard for books whose default slot is owned by a dedicated pitch exporter. Last-writer-wins
inside one slot is intended; ACROSS slots nothing clobbers.
"""
from __future__ import annotations

import pickle
from pathlib import Path

MODEL_FILE = ".report_model.pkl"
DASHBOARD_MODEL_FILE = ".report_model_dashboard.pkl"


def save_model(model, folder, stem: str = MODEL_FILE) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / stem
    p.write_bytes(pickle.dumps(model))
    return p


def load_model(folder, stem: str = MODEL_FILE):
    p = Path(folder) / stem
    if not p.exists():
        return None
    return pickle.loads(p.read_bytes())
