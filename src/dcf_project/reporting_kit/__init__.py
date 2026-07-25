"""Reporting kit: write report content once (ReportModel) -> render to xlsx / pdf / pptx.

Excel (xlsx) is the primary product; pdf/pptx are on-demand secondary renders of the same model.
"""
from __future__ import annotations

from . import charts2d, viz3d, xlsx
from .model import Image, Metric, NativeChart, ReportModel, Section, Table
from .persist import load_model, save_model
from .templates import new_report

__all__ = [
    "ReportModel", "Section", "Metric", "Table", "NativeChart", "Image",
    "new_report", "save_model", "load_model", "xlsx", "viz3d", "charts2d",
]
