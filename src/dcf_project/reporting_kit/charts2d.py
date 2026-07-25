"""Native (editable) openpyxl charts + conditional-format heatmap.

These are real Excel charts the user can re-style/re-data in Excel, used for the spec's core
charts (equity line, sector bar, risk/return bubble). True 3D and waterfalls go through viz3d
as embedded images.
"""
from __future__ import annotations

from openpyxl.chart import BarChart, BubbleChart, LineChart, Reference, Series
from openpyxl.formatting.rule import ColorScaleRule

_NAVY = "1F3864"


def line_chart(ws, anchor, title, min_col, max_col, min_row, max_row, cats_col,
               x_title=None, y_title=None):
    ch = LineChart()
    ch.title = title
    ch.height, ch.width = 8.5, 17
    ch.y_axis.majorGridlines = None
    data = Reference(ws, min_col=min_col, max_col=max_col, min_row=min_row - 1, max_row=max_row)
    cats = Reference(ws, min_col=cats_col, min_row=min_row, max_row=max_row)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    for s in ch.series:
        s.smooth = False
    if x_title:
        ch.x_axis.title = x_title
    if y_title:
        ch.y_axis.title = y_title
    # openpyxl hides axis titles unless the axis is marked delete=False.
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ws.add_chart(ch, anchor)
    return ch


def bar_chart(ws, anchor, title, min_col, max_col, min_row, max_row, cats_col, horizontal=False,
              x_title=None, y_title=None):
    ch = BarChart()
    ch.type = "bar" if horizontal else "col"
    ch.title = title
    ch.height, ch.width = 8.5, 16
    data = Reference(ws, min_col=min_col, max_col=max_col, min_row=min_row - 1, max_row=max_row)
    cats = Reference(ws, min_col=cats_col, min_row=min_row, max_row=max_row)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.legend = None if (max_col == min_col) else ch.legend
    if x_title:
        ch.x_axis.title = x_title
    if y_title:
        ch.y_axis.title = y_title
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ws.add_chart(ch, anchor)
    return ch


def bubble_chart(ws, anchor, title, x_col, y_col, size_col, min_row, max_row):
    ch = BubbleChart()
    ch.title = title
    ch.height, ch.width = 9, 16
    xref = Reference(ws, min_col=x_col, min_row=min_row, max_row=max_row)
    yref = Reference(ws, min_col=y_col, min_row=min_row, max_row=max_row)
    sref = Reference(ws, min_col=size_col, min_row=min_row, max_row=max_row)
    ch.series.append(Series(values=yref, xvalues=xref, zvalues=sref, title="Holdings"))
    ch.x_axis.title = "Volatility"
    ch.y_axis.title = "Expected return"
    ws.add_chart(ch, anchor)
    return ch


def heatmap(ws, cell_range: str):
    """Red-white-green colour scale over a numeric cell range (e.g. 'B5:F10')."""
    ws.conditional_formatting.add(
        cell_range,
        ColorScaleRule(start_type="min", start_color="F8696B",
                       mid_type="percentile", mid_value=50, mid_color="FFEB84",
                       end_type="max", end_color="63BE7B"),
    )
