"""Matplotlib figure generators -> PNG paths, embedded into workbooks/decks.

Covers what native Excel charts can't do well: waterfalls, alpha decomposition, colour-by-sector
risk/return bubbles, underwater drawdown, sensitivity heatmaps, and the three 3D plots. All
functions are defensive (return None on bad/empty input) so a figure failure never breaks a report.
"""
from __future__ import annotations

from pathlib import Path

NAVY = "#1F3864"
BLUE = "#2E5496"
GREEN = "#2E7D32"
RED = "#B23A1E"
AMBER = "#BF8F00"
GREY = "#9AA3B2"
_SECTOR_COLORS = ["#1F3864", "#2E7D32", "#B23A1E", "#BF8F00", "#5B2C83", "#0E7C7B", "#9AA3B2", "#C55A11"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def _guard(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001 -- a viz failure must never break a report
            print(f"  [viz] {fn.__name__} skipped: {type(exc).__name__}: {exc}")
            return None
    return wrap


@_guard
def waterfall(labels, values, out: Path, title: str = "", value_fmt="${:,.0f}") -> Path | None:
    """Bridge/waterfall: floating bars (greens up, reds down). A trailing val=None => total bar."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n = len(labels)
    cum = 0.0
    for i, (lab, val) in enumerate(zip(labels, values)):
        is_total = (i == n - 1) and (val is None)
        if is_total:
            bottom, height, color, shown = 0.0, cum, NAVY, cum
        else:
            v = float(val or 0.0)
            bottom = cum if v >= 0 else cum + v
            height = abs(v)
            color = GREEN if v >= 0 else RED
            shown = v
            cum += v
        ax.bar(i, height, bottom=bottom, color=color, edgecolor="white")
        ax.text(i, bottom + height, value_fmt.format(shown), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#444", linewidth=0.8)
    return _save(fig, out)


@_guard
def alpha_decomposition(parts: dict, out: Path, title="Alpha decomposition") -> Path | None:
    """parts: ordered {label: value}; e.g. Market / Selection / Sector -> Total."""
    labels = list(parts.keys()) + ["Total"]
    values = list(parts.values()) + [None]
    return waterfall(labels, values, out, title=title, value_fmt="{:+.1%}")


@_guard
def risk_return_bubble(points, out: Path, title="Risk / Return (bubble = weight, colour = sector)") -> Path | None:
    """points: list of dict(name, vol, ret, weight, sector)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    secs = sorted({p.get("sector", "?") for p in points})
    cmap = {s: _SECTOR_COLORS[i % len(_SECTOR_COLORS)] for i, s in enumerate(secs)}
    for p in points:
        ax.scatter(p["vol"], p["ret"], s=max(40, (p.get("weight", 0.05)) * 6000),
                   color=cmap.get(p.get("sector", "?"), GREY), alpha=0.7, edgecolor="white")
        ax.annotate(p.get("name", ""), (p["vol"], p["ret"]), fontsize=7,
                    ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlabel("Volatility (annualized)"); ax.set_ylabel("Expected return (annualized)")
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmap[s], label=s) for s in secs]
    ax.legend(handles=handles, fontsize=7, loc="best", frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, out)


@_guard
def drawdown(dates, nav, out: Path, title="Drawdown (underwater)") -> Path | None:
    plt = _mpl()
    peak, dd = -1e18, []
    for v in nav:
        peak = max(peak, v)
        dd.append(v / peak - 1.0 if peak > 0 else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(range(len(dd)), [d * 100 for d in dd], 0, color=RED, alpha=0.35)
    ax.plot(range(len(dd)), [d * 100 for d in dd], color=RED, linewidth=1.2)
    ax.set_xticks(range(0, len(dates), max(1, len(dates) // 8)))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), max(1, len(dates) // 8))], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Drawdown (%)"); ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


@_guard
def equity_curve(dates, series: dict, out: Path, title="Cumulative NAV") -> Path | None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    styles = {"portfolio": (NAVY, "-", 2.2), "SPY": (GREY, "--", 1.4), "QQQ": (AMBER, ":", 1.6)}
    for name, ys in series.items():
        c, ls, lw = styles.get(name, (BLUE, "-", 1.5))
        ax.plot(range(len(ys)), ys, label=name, color=c, linestyle=ls, linewidth=lw)
    step = max(1, len(dates) // 8)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Growth of $1"); ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    return _save(fig, out)


@_guard
def sector_bar(weights: dict, out: Path, title="Sector exposure", bench: dict | None = None) -> Path | None:
    plt = _mpl()
    secs = list(weights.keys())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = range(len(secs))
    ax.barh(list(y), [weights[s] * 100 for s in secs], color=BLUE, label="Portfolio")
    if bench:
        ax.barh([i + 0.0 for i in y], [bench.get(s, 0) * 100 for s in secs], color="none",
                edgecolor=RED, linewidth=1.3, label="Benchmark")
    ax.set_yticks(list(y)); ax.set_yticklabels(secs, fontsize=8)
    ax.set_xlabel("Weight (%)"); ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    if bench:
        ax.legend(fontsize=8, frameon=False)
    return _save(fig, out)


@_guard
def concentration(weights: list, out: Path, title="Concentration (weight by holding)") -> Path | None:
    plt = _mpl()
    ws = sorted([w * 100 for w in weights], reverse=True)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(range(len(ws)), ws, color=NAVY)
    hhi = sum((w / 100) ** 2 for w in ws)
    ax.set_xlabel("Holding (sorted)"); ax.set_ylabel("Weight (%)")
    ax.set_title(f"{title}  -  HHI={hhi:.3f}", fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


@_guard
def capacity_gauge(util_by_aum: dict, out: Path, title="Capacity utilization") -> Path | None:
    """util_by_aum: {aum_label: alpha_or_metric}."""
    plt = _mpl()
    labels = list(util_by_aum.keys())
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(range(len(labels)), [util_by_aum[k] * 100 for k in labels], marker="o", color=NAVY)
    ax.axhline(0, color=RED, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net alpha vs universe (%)"); ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


@_guard
def sensitivity_heatmap(matrix, xlabels, ylabels, out: Path, title="Sensitivity") -> Path | None:
    plt = _mpl()
    import numpy as np
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(arr, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:,.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    return _save(fig, out)


@_guard
def scatter3d(xs, ys, zs, labels, out: Path, axis=("X", "Y", "Z"), title="", colors=None) -> Path | None:
    plt = _mpl()
    fig = plt.figure(figsize=(7.8, 6.0))
    ax = fig.add_subplot(projection="3d")
    cs = colors or [NAVY] * len(xs)
    ax.scatter(xs, ys, zs, c=cs, s=60, depthshade=True, edgecolor="white")
    for x, y, z, lab in zip(xs, ys, zs, labels):
        ax.text(x, y, z, f"  {lab}", fontsize=7)
    ax.set_xlabel(axis[0]); ax.set_ylabel(axis[1]); ax.set_zlabel(axis[2])
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.view_init(elev=22, azim=-58)
    return _save(fig, out)


@_guard
def bar3d_allocation(sectors, discounts, weights, out: Path,
                     title="3D allocation (sector x discount x weight)") -> Path | None:
    """One bar per holding at (sector index, discount), height = weight."""
    plt = _mpl()
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(projection="3d")
    sec_list = sorted(set(sectors))
    xi = [sec_list.index(s) for s in sectors]
    cs = [_SECTOR_COLORS[i % len(_SECTOR_COLORS)] for i in xi]
    ax.bar3d(xi, discounts, [0] * len(weights), 0.6, 0.06, weights, color=cs, shade=True)
    ax.set_xticks(range(len(sec_list))); ax.set_xticklabels(sec_list, fontsize=7, rotation=30, ha="right")
    ax.set_xlabel("Sector"); ax.set_ylabel("Discount"); ax.set_zlabel("Weight")
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.view_init(elev=24, azim=-52)
    return _save(fig, out)


@_guard
def football_field(rows, price, out: Path, title="Valuation range") -> Path | None:
    """Horizontal valuation-range bars per scenario/engine.

    rows: list of {"label", "low", "high", "point"|"mid"?}. Each draws a bar from low to
    high with a marker at the point estimate; a vertical dashed line marks the price.
    """
    plt = _mpl()
    rows = [r for r in rows if r and r.get("low") is not None and r.get("high") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 0.7 * len(rows) + 1.6))
    for i, r in enumerate(rows):
        lo, hi = sorted((float(r["low"]), float(r["high"])))
        ax.barh(i, hi - lo, left=lo, height=0.5, color="#BBD0EA", edgecolor=NAVY)
        pt = r.get("point", r.get("mid"))
        if pt is not None:
            ax.scatter(float(pt), i, color=NAVY, zorder=5, s=42, marker="D")
        ax.text(lo, i, f"{lo:,.0f} ", va="center", ha="right", fontsize=7, color="#555")
        ax.text(hi, i, f" {hi:,.0f}", va="center", ha="left", fontsize=7, color="#555")
    if price is not None:
        ax.axvline(float(price), color=RED, linestyle="--", linewidth=1.4,
                   label=f"Price {float(price):,.0f}")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.get("label", "") for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Intrinsic value / share ($)")
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, out)


@_guard
def tornado(items, base, out: Path, title="Driver sensitivity") -> Path | None:
    """One-at-a-time driver-sensitivity bars around a base value.

    items: list of {"label", "low", "high"} -- the output value when the driver is moved
    down / up. Sorted by swing width (widest on top); ``base`` is the reference line, with
    the below-base span in red and the above-base span in green.
    """
    plt = _mpl()
    items = [d for d in items if d and d.get("low") is not None and d.get("high") is not None]
    if not items or base is None:
        return None
    base = float(base)
    items = sorted(items, key=lambda d: abs(float(d["high"]) - float(d["low"])))  # widest last
    fig, ax = plt.subplots(figsize=(8.4, 0.6 * len(items) + 1.6))
    for i, d in enumerate(items):
        left, right = sorted((float(d["low"]), float(d["high"])))
        if left < base:
            ax.barh(i, min(right, base) - left, left=left, height=0.6,
                    color="#E1A59A", edgecolor=RED)
        if right > base:
            ax.barh(i, right - max(left, base), left=max(left, base), height=0.6,
                    color="#A9C7A2", edgecolor=GREEN)
    ax.axvline(base, color=NAVY, linewidth=1.5)
    ax.text(base, len(items) - 0.4, f" base {base:,.0f}", fontsize=8, color=NAVY, va="top")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([d.get("label", "") for d in items], fontsize=9)
    ax.set_xlabel("Intrinsic value / share ($)")
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, out)
