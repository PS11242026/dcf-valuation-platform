"""Distribution outputs + candidate signals for engine_v2 (U4, U5-signals).

Turns the Monte Carlo intrinsic-value sample into the ``monte_carlo_distribution`` payload
(percentiles, mode, histogram, fan charts, state trajectories, terminal/parameter/convergence
diagnostics) and the distribution-aware candidate signals.
"""
from __future__ import annotations

import numpy as np

from dcf_project.engine_v2.posterior import PosteriorDraws
from dcf_project.engine_v2.posterior_predictive import SimResult
from dcf_project.engine_v2.terminal import equilibrium_state


def _mode(samples: np.ndarray, bins: int = 50) -> float:
    counts, edges = np.histogram(samples, bins=bins)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def _fan(matrix: np.ndarray) -> dict:
    return {
        "p5": np.percentile(matrix, 5, axis=0).tolist(),
        "p50": np.percentile(matrix, 50, axis=0).tolist(),
        "p95": np.percentile(matrix, 95, axis=0).tolist(),
    }


def distribution_outputs(iv: np.ndarray, price: float | None) -> dict:
    iv = np.asarray(iv, dtype=float)
    iv = iv[np.isfinite(iv)]
    mean = float(np.mean(iv))
    std = float(np.std(iv, ddof=1)) if iv.size > 1 else 0.0
    out = {
        "expected": mean,
        "median": float(np.median(iv)),
        "mode": _mode(iv) if iv.size else mean,
        "std": std,
        "p5": _pct(iv, 5),
        "p25": _pct(iv, 25),
        "p50": _pct(iv, 50),
        "p75": _pct(iv, 75),
        "p95": _pct(iv, 95),
    }
    if price and price > 0:
        out["prob_value_gt_price"] = float(np.mean(iv > price))
        out["prob_value_gt_1_2_price"] = float(np.mean(iv > 1.2 * price))
    return out


def candidate_signals(iv: np.ndarray, price: float | None) -> dict:
    iv = np.asarray(iv, dtype=float)
    iv = iv[np.isfinite(iv)]
    mean = float(np.mean(iv))
    std = float(np.std(iv, ddof=1)) if iv.size > 1 else 0.0
    conf = float(np.clip(1.0 - std / abs(mean), 0.0, 1.0)) if mean != 0 else 0.0
    sig = {"mc_distribution_confidence": conf}
    if price and price > 0:
        sig["mc_expected_discount"] = mean / price - 1.0
        sig["mc_probability_undervalued"] = float(np.mean(iv > price))
        sig["mc_margin_of_safety_probability"] = float(np.mean(iv > 1.2 * price))
    return sig


def build_distribution_payload(
    sim: SimResult,
    posterior: PosteriorDraws,
    model,
    price: float | None,
    histogram_bins: int = 50,
) -> dict:
    iv = sim.iv[np.isfinite(sim.iv)]
    counts, edges = np.histogram(iv, bins=histogram_bins)
    dist = distribution_outputs(sim.iv, price)
    return {
        "engine": "v2",
        "n_paths": int(sim.n_paths),
        "seed": int(sim.seed),
        "iv": dist,
        "signals": candidate_signals(sim.iv, price),
        "histogram": {"bin_edges": edges.tolist(), "counts": counts.tolist()},
        "fan_chart": {
            "revenue": _fan(sim.revenue),
            "ebit": _fan(sim.ebit),
            "fcf": _fan(sim.fcf),
        },
        "state_trajectories": {
            col: _fan(sim.states[:, :, i])
            for i, col in enumerate(posterior.state_columns)
        },
        "terminal_state": {
            **equilibrium_state(model.mu, model.phi),
            "equilibrium_g": {
                "mean": float(np.mean(sim.equilibrium_g)),
                "p5": _pct(sim.equilibrium_g, 5),
                "p95": _pct(sim.equilibrium_g, 95),
            },
        },
        "param_posterior": posterior.summary(),
        "convergence": posterior.diagnostics,
    }
