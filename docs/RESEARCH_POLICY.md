# Research Policy

**Status:** in force from the `research-complete-v1` tag (2026-06-10).
**Purpose:** keep the shipped product trustworthy by separating *product engineering*
from *signal research*, so that no exploratory finding can silently alter a live
valuation, a portfolio, or a published claim until it has survived the full validation
battery.

This policy is binding on every future change to this repository. It exists because the
forecasting research program reached a valid, documented conclusion (see
`reports/research_v2/research_journal.md`, entry *RESEARCH FREEZE*): once point-in-time
data was made split-consistent, no broad cross-sectional alpha survived, and added
statistical sophistication (v2 Bayesian, Expectation Gap, regime/sector conditioning, v3
economic drivers) did not beat the simple v1 benchmark out of sample. The product's value
proposition is therefore **rigor, transparency, and auditability — not ranking edge.**

---

## 1. Two tracks, never blurred

| Track | What it is | Where it runs | Claims it may make |
|-------|------------|---------------|--------------------|
| **Product** | The valuation engines, reporting, CLI, portfolio tooling that users touch | `main` / `master` | Economic coherence, transparency, auditability. **No predictive/alpha claim.** |
| **Research** | Hypothesis tests on signals, factors, regimes, engines | `research/*` branches only | Only what the validation battery in §3 supports, scoped honestly |

A change is **research** the moment its justification is "this should predict / rank /
outperform better." Such changes do not land on the mainline until §4 is satisfied.

## 2. What is frozen (do not modify outside a validated promotion)

- **v1 engine** (`pipeline.py`) — byte-identical; the v1 golden is the lock. It is the
  default engine everywhere and the benchmark every experiment is paired against.
- **Live paper portfolio** `reports/managed_portfolios/conf_top10/` — the confidence-
  weighted long-short book is a genuine out-of-sample test. Only user-run `manage`
  commands may write its state. No test, demo, script, or research run may mutate it.
  Next OOS information event: the **2026-09-04** rebalance.
- **Portfolio signal math** (`portfolio/scoring.py`, `portfolio/signals.py`) — the live
  signal is frozen; display layers may relabel but must not change the computation.
- **Sector-state tilt** κ = 0.0 — sector fields are display-only until an ablation earns a
  non-zero κ under §3.

## 3. Validation battery (required before any signal claim)

Every predictive claim must clear, in order:

1. **Pre-registration.** Write the hypothesis, sample, metric, and *kill thresholds* into
   the research journal **before** seeing the result. Gates are numbered (G0…) with
   explicit PASS / KILL criteria.
2. **Split-consistent PIT data.** Shares re-expressed on today's basis via whitelisted
   cumulative-split factors; Close basis for the discount, Adj Close for forward returns.
   This is the defect that overturned the original program — it is non-negotiable.
3. **Paired in-process v1 comparison.** The experimental engine and v1 are valued on the
   *identical* bundles; the only admissible comparison is the paired per-date IC
   difference, not two separately-run panels.
4. **Overlap-aware inference.** Newey-West t-stats (lag = horizon in quarters) and/or block
   bootstrap. Raw IC t-stats on overlapping windows are not evidence.
5. **Thin-cross-section guards.** Per-date cross-sections below the floor (≈4 names) are
   reported but excluded from headline conclusions; a high IC on 3 names/date is an
   artifact, not a finding.
6. **Corrected panel only.** Headline numbers come from the split-corrected panels.
   Pre-correction numbers may appear only inside a discovery narrative, captioned
   *superseded*.

A claim "**proves predictive**" only if: it ranks at least as well as paired v1 in the
families where DCF is known reliable; it has a broad bootstrap CI excluding zero **or** an
honestly scoped (e.g. single-family) claim; positive factor-residual IC where breadth is
claimed (B/M control); and — if uncertainty is asserted — informative bands (width must
rank realized error). The v3 MC layer failed this last test (G6) and carries **no
calibration claim**.

## 4. Promotion rule

No engine, signal, conditioner, or factor is promoted from `research/*` to the product
default until it has **passed the full §3 battery on the broad universe** and the journal
records the gate ledger. Absent that, it may ship only as an **opt-in, clearly-labelled
diagnostic** that makes no predictive claim (this is exactly the status of engine v3).

## 5. Product-track engine changes (the narrow, sanctioned exception)

The product may evolve an engine's *economics and presentation* for interpretability,
coherence, or audit value **without** a predictive claim — e.g. the Fundamentals Fade
flagship's per-driver half-lives, the 10-year explicit horizon, and Monte Carlo defaulted
off. Such changes:

- are allowed on the mainline,
- must be documented in the commit message and a journal product-note,
- regenerate any affected **non-v1** golden once, with the change recorded,
- must **never** touch the v1 golden or the live-book signal, and
- carry **no** validation, ranking, or alpha claim — only "more economically legible."

If a product-track change is ever proposed *because it should predict better*, it is
research (§1) and must go through §3–§4 instead.

## 6. Journal discipline

`reports/research_v2/research_journal.md` and its `.jsonl` mirror are **append-only**.
Every research action — and every sanctioned product-track engine change — gets an entry
with the schema: title · hypothesis · sample_size · engine · command · output_artifact ·
metrics · interpretation · decision · next_action · datetime. Convert relative dates to
absolute. The `.jsonl` must stay a faithful mirror of the `.md`.

## 7. Paper / external claims

The CFA / FAJ manuscript and any external communication report **corrected numbers only**.
The split-correction discovery is presented as a methods contribution; pre-correction
figures appear solely inside the discovery narrative, captioned superseded. No probability
or coverage language may be attached to the driver-sensitivity bands.
