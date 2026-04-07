# Observability Collapse in Stochastic Nonlinear Systems

**Author:** Aldo A. Aguilar Bermúdez

## Overview

This repository contains the simulation code for the paper:

> **State-Dependent Observation and Detectability Collapse in Stochastic Nonlinear Systems**
>
> Transition indicators based on critical slowing down can fail when the observation function loses sensitivity near attractor states. We identify this mechanism — *observability collapse* — and derive an analytical criterion separating collapse from non-collapse regimes in a supercritical pitchfork normal form.

## System

Supercritical pitchfork normal form under additive noise and slow parameter drift:

```
dx = (μx − x³)dt + σ dW
```

with μ drifting from 2.0 → 0.05 (approaching bifurcation at μ = 0).

**Observation family:** g_α(x) = sign(x)|x|^α

- α > 2: observability collapse (observed variance falls despite rising latent variance)
- α = 2: balanced regime (signal-bearing term constant)
- α < 2: CSD preserved (observed variance rises as expected)

## Reproducing the results

### Requirements

```bash
pip install -r requirements.txt
```

### Generate all figures

```bash
python simulation_code.py
```

This produces:
- `figure1_observability_collapse.png/pdf` — Mechanism demonstration (4 panels)
- `figure2_ablation.png/pdf` — Ablation, criterion validation, parameter sweep (4 panels)

### Simulation parameters

| Parameter | Value |
|-----------|-------|
| μ₀ | 2.0 |
| μ_end | 0.05 |
| σ (noise) | 0.3 |
| Δt | 0.01 |
| T | 30,000 steps (300 time units) |
| α (collapse) | 3 |
| α (control) | 1 |
| Rolling window | 1,500 steps |
| Ensemble (Fig 1C) | 30 runs |
| Sweep grid | 16 × 16 (α × σ) |
| Sweep ensemble | 8 runs per point |

### Key equations

**Observed variance (Eq. 4):**

Var[S] ≈ (g'(x*))² · Var[x] + σ_ε²

**Signal-bearing term for pitchfork + power-law observation:**

(g')² · Var[x] = (α² σ² / 4) · μ^(α−2)

**Collapse criterion:** α > 2 (signal-bearing term → 0 as μ → 0)

## Figure descriptions

### Figure 1: Mechanism demonstration
- **A.** Pitchfork potential V(x) = −μx²/2 + x⁴/4 with observation g(x) = |x|³
- **B.** Smoothed observable signal over time (collapse vs control)
- **C.** Eq. (4) decomposition: ensemble-averaged latent variance (rising), sensitivity (falling), observed variance (suppressed)
- **D.** Rolling variance ablation: |x|³ variance decreases while |x| variance increases

### Figure 2: Validation
- **A.** Three observation functions: x³ (collapse), tanh(2x) (nonlinear control), x (linear control)
- **B.** Criterion validation: theoretical boundary at α = 2 vs numerically observed boundary
- **C.** Rolling lag-1 autocorrelation for all three observations
- **D.** Full collapse regime: parameter sweep heatmap over (α, σ)

## Citation

```
Aguilar Bermúdez, A.A. (2026). State-Dependent Observation and Detectability
Collapse in Stochastic Nonlinear Systems. Preprint.
DOI: 10.5281/zenodo.15088837
```

## License

MIT
