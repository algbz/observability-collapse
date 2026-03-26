# Observability Collapse: A Mechanism for the Failure of Early Warning Signals

**Author:** Aldo A. Aguilar B.

## Overview

This repository contains the simulation code and analysis scripts for the paper:

> **Observability Collapse: A Mechanism for the Failure of Early Warning Signals**
>
> Early warning signals based on critical slowing down are widely used to anticipate transitions in complex systems. We propose a dynamical mechanism — *observability collapse* — by which these signals may be suppressed or reversed when the observation function is state-dependent and vanishes near attractor states.

## Reproducing the results

### Requirements

```bash
pip install -r requirements.txt
```

### Figure 1 (main paper figure)

```bash
python figure1_code.py
```

Generates `figure1_observability_collapse.png` and `.pdf`. This is a self-contained script that runs the simulation, computes rolling variance under two observation functions (state-dependent and control), and produces all four panels.

**Simulation parameters:**
- θ₀ = 75°, N₀ = 0.45, dN/dt = −10⁻⁵ per step
- k = 0.1, D = 15, dt = 0.05, T = 30,000 steps
- Random seed: 123 (deterministic output)

### Empirical analysis (openESM)

```bash
python analysis_openesm.py
```

Downloads ESM datasets from [openesmdata.org](https://openesmdata.org), identifies behavioral transitions, and tests whether pre-transition variance decreases (observability collapse prediction) or increases (standard critical slowing down prediction). Results reported in Section 5 of the paper.

This analysis is exploratory and does not constitute a direct test of the mechanism, which requires state-dependent observability of latent variables not available in the current dataset.

**Note:** Requires internet access to download datasets on first run.

## Key equations

**State dynamics:**

dθ/dt = −dV(θ)/dθ + η(t)

**Observation function (state-dependent):**

S(t) = g(θ(t)), where g(θ*) = 0 at attractor states

**Variance propagation:**

Var[S] ≈ (g')² · Var[θ]

When g' → 0 near attractors, observed variance decreases even if latent variance increases.

## Citation

If you use this code, please cite:

```
Aguilar B., A.A. (2026). Observability Collapse: A Mechanism for the Failure
of Early Warning Signals. [Preprint]
```

## License

MIT
