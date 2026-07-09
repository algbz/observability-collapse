# Observability Collapse in Stochastic Nonlinear Systems

**Author:** Aldo A. Aguilar Bermúdez
**Paper:** *State-dependent observation and detectability collapse in stochastic nonlinear systems*
**Journal:** International Journal of Dynamics and Control, 14, Article 254 (2026)
**DOI:** https://doi.org/10.1007/s40435-026-02220-z

## Overview

This repository contains the simulation code for the peer-reviewed paper:

> **State-dependent observation and detectability collapse in stochastic nonlinear systems**
> *International Journal of Dynamics and Control* 14, Article 254 (2026).
> https://doi.org/10.1007/s40435-026-02220-z

The paper studies a mechanism called **observability collapse**: transition indicators based on critical slowing down can fail when the observation function loses local sensitivity near attractor states. In this setting, the latent variance can increase as a bifurcation is approached, while the observed variance decreases because the measurement map suppresses the signal-bearing component of the fluctuations.

The repository reproduces the main numerical figures: the mechanism demonstration, the observation-function ablation, the collapse-boundary validation, and the parameter sweep.

## System

The latent dynamics are given by a supercritical pitchfork normal form under additive noise and slow parameter drift:

```text
dx = (μx − x³)dt + σ dW
```

where the parameter μ drifts from 2.0 to 0.05, approaching the bifurcation at μ = 0.

The analytical observation family is:

```text
g_α(x) = sign(x)|x|^α
```

For this family:

* α > 2: observability collapse occurs; observed variance falls despite rising latent variance.
* α = 2: balanced regime; the signal-bearing term remains asymptotically constant.
* α < 2: critical-slowing-down variance growth is transmitted to the observable.

## Note on x, |x|, and signed observables

The paper uses two related types of observables:

* The analytical family is the signed-power family
  `g_α(x) = sign(x)|x|^α`.
  For α = 1, this gives `g_1(x) = x`.

* Figure 1 uses magnitude observables, such as `|x|³` and `|x|`, for amplitude-based visualization.

* Figure 2 uses signed observables, `x³`, `tanh(2x)`, and `x`, to isolate the effect of vanishing local sensitivity.

The local collapse criterion is unchanged because it depends on squared local sensitivity.

## Reproducing the results

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Generate all figures

Run:

```bash
python simulation_code.py
```

This produces:

* `figure1_observability_collapse.png/pdf` — Mechanism demonstration.
* `figure2_ablation.png/pdf` — Ablation, criterion validation, and parameter sweep.

## Simulation parameters

| Parameter              | Value                 |
| ---------------------- | --------------------- |
| μ₀                     | 2.0                   |
| μ_end                  | 0.05                  |
| σ                      | 0.3                   |
| Δt                     | 0.01                  |
| T                      | 30,000 steps          |
| α collapse case        | 3                     |
| α control case         | 1                     |
| Rolling window         | 1,500 steps           |
| Ensemble for Fig. 1(c) | 30 runs               |
| Sweep grid             | 16 × 16 over α and σ  |
| Sweep ensemble         | 8 runs per grid point |

## Key equations

The local variance decomposition used in the paper is:

```text
Var[S] ≈ (g'(x*))² Var[x] + σ_ε²
```

For the pitchfork normal form, local linearization gives:

```text
Var[x] ≈ σ² / (4μ)
```

For the signed-power observation family,

```text
(g'_α(x*))² = α² μ^(α−1)
```

so the signal-bearing observed-variance term scales as:

```text
(g'_α(x*))² Var[x] = (α² σ² / 4) μ^(α−2)
```

Therefore, the collapse criterion is:

```text
α > 2
```

In this regime, the signal-bearing contribution to observed variance tends to zero as μ → 0, even though the latent variance increases.

## Figure descriptions

### Figure 1: Mechanism demonstration

* **A.** Pitchfork potential
  `V(x) = −μx²/2 + x⁴/4`
  with observation function `g(x) = |x|³`.

* **B.** Smoothed observable signal over time for the collapsing observation and bounded-sensitivity control.

* **C.** Eq. (5) decomposition: ensemble-averaged latent variance rises, local sensitivity falls, and observed variance is suppressed.

* **D.** Rolling variance ablation: `|x|³` variance decreases while `|x|` variance increases under identical latent dynamics.

### Figure 2: Validation and ablation

* **A.** Three observation functions: `x³` collapsing observation, `tanh(2x)` nonlinear non-collapsing control, and `x` linear control.

* **B.** Criterion validation: theoretical boundary at α = 2 compared with the numerically observed collapse boundary.

* **C.** Rolling lag-1 autocorrelation for all three observations.

* **D.** Full collapse regime: parameter sweep over observation exponent α and noise level σ.

## Citation

Please cite the published article as:

```text
Bermúdez, A.A.A. State-dependent observation and detectability collapse in stochastic nonlinear systems.
International Journal of Dynamics and Control 14, 254 (2026).
https://doi.org/10.1007/s40435-026-02220-z
```

BibTeX:

```bibtex
@article{Bermudez2026ObservabilityCollapse,
  author  = {Berm{\'u}dez, Aldo Alberto Aguilar},
  title   = {State-dependent observation and detectability collapse in stochastic nonlinear systems},
  journal = {International Journal of Dynamics and Control},
  volume  = {14},
  pages   = {254},
  year    = {2026},
  doi     = {10.1007/s40435-026-02220-z}
}
```

## License

MIT
