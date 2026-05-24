"""
Observability Collapse in Stochastic Nonlinear Systems
=======================================================
Self-contained code producing Figure 1 and Figure 2.

System: supercritical pitchfork  dx = (μx - x³)dt + σ dW
Observation family: g_α(x) = sign(x)|x|^α
Collapse criterion: α > 2

Usage:
    python simulation_code.py

Outputs:
    figure1_observability_collapse.png/pdf
    figure2_ablation.png/pdf

Author: Aldo A. Aguilar Bermúdez
Repository: https://github.com/algbz/observability-collapse
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

SIGMA = 0.3
DT = 0.01
N_STEPS = 30000
MU_START = 2.0
MU_END = 0.05
ALPHA_OBS = 3
W_ROLL = 1500
N_ENS = 30
SEED_SINGLE = 42

SWEEP_ALPHAS = np.linspace(1.0, 4.0, 16)
SWEEP_NOISES = np.linspace(0.15, 0.7, 16)
SWEEP_ENS = 8
SWEEP_STEPS = 15000
SWEEP_W = 1000

C_DARK = '#1a1a2e'
C_GREEN = '#3a7d44'
C_PURP = '#6a4c93'
C_CTRL = '#777777'
C_TEAL = '#2a7a6f'
C_GREY = '#999999'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333',
})


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def simulate(n_steps, mu_start, mu_end, sigma, dt, seed, x0=None):
    rng = np.random.RandomState(seed)
    mu_vals = np.linspace(mu_start, mu_end, n_steps)
    x = np.zeros(n_steps)
    x[0] = (np.sqrt(mu_start) + rng.normal(0, 0.05)) if x0 is None else x0
    for i in range(1, n_steps):
        mu = mu_vals[i]
        x[i] = x[i-1] + (mu*x[i-1] - x[i-1]**3)*dt + sigma*np.sqrt(dt)*rng.normal()
    return x, mu_vals


def observe(x, alpha):
    return np.sign(x) * np.abs(x)**alpha


def rolling_var(s, w):
    return np.array([np.var(s[j-w:j]) for j in range(w, len(s))])


def rolling_ac1(s, w):
    ac = []
    for j in range(w, len(s)):
        c = s[j-w:j]; c = c - c.mean(); v = np.var(c)
        ac.append(np.corrcoef(c[:-1], c[1:])[0, 1] if v > 1e-15 else 0.0)
    return np.array(ac)


def smooth(arr, kern):
    return np.convolve(arr, np.ones(kern)/kern, mode='same')


# ══════════════════════════════════════════════════════════════
# FIGURE 1: MECHANISM
# ══════════════════════════════════════════════════════════════

def make_figure1():
    print("Figure 1: mechanism...")
    time = np.arange(N_STEPS) * DT
    x, mu = simulate(N_STEPS, MU_START, MU_END, SIGMA, DT, SEED_SINGLE)
    S_col = observe(x, ALPHA_OBS)
    S_ctl = np.abs(x)

    # B: smoothed signals
    kb = 400
    sm_col = np.convolve(np.abs(S_col), np.ones(kb)/kb, mode='valid')
    sm_ctl = np.convolve(S_ctl, np.ones(kb)/kb, mode='valid')
    t_sm = time[kb//2:kb//2+len(sm_col)]

    # D: rolling variance
    rv_col = rolling_var(S_col, W_ROLL)
    rv_ctl = rolling_var(S_ctl, W_ROLL)
    rt = time[W_ROLL:]
    skip = 300
    ref_c = np.mean(rv_col[skip:skip+400])
    ref_l = np.mean(rv_ctl[skip:skip+400])
    rv_cn = smooth(rv_col/max(ref_c, 1e-12), 600)[skip:-300]
    rv_ln = smooth(rv_ctl/max(ref_l, 1e-12), 600)[skip:-300]
    t_d = rt[skip:-300]

    # C: ensemble
    print(f"  ensemble ({N_ENS} runs)...")
    all_rv_x, all_rv_col, all_gp = [], [], []
    for run in range(N_ENS):
        xr, _ = simulate(N_STEPS, MU_START, MU_END, SIGMA, DT, run*17+3)
        Sr = observe(xr, ALPHA_OBS)
        gpr = (ALPHA_OBS * np.abs(xr)**(ALPHA_OBS-1))**2
        all_rv_x.append(rolling_var(xr, W_ROLL))
        all_rv_col.append(rolling_var(Sr, W_ROLL))
        all_gp.append(np.array([np.mean(gpr[j-W_ROLL:j]) for j in range(W_ROLL, N_STEPS)]))

    rt_e = time[W_ROLL:]
    trim = 1500
    mx = smooth(np.mean(all_rv_x, 0), 1500)[trim:-trim]
    mc = smooth(np.mean(all_rv_col, 0), 1500)[trim:-trim]
    mg = smooth(np.mean(all_gp, 0), 1500)[trim:-trim]
    t_pe = rt_e[trim:-trim]
    mx_n = mx/mx.max(); mc_n = mc/mc.max(); mg_n = mg/mg.max()

    # Stats
    sl_c, _, _, p_c, _ = sp_stats.linregress(np.arange(len(rv_col)), rv_col)
    sl_l, _, _, p_l, _ = sp_stats.linregress(np.arange(len(rv_ctl)), rv_ctl)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # A
    ax = axes[0, 0]
    xr = np.linspace(-2, 2, 300)
    V = -xr**2/2 + xr**4/4; V = (V-V.min())/(V.max()-V.min())
    g = np.abs(xr)**ALPHA_OBS; g = g/g.max()
    ax.plot(xr, V, color=C_DARK, lw=2.2, label='$V(x)$')
    ax.plot(xr, g, color=C_PURP, lw=2, label=f'$g(x) = |x|^{ALPHA_OBS}$')
    ax.fill_between(xr, 0, g, alpha=0.06, color=C_PURP)
    ax.plot([-1, 1], [0, 0], 'o', color=C_GREEN, ms=7, zorder=5)
    ax.text(-1, -0.13, '$x^*$', ha='center', fontsize=9, color=C_GREEN)
    ax.text(1, -0.13, '$x^*$', ha='center', fontsize=9, color=C_GREEN)
    ax.set_xlabel('$x$'); ax.set_ylabel('Normalised')
    ax.set_title('A.  Potential and observation', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9); ax.set_ylim(-0.18, 1.1)

    # B
    ax = axes[0, 1]
    ax.plot(t_sm, sm_col, color=C_PURP, lw=1.2, label=f'$|x|^{ALPHA_OBS}$ (collapse)')
    ax.plot(t_sm, sm_ctl, color=C_CTRL, lw=1.2, ls='--', label='$|x|$ (control)')
    ax.set_xlabel('Time'); ax.set_ylabel('Smoothed signal')
    ax.set_title('B.  Observation over time', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9); ax.set_ylim(bottom=-0.01)

    # C
    ax = axes[1, 0]
    ax.plot(t_pe, mx_n, color=C_GREEN, lw=2.2, label='Var[$x$] (latent)')
    ax.plot(t_pe, mg_n, color=C_CTRL, lw=2, ls='--', label="$\\langle(g')^2\\rangle$ (sensitivity)")
    ax.plot(t_pe, mc_n, color=C_PURP, lw=2.2, label=f'Var[$x^{ALPHA_OBS}$] (observed)')
    ax.fill_between(t_pe, mc_n, mx_n, where=mx_n > mc_n*1.1, alpha=0.07, color=C_PURP)
    for i in range(len(t_pe)//3, 2*len(t_pe)//3):
        if mx_n[i] > mc_n[i]*1.5:
            ax.annotate('Observability\ncollapse', xy=(t_pe[i], (mx_n[i]+mc_n[i])/2),
                        fontsize=9, color=C_GREY, fontstyle='italic', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GREY, alpha=0.8))
            break
    ax.set_xlabel('Time'); ax.set_ylabel('Normalised to peak')
    ax.set_title(f'C.  Eq. (5) decomposition ($n={N_ENS}$ ensemble)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9); ax.set_ylim(0, 1.05)

    # D
    ax = axes[1, 1]
    ax.plot(t_d, rv_cn, color=C_PURP, lw=1.5, label=f'$|x|^{ALPHA_OBS}$ (variance $\\downarrow$)')
    ax.plot(t_d, rv_ln, color=C_CTRL, lw=1.5, ls='--', label='$|x|$ (control, variance $\\uparrow$)')
    ax.set_xlabel('Time'); ax.set_ylabel('Variance (normalised)')
    ax.set_title('D.  Rolling variance: ablation', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=1.8)
    plt.savefig('figure1_observability_collapse.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure1_observability_collapse.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  |x|^{ALPHA_OBS} slope = {sl_c:.2e}, |x| slope = {sl_l:.2e}")
    print("  saved.\n")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: VALIDATION
# ══════════════════════════════════════════════════════════════

def make_figure2():
    print("Figure 2: validation...")
    time = np.arange(N_STEPS) * DT

    # Single trajectory, three observations
    x, _ = simulate(N_STEPS, MU_START, MU_END, SIGMA, DT, SEED_SINGLE)
    S_col = observe(x, ALPHA_OBS)
    S_tanh = np.tanh(2*x)
    S_lin = x.copy()

    ac_col = rolling_ac1(S_col, W_ROLL)
    ac_tanh = rolling_ac1(S_tanh, W_ROLL)
    ac_lin = rolling_ac1(S_lin, W_ROLL)
    rt = time[W_ROLL:]

    # Ablation stats
    rv_x = rolling_var(x, W_ROLL)
    rv_c = rolling_var(S_col, W_ROLL)
    rv_t = rolling_var(S_tanh, W_ROLL)
    rv_l = rolling_var(S_lin, W_ROLL)
    sl_lat, _, _, _, _ = sp_stats.linregress(np.arange(len(rv_x)), rv_x)
    for name, rv in [('|x|^3', rv_c), ('tanh(2x)', rv_t), ('x', rv_l)]:
        sl, _, _, _, _ = sp_stats.linregress(np.arange(len(rv)), rv)
        print(f"  {name}: slope = {sl:+.2e} {'COLLAPSE' if sl < 0 else ''}")
    print(f"  latent: slope = {sl_lat:+.2e}")

    # Parameter sweep
    print(f"  parameter sweep ({len(SWEEP_ALPHAS)}x{len(SWEEP_NOISES)})...")
    mu_sw = np.linspace(MU_START, 0.1, SWEEP_STEPS)
    collapse_map = np.zeros((len(SWEEP_NOISES), len(SWEEP_ALPHAS)))

    for ni, sig in enumerate(SWEEP_NOISES):
        for ai, alpha in enumerate(SWEEP_ALPHAS):
            rng = np.random.RandomState(ni*200 + ai*13)
            slopes = []
            for r in range(SWEEP_ENS):
                xs = np.zeros(SWEEP_STEPS)
                xs[0] = np.sqrt(MU_START) + rng.normal(0, 0.05)
                for i in range(1, SWEEP_STEPS):
                    xs[i] = xs[i-1] + (mu_sw[i]*xs[i-1] - xs[i-1]**3)*DT + sig*np.sqrt(DT)*rng.normal()
                S = observe(xs, alpha)
                rv = np.array([np.var(S[j-SWEEP_W:j]) for j in range(SWEEP_W, SWEEP_STEPS, 50)])
                sl, _, _, _, _ = sp_stats.linregress(np.arange(len(rv)), rv)
                slopes.append(sl)
            collapse_map[ni, ai] = np.mean(slopes)
        if (ni+1) % 4 == 0:
            print(f"    {ni+1}/{len(SWEEP_NOISES)}")

    # Extract observed boundary
    obs_alpha, obs_noise = [], []
    for ni, sig in enumerate(SWEEP_NOISES):
        row = collapse_map[ni, :]
        for ai in range(len(SWEEP_ALPHAS)-1):
            if row[ai] >= 0 and row[ai+1] < 0:
                frac = row[ai] / (row[ai] - row[ai+1])
                obs_alpha.append(SWEEP_ALPHAS[ai] + frac*(SWEEP_ALPHAS[ai+1]-SWEEP_ALPHAS[ai]))
                obs_noise.append(sig)
                break
    obs_alpha = np.array(obs_alpha); obs_noise = np.array(obs_noise)

    # Smoothed autocorrelation
    trim = 500; t_p = rt[trim:-trim]
    ac_col_s = smooth(ac_col, 600)[trim:-trim]
    ac_tanh_s = smooth(ac_tanh, 600)[trim:-trim]
    ac_lin_s = smooth(ac_lin, 600)[trim:-trim]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # A: Observation functions
    ax = axes[0, 0]
    xr = np.linspace(-2, 2, 300)
    ax.plot(xr, np.sign(xr)*np.abs(xr)**3, color=C_PURP, lw=2, label=f'$x^{ALPHA_OBS}$ ($\\alpha={ALPHA_OBS}$, collapse)')
    ax.plot(xr, np.tanh(2*xr), color=C_TEAL, lw=2, label='$\\tanh(2x)$ (nonlinear ctrl)')
    ax.plot(xr, xr, color=C_CTRL, lw=2, ls='--', label='$x$ (linear ctrl)')
    ax.axhline(0, color='#ddd', lw=0.5); ax.axvline(0, color='#ddd', lw=0.5)
    ax.axvspan(-0.3, 0.3, alpha=0.06, color=C_PURP)
    ax.text(0, -2.3, 'attractor\nregion', ha='center', fontsize=8, color=C_GREY, fontstyle='italic')
    ax.set_xlabel('$x$'); ax.set_ylabel('$g(x)$')
    ax.set_title('A.  Observation functions', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper left')
    ax.set_ylim(-2.5, 2.5); ax.set_xlim(-2, 2)

    # B: Criterion validation
    ax = axes[0, 1]
    ax.axvline(2.0, color=C_DARK, lw=2.5, ls='--', label='Theory: $\\alpha_c = 2$', zorder=3)
    if len(obs_alpha) > 0:
        ax.plot(obs_alpha, obs_noise, 'o-', color=C_PURP, lw=2, ms=5,
                markeredgecolor='white', markeredgewidth=0.6, label='Numerical boundary', zorder=4)
    ax.text(1.3, 0.42, 'CSD\npreserved', ha='center', fontsize=10, color=C_GREEN, alpha=0.5)
    ax.text(3.0, 0.42, 'Collapse', ha='center', fontsize=10, color=C_PURP, alpha=0.5)
    ax.set_xlabel('Observation exponent $\\alpha$'); ax.set_ylabel('Noise level $\\sigma$')
    ax.set_title('B.  Criterion validation: theory vs numerics', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='lower right')
    ax.set_xlim(SWEEP_ALPHAS[0], SWEEP_ALPHAS[-1])
    ax.set_ylim(SWEEP_NOISES[0]-0.02, SWEEP_NOISES[-1]+0.02)

    # C: Autocorrelation
    ax = axes[1, 0]
    ax.plot(t_p, ac_col_s, color=C_PURP, lw=2, label=f'$x^{ALPHA_OBS}$ (collapse)')
    ax.plot(t_p, ac_tanh_s, color=C_TEAL, lw=1.8, label='$\\tanh(2x)$ (nonlin ctrl)')
    ax.plot(t_p, ac_lin_s, color=C_CTRL, lw=1.8, ls='--', label='$x$ (linear ctrl)')
    ax.set_xlabel('Time'); ax.set_ylabel('Lag-1 autocorrelation')
    ax.set_title('C.  Rolling autocorrelation', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)

    # D: Full collapse regime
    ax = axes[1, 1]
    vmax = np.percentile(np.abs(collapse_map), 90)
    im = ax.imshow(collapse_map, aspect='auto', origin='lower',
                   extent=[SWEEP_ALPHAS[0], SWEEP_ALPHAS[-1], SWEEP_NOISES[0], SWEEP_NOISES[-1]],
                   cmap='PiYG', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    try:
        X_a, Y_n = np.meshgrid(SWEEP_ALPHAS, SWEEP_NOISES)
        cs = ax.contour(X_a, Y_n, collapse_map, levels=[0], colors=[C_DARK], linewidths=2)
        ax.clabel(cs, fmt='slope=0', fontsize=8)
    except: pass
    ax.axvline(2.0, color=C_GREY, ls=':', lw=0.8, alpha=0.5)
    ax.text(2.08, SWEEP_NOISES[-1]*0.93, '$\\alpha_c=2$', fontsize=8, va='top', color=C_GREY, alpha=0.6)
    ax.plot(ALPHA_OBS, SIGMA, '*', color=C_PURP, ms=12, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5, label=f'$\\alpha={ALPHA_OBS}$ (tested)')
    ax.plot(1.0, SIGMA, 's', color=C_CTRL, ms=8, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5, label='$\\alpha=1$ (control)')
    ax.set_xlabel('Observation exponent $\\alpha$'); ax.set_ylabel('Noise level $\\sigma$')
    ax.set_title('D.  Full collapse regime (parameter sweep)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7, framealpha=0.9, loc='upper left')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Var[$S$] trend', fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.savefig('figure2_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure2_ablation.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  saved.\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Observability Collapse — Simulation Code")
    print("=" * 60)
    print(f"System: dx = (μx − x³)dt + σdW")
    print(f"μ: {MU_START} → {MU_END}, σ = {SIGMA}, dt = {DT}")
    print(f"T = {N_STEPS} steps ({N_STEPS*DT:.0f} time units)")
    print(f"Observation: g(x) = sign(x)|x|^α, α = {ALPHA_OBS}")
    print(f"Collapse boundary: α = 2 (analytical)\n")

    make_figure1()
    make_figure2()

    print("All figures generated.")
