#!/usr/bin/env python3
"""
Butane dihedral: reactive flux vs kernel extraction comparison.

Uses shooting trajectories from the eclipsed barrier (~120° PLUMED / ~245° phi_numpy).
From the same trajectories, computes:
  1. Reactive flux κ(t)
  2. Kernel extraction (Free LSQ + Prony) → Grote-Hynes κ
  3. VACF at the barrier → visibility window for kernel extraction

Key finding: the cold-start kernel extraction captures friction at timescales
shorter than the VACF decorrelation time (~150 fs for butane). The GH equation
for the full κ requires friction at ~170 fs, just beyond this window.

Requires: gromacs_run/shooting_barrier/shoot_seed*/phi_data.npz
"""

import numpy as np
import scipy.optimize
from scipy.interpolate import CubicSpline
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../kernel_extraction'))
from nonstationary_kernel_lsq import extract_kernel_lsq
from benchmark_prony_nlsq import extract_kernel_prony

SHOOT_DIR = os.path.join(os.path.dirname(__file__),
                         'gromacs_run/shooting_barrier')


# ── Load data ───────────────────────────────────────────────────────────

def load_shooting_data(phi_tol_deg=10.0):
    """Load shooting trajectory data, filtering by starting position."""
    pattern = os.path.join(SHOOT_DIR, 'shoot_seed*/phi_data.npz')
    files = sorted(glob.glob(pattern))

    dt = None
    phi_barrier_plumed = None
    for f in files:
        d = np.load(f)
        if dt is None:
            dt = float(d['dt'])
        if phi_barrier_plumed is None and 'phi_barrier' in d:
            phi_barrier_plumed = float(d['phi_barrier'])

    if phi_barrier_plumed is None:
        phi_barrier_plumed = 2.0071

    # Convert PLUMED [-π,π] → phi_numpy [0,2π]
    phi_barrier = (-phi_barrier_plumed) % (2 * np.pi)

    phi_tol = np.radians(phi_tol_deg)
    phi_list, phidot_list, ddot_list = [], [], []
    n_rejected = 0
    for f in files:
        d = np.load(f)
        phi0 = d['phi'][0]
        dist = abs(phi0 - phi_barrier)
        if dist > np.pi:
            dist = 2 * np.pi - dist
        if dist < phi_tol:
            phi_list.append(d['phi'])
            phidot_list.append(d['phidot'])
            ddot_list.append(d['ddot'])
        else:
            n_rejected += 1

    print(f"Loaded {len(phi_list)} trajectories "
          f"({n_rejected} rejected, phi_tol={phi_tol_deg}°)")
    print(f"Barrier at phi_numpy = {np.degrees(phi_barrier):.1f}° "
          f"(PLUMED = {np.degrees(phi_barrier_plumed):.1f}°)")
    print(f"dt = {dt} ps, {len(phi_list[0])} frames per trajectory")
    return phi_list, phidot_list, ddot_list, dt, phi_barrier


# ── Mean force from shooting data ──────────────────────────────────────

def compute_mean_force(phi_list, ddot_list, nbins=36):
    """Compute F(φ) = ⟨φ̈|φ⟩ from shooting trajectory data."""
    all_phi = np.concatenate(phi_list)
    all_ddot = np.concatenate(ddot_list)

    phi_edges = np.linspace(0, 2 * np.pi, nbins + 1)
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    force = np.zeros(nbins)
    for i in range(nbins):
        mask = (all_phi >= phi_edges[i]) & (all_phi < phi_edges[i + 1])
        if np.sum(mask) > 10:
            force[i] = np.mean(all_ddot[mask])

    # Periodic cubic spline
    phi_ext = np.concatenate([phi_centers - 2*np.pi, phi_centers,
                              phi_centers + 2*np.pi])
    force_ext = np.tile(force, 3)
    cs = CubicSpline(phi_ext, force_ext)
    return cs, phi_centers, force


# ── Reactive flux ───────────────────────────────────────────────────────

def compute_reactive_flux(phi_list, phidot_list, dt, phi_barrier):
    """Compute κ(t) from barrier-top shooting trajectories.

    κ(t) = 2 ⟨φ̇(0) · θ(φ(t) > φ_barrier)⟩ / ⟨|φ̇(0)|⟩

    The barrier position doesn't affect the plateau (only the transient).
    """
    phi_all = np.array(phi_list)
    v0 = np.array([pd[0] for pd in phidot_list])
    abs_v0_mean = np.mean(np.abs(v0))

    T = phi_all.shape[1]
    kappa = np.zeros(T)
    for t in range(T):
        on_product = (phi_all[:, t] > phi_barrier).astype(float)
        kappa[t] = 2 * np.mean(v0 * on_product) / abs_v0_mean

    return kappa, np.arange(T) * dt


# ── Subsample + FD ─────────────────────────────────────────────────────

def subsample_jv(phi_list, phidot_list, dt, stride):
    """Subsample Jv-based data and compute acceleration from FD of Jv."""
    dt_eff = dt * stride
    x_sub, v_sub, a_sub = [], [], []
    for phi, jv in zip(phi_list, phidot_list):
        phi_s = phi[::stride]
        jv_s = jv[::stride]
        T = len(jv_s)
        a = np.zeros(T)
        a[1:-1] = (jv_s[2:] - jv_s[:-2]) / (2 * dt_eff)
        a[0] = a[1]
        a[-1] = a[-2]
        x_sub.append(phi_s)
        v_sub.append(jv_s)
        a_sub.append(a)
    return x_sub, v_sub, a_sub, dt_eff


# ── GH from kernel ────────────────────────────────────────────────────

def K_hat_numerical(K_arr, dt_k, s):
    t = np.arange(len(K_arr)) * dt_k
    return np.trapz(K_arr * np.exp(-s * t), t)


def kappa_from_kernel(K_arr, dt_k, omega_b):
    def eq(lam):
        Kh = K_hat_numerical(K_arr, dt_k, lam)
        return lam - omega_b**2 / (lam + Kh)
    try:
        lam_r = scipy.optimize.brentq(eq, 1e-10, omega_b * 2, maxiter=200)
        return lam_r / omega_b
    except (ValueError, RuntimeError):
        return np.nan


# ── VACF at barrier ────────────────────────────────────────────────────

def compute_vacf(phidot_list, dt, max_lag_ps=1.0):
    """Compute ⟨v(0)·v(τ)⟩/⟨v(0)²⟩ from shooting trajectories."""
    v_all = np.array([pd[:int(max_lag_ps / dt) + 1] for pd in phidot_list])
    v0 = v_all[:, 0]
    v0_sq = np.mean(v0**2)
    T = v_all.shape[1]
    vacf = np.array([np.mean(v0 * v_all[:, t]) / v0_sq for t in range(T)])
    return vacf, np.arange(T) * dt


# ── Main analysis ──────────────────────────────────────────────────────

if __name__ == '__main__':
    phi_list, phidot_list, ddot_list, dt, phi_barrier = load_shooting_data()
    N_total = len(phi_list)

    # Mean force from shooting data
    cs_force, phi_c, force_arr = compute_mean_force(phi_list, ddot_list)
    force_func = lambda x: cs_force(x)

    # Barrier frequency
    V_pp = -cs_force(phi_barrier, 1)  # V'' = -dF/dφ
    omega_b = np.sqrt(abs(V_pp))
    print(f"V''(φ‡) = {V_pp:.1f}, ω_b = {omega_b:.2f} rad/ps")

    # ── (1) Reactive flux ──
    kap_t, t_rf = compute_reactive_flux(phi_list, phidot_list, dt, phi_barrier)
    # Plateau: 3-8 ps
    mask_plat = (t_rf >= 3.0) & (t_rf <= 8.0)
    kap_rf = np.mean(kap_t[mask_plat])
    print(f"\nReactive flux plateau (3-8 ps): κ_RF = {kap_rf:.3f}")

    # ── (2) Kernel extraction — sweep over t0_max ──
    stride = 5
    x_sub, v_sub, a_sub, dt_e = subsample_jv(phi_list, phidot_list, dt, stride)
    nk = max(10, int(1.0 / dt_e))
    tm = max(5, int(0.5 / dt_e))
    print(f"\nKernel extraction: stride={stride}, dt_eff={dt_e:.4f} ps, "
          f"nk={nk}, tau_max={tm}")

    # Sweep t0_max to see how the kernel changes
    t0_list = [0, 5, 10, 20, 50, 100]
    kernels_by_t0 = {}
    kappa_by_t0 = {}

    for t0_max in t0_list:
        # Free LSQ
        res = extract_kernel_lsq(
            x_sub, v_sub, a_sub, dt_e,
            n_kernel=nk, t0_max_idx=t0_max, tau_max_idx=tm,
            force_func=force_func)
        K = res['K']
        gamma = K_hat_numerical(K, dt_e, 0)
        kap_gh = kappa_from_kernel(K, dt_e, omega_b)
        kernels_by_t0[t0_max] = K

        # Prony
        try:
            res_p = extract_kernel_prony(
                x_sub, v_sub, a_sub, dt_e,
                t0_max_idx=t0_max, tau_max_idx=tm,
                force_func=force_func, n_exp=1, n_kernel=nk)
            kap_prony = kappa_from_kernel(res_p['K'], dt_e, omega_b)
            amp, tau = res_p['amps'][0], res_p['taus'][0]
        except:
            kap_prony = np.nan
            amp, tau = np.nan, np.nan

        kappa_by_t0[t0_max] = {
            'free': kap_gh, 'prony': kap_prony,
            'gamma': gamma, 'amp': amp, 'tau': tau
        }
        print(f"  t0_max={t0_max:3d} ({t0_max*dt_e*1000:.0f} fs): "
              f"γ={gamma:.1f}, κ_Free={kap_gh:.3f}, κ_Prony={kap_prony:.3f}, "
              f"amp={amp:.0f}, τ={tau*1000:.0f} fs")

    # Use t0=0 as reference
    K_free = kernels_by_t0[0]
    t_kernel = np.arange(nk) * dt_e
    res_p0 = extract_kernel_prony(
        x_sub, v_sub, a_sub, dt_e,
        t0_max_idx=0, tau_max_idx=tm,
        force_func=force_func, n_exp=1, n_kernel=nk)
    K_prony = res_p0['K']
    amp_p, tau_p = res_p0['amps'][0], res_p0['taus'][0]
    kap_free_gh = kappa_by_t0[0]['free']
    kap_prony_gh = kappa_by_t0[0]['prony']
    gamma_free = kappa_by_t0[0]['gamma']

    # What K̂ is needed for κ_RF?
    lam_rf = kap_rf * omega_b
    Khat_needed = omega_b**2 / lam_rf - lam_rf
    Khat_extracted = K_hat_numerical(K_free, dt_e, lam_rf)
    print(f"\nFor κ_RF={kap_rf:.3f}: λ_r={lam_rf:.2f}, "
          f"need K̂={Khat_needed:.1f}, extracted K̂={Khat_extracted:.1f} "
          f"({Khat_needed/max(Khat_extracted,1e-10):.0f}× gap)")

    # ── (3) VACF ──
    vacf, t_vacf = compute_vacf(phidot_list, dt, max_lag_ps=0.5)
    # Find zero crossing
    t_zero = None
    for i in range(1, len(vacf)):
        if vacf[i] < 0:
            t_zero = t_vacf[i-1] - vacf[i-1] * (t_vacf[i] - t_vacf[i-1]) / (vacf[i] - vacf[i-1])
            print(f"\nVACF zero crossing at {t_zero*1000:.0f} fs")
            print(f"1/λ_r (for κ_RF) = {1/lam_rf*1000:.0f} fs")
            break

    # ══════════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle(f'Butane dihedral in water — N={N_total} barrier-top trajectories',
                 fontsize=13)

    # (a) Reactive flux κ(t)
    ax = axes[0, 0]
    ax.plot(t_rf * 1000, kap_t, 'C0-', lw=1)
    ax.axhline(kap_rf, color='C0', ls=':', lw=1.5,
               label=f'RF plateau = {kap_rf:.3f}')
    if K_prony is not None:
        ax.axhline(kap_prony_gh, color='C1', ls='--', lw=1.5,
                   label=f'Prony→GH (t₀=0) = {kap_prony_gh:.3f}')
    ax.set_xlabel('t (fs)')
    ax.set_ylabel(r'$\kappa(t)$')
    ax.set_title('(a) Reactive flux vs Grote-Hynes')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-0.1, 1.1)

    # (b) Extracted kernel at t0=0 vs selected t0>0
    ax = axes[0, 1]
    n_show = min(nk, int(0.3 / dt_e))
    colors_t0 = plt.cm.viridis(np.linspace(0, 0.85, len(t0_list)))
    for t0_max, color in zip(t0_list, colors_t0):
        K = kernels_by_t0[t0_max]
        label = f't₀≤{t0_max} ({t0_max*dt_e*1000:.0f} fs)'
        ax.plot(t_kernel[:n_show] * 1000, K[:n_show],
                color=color, lw=1.2, alpha=0.8, label=label)
    ax.set_xlabel('t (fs)')
    ax.set_ylabel(r'$K(t)$ (rad/ps²)')
    ax.set_title('(b) Kernel vs time origin')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 300)

    # (c) VACF at barrier
    ax = axes[0, 2]
    n_vacf = min(len(vacf), int(0.5 / dt) + 1)
    ax.plot(t_vacf[:n_vacf] * 1000, vacf[:n_vacf], 'k-', lw=1.5,
            label='VACF')
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.axvline(1/omega_b * 1000, color='C3', ls=':', lw=1.5,
               label=f'1/ω_b = {1/omega_b*1000:.0f} fs')
    ax.axvline(1/lam_rf * 1000, color='C0', ls=':', lw=1.5,
               label=f'1/λ_r (κ_RF) = {1/lam_rf*1000:.0f} fs')
    if K_prony is not None:
        ax.axvline(tau_p * 1000, color='C1', ls='--', lw=1.5,
                   label=f'τ_kernel = {tau_p*1000:.0f} fs')
    ax.fill_between(t_vacf[:n_vacf] * 1000, 0, vacf[:n_vacf],
                    where=vacf[:n_vacf] > 0, alpha=0.1, color='green',
                    label='Kernel-visible window')
    ax.set_xlabel('t (fs)')
    ax.set_ylabel(r'$C_{vv}(\tau) / C_{vv}(0)$')
    ax.set_title('(c) Barrier-top VACF')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.5, 2.0)

    # (d) κ_GH vs t0_max
    ax = axes[1, 0]
    t0_fs = [t0 * dt_e * 1000 for t0 in t0_list]
    kap_free_arr = [kappa_by_t0[t0]['free'] for t0 in t0_list]
    kap_prony_arr = [kappa_by_t0[t0]['prony'] for t0 in t0_list]
    gamma_arr = [kappa_by_t0[t0]['gamma'] for t0 in t0_list]

    ax.plot(t0_fs, kap_free_arr, 'D-', color='C2', lw=1.5, ms=5,
            label='Free LSQ → GH')
    ax.plot(t0_fs, kap_prony_arr, 's--', color='C1', lw=1.5, ms=5,
            label='Prony → GH')
    ax.axhline(kap_rf, color='C0', ls=':', lw=2,
               label=f'RF plateau = {kap_rf:.3f}')
    if t_zero is not None:
        ax.axvline(t_zero * 1000, color='gray', ls=':', lw=1,
                   label=f'VACF zero = {t_zero*1000:.0f} fs')
    ax.set_xlabel('t₀_max (fs)')
    ax.set_ylabel(r'$\kappa_{GH}$')
    ax.set_title('(d) κ vs time origin window')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.1)

    # (e) Friction coefficient γ vs t0_max
    ax = axes[1, 1]
    ax.plot(t0_fs, gamma_arr, 'o-', color='C4', lw=1.5, ms=5,
            label=r'$\gamma = \hat{K}(0) = \int K\,dt$')
    # What γ is needed for Kramers?
    gamma_needed = omega_b / kap_rf  # Kramers: κ ≈ ω_b/γ (high friction)
    ax.axhline(gamma_needed, color='C0', ls=':', lw=1.5,
               label=f'γ for Kramers κ={kap_rf:.2f}: {gamma_needed:.0f}')
    ax.set_xlabel('t₀_max (fs)')
    ax.set_ylabel(r'$\gamma$ (rad/ps)')
    ax.set_title('(e) Friction coefficient vs t₀')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (f) K̂(s) — Laplace transform
    ax = axes[1, 2]
    s_arr = np.linspace(0.01, 50, 200)
    for t0_max, color in zip([0, 20, 100], ['C2', 'C5', 'C6']):
        K = kernels_by_t0[t0_max]
        Khat = np.array([K_hat_numerical(K, dt_e, s) for s in s_arr])
        ax.plot(s_arr, Khat, color=color, lw=1.5,
                label=f't₀≤{t0_max} ({t0_max*dt_e*1000:.0f} fs)')

    # Mark λ_r and needed value
    ax.axvline(lam_rf, color='C0', ls=':', lw=1.5)
    ax.plot(lam_rf, Khat_needed, 'C0*', ms=12,
            label=f'Need K̂={Khat_needed:.0f} for κ_RF')
    ax.plot(lam_rf, Khat_extracted, 'C2o', ms=8,
            label=f'Have K̂={Khat_extracted:.1f} (t₀=0)')

    # GH self-consistency line
    lam_arr = np.linspace(0.5, omega_b * 1.5, 200)
    Khat_gh = omega_b**2 / lam_arr - lam_arr
    ax.plot(lam_arr, Khat_gh, 'k:', lw=1, alpha=0.5,
            label=r'GH: $\hat{K} = \omega_b^2/\lambda - \lambda$')

    ax.set_xlabel(r'$s$ or $\lambda$ (1/ps)')
    ax.set_ylabel(r'$\hat{K}(s)$ (rad/ps)')
    ax.set_title(r'(f) Laplace domain — effect of t₀')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 50)
    ax.set_ylim(-10, max(Khat_needed * 1.2, 150))

    plt.savefig('butane_rf_vs_kernel.png', dpi=200, bbox_inches='tight')
    plt.savefig('butane_rf_vs_kernel.pdf', bbox_inches='tight')
    print(f"\nSaved butane_rf_vs_kernel.png/pdf")
    plt.close()
