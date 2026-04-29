#!/usr/bin/env python3
"""
GLE simulation of butane dihedral barrier crossing.

Tests whether the gap between κ_RF (~0.25, MD) and κ_GH (~0.89, kernel +
parabolic-barrier) is due to anharmonicity of the dihedral PMF.

Two GLE runs with the same Prony 1-exp kernel K(t)=a*exp(-t/τ):
  (1) full anharmonic F(φ) from binned ⟨φ̈|φ⟩
  (2) parabolic F(φ) = ω_b² (φ-φ_b)  — Grote-Hynes assumption

Plateau of κ_GLE(t) is compared to κ_RF (MD) and κ_GH (analytic).

Reuses the OU-noise + integrated-memory scheme from
kernel_extraction/benchmark_kernel_vs_reactiveflux.py.
"""

import numpy as np
import os
import sys
import glob
import scipy.optimize
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../kernel_extraction'))
from benchmark_prony_nlsq import extract_kernel_prony
from benchmark_extended_cases import generate_gle_multiexp

SHOOT_DIR = os.path.join(os.path.dirname(__file__),
                         'gromacs_run/shooting_barrier')


# ── Load shooting data ────────────────────────────────────────────────────

def load_shooting_data(phi_tol_deg=10.0):
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

    phi_barrier = (-phi_barrier_plumed) % (2 * np.pi)
    phi_tol = np.radians(phi_tol_deg)

    phi_list, phidot_list, ddot_list = [], [], []
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

    print(f"Loaded {len(phi_list)} trajectories, "
          f"barrier @ {np.degrees(phi_barrier):.1f}°, dt={dt} ps")
    return phi_list, phidot_list, ddot_list, dt, phi_barrier


def compute_mean_force(phi_list, ddot_list, nbins=36):
    all_phi = np.concatenate(phi_list)
    all_ddot = np.concatenate(ddot_list)
    edges = np.linspace(0, 2*np.pi, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    force = np.zeros(nbins)
    for i in range(nbins):
        mask = (all_phi >= edges[i]) & (all_phi < edges[i+1])
        if mask.sum() > 10:
            force[i] = np.mean(all_ddot[mask])
    # Periodic spline
    phi_ext = np.concatenate([centers - 2*np.pi, centers, centers + 2*np.pi])
    f_ext = np.tile(force, 3)
    cs = CubicSpline(phi_ext, f_ext)
    return cs


# GLE integrator: reuses generate_gle_multiexp (one aux per exp term).


def compute_reactive_flux(x, v, phi_barrier):
    v0 = v[:, 0]
    abs_v0_mean = np.mean(np.abs(v0))
    T = x.shape[1]
    kappa = np.zeros(T)
    for t in range(T):
        kappa[t] = 2 * np.mean(v0 * (x[:, t] > phi_barrier).astype(float)) / abs_v0_mean
    return kappa


# ── Grote-Hynes (1-exp kernel) ────────────────────────────────────────────

def grote_hynes_kappa(omega_b, amps, taus):
    """κ_GH for K(t) = Σ aᵢ exp(-t/τᵢ); K̂(s) = Σ aᵢτᵢ/(1+τᵢs)."""
    amps = np.atleast_1d(amps)
    taus = np.atleast_1d(taus)
    def eq(lam):
        Kh = float(np.sum(amps * taus / (1 + taus * lam)))
        return lam - omega_b**2 / (lam + Kh)
    lam_r = scipy.optimize.brentq(eq, 1e-10, omega_b - 1e-10)
    return lam_r / omega_b


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-traj-fit', type=int, default=None,
                        help='Subsample N MD trajectories for Prony fit '
                             '(default: use all)')
    args = parser.parse_args()

    phi_list, phidot_list, ddot_list, dt_md, phi_b = load_shooting_data()

    if args.n_traj_fit is not None and args.n_traj_fit < len(phi_list):
        rng_sub = np.random.default_rng(0)
        idx = rng_sub.choice(len(phi_list), args.n_traj_fit, replace=False)
        phi_list = [phi_list[i] for i in idx]
        phidot_list = [phidot_list[i] for i in idx]
        ddot_list = [ddot_list[i] for i in idx]
        print(f"Subsampled to {len(phi_list)} trajectories for fitting")

    cs_force = compute_mean_force(phi_list, ddot_list)
    force_full = lambda x: cs_force(x)
    V_pp = -cs_force(phi_b, 1)
    omega_b = np.sqrt(abs(V_pp))
    print(f"ω_b = {omega_b:.2f} rad/ps")

    # Effective kBT from initial-velocity variance of barrier shoots
    v0_arr = np.array([pd[0] for pd in phidot_list])
    kBT_eff = np.var(v0_arr)
    print(f"kBT_eff = ⟨v0²⟩ = {kBT_eff:.2f} rad²/ps²")

    # Prony 1-exp kernel from MD shooting data (same protocol as butane_rf_vs_kernel.py)
    stride = 5
    dt_e = dt_md * stride
    nk = max(10, int(1.0 / dt_e))
    tm = max(5, int(0.5 / dt_e))
    x_sub, v_sub, a_sub = [], [], []
    for phi, jv in zip(phi_list, phidot_list):
        phi_s = phi[::stride]
        jv_s = jv[::stride]
        T = len(jv_s)
        a_s = np.zeros(T)
        a_s[1:-1] = (jv_s[2:] - jv_s[:-2]) / (2 * dt_e)
        a_s[0] = a_s[1]; a_s[-1] = a_s[-2]
        x_sub.append(phi_s); v_sub.append(jv_s); a_sub.append(a_s)

    res = extract_kernel_prony(
        x_sub, v_sub, a_sub, dt_e,
        t0_max_idx=0, tau_max_idx=tm,
        force_func=force_full, n_exp=2, n_kernel=nk)
    amps = np.array(res['amps'])
    taus = np.array(res['taus'])
    for i, (a, t) in enumerate(zip(amps, taus)):
        print(f"  Prony exp {i}: amp={a:.1f} rad²/ps², τ={t*1000:.0f} fs, "
              f"γᵢ={a*t:.2f} rad/ps")
    print(f"  γ_int_total = {np.sum(amps*taus):.2f} rad/ps")

    kap_gh = grote_hynes_kappa(omega_b, amps, taus)
    print(f"κ_GH (2-exp) = {kap_gh:.3f}")

    # κ_RF: compute on the same subsample as the kernel fit (apples-to-apples)
    phi_md = np.array(phi_list)
    pdot_md = np.array(phidot_list)
    v0_md = pdot_md[:, 0]
    abs_v0_md = np.mean(np.abs(v0_md))
    Tmd = phi_md.shape[1]
    kap_rf_t = np.zeros(Tmd)
    for k in range(Tmd):
        kap_rf_t[k] = 2 * np.mean(v0_md * (phi_md[:, k] > phi_b).astype(float)) / abs_v0_md
    t_rf = np.arange(Tmd) * dt_md
    plat_mask = (t_rf >= 3.0) & (t_rf <= 8.0)
    if plat_mask.sum() < 5:
        plat_mask = (t_rf >= 0.5 * t_rf[-1])
    kap_rf = float(np.mean(kap_rf_t[plat_mask]))
    print(f"κ_RF (same subsample, plateau) = {kap_rf:.3f}")

    # ── Run GLE simulations ───────────────────────────────────────────────
    N_traj = 5000
    T_total = 5.0  # ps
    dt_gle = min(0.0005, 0.1 * float(taus.min()))
    nsteps = int(T_total / dt_gle) + 1
    print(f"\nGLE: N={N_traj}, T={T_total} ps, dt={dt_gle*1000:.2f} fs, "
          f"nsteps={nsteps}")

    force_harm = lambda x: omega_b**2 * (x - phi_b)

    print("Running GLE with full anharmonic F(φ), 2-exp kernel...", flush=True)
    rng = np.random.default_rng(42)
    x_full, v_full = generate_gle_multiexp(
        amps.tolist(), taus.tolist(), kBT_eff, dt_gle, nsteps, N_traj,
        x0=phi_b, rng=rng, force=force_full)

    print("Running GLE with parabolic F(φ), 2-exp kernel...", flush=True)
    rng = np.random.default_rng(43)
    x_harm, v_harm = generate_gle_multiexp(
        amps.tolist(), taus.tolist(), kBT_eff, dt_gle, nsteps, N_traj,
        x0=phi_b, rng=rng, force=force_harm)

    # κ(t)
    # For harmonic case, "barrier" is at x0 = phi_b (the unstable max).
    kap_full = compute_reactive_flux(x_full, v_full, phi_barrier=phi_b)
    kap_harm = compute_reactive_flux(x_harm, v_harm, phi_barrier=phi_b)
    t_gle = np.arange(nsteps) * dt_gle

    # Plateau (last 30%)
    plat_full = np.mean(kap_full[int(0.7*nsteps):])
    plat_harm = np.mean(kap_harm[int(0.7*nsteps):])
    print(f"\nκ_GLE_full (anharm)   = {plat_full:.3f}  vs κ_RF = {kap_rf:.3f}")
    print(f"κ_GLE_harm (parabolic) = {plat_harm:.3f}  vs κ_GH = {kap_gh:.3f}")

    np.savez('butane_gle_kappa.npz',
             t=t_gle, kappa_full=kap_full, kappa_harm=kap_harm,
             omega_b=omega_b, amps=amps, taus=taus, kBT=kBT_eff,
             phi_barrier=phi_b,
             plateau_full=plat_full, plateau_harm=plat_harm,
             kap_rf=kap_rf, kap_gh=kap_gh)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(t_gle*1000, kap_full, 'C2-', lw=1.5,
            label=f'GLE, full F(φ): plateau={plat_full:.3f}')
    ax.plot(t_gle*1000, kap_harm, 'C3-', lw=1.5,
            label=f'GLE, parabolic: plateau={plat_harm:.3f}')
    ax.axhline(kap_rf, color='C0', ls=':', lw=2,
               label=f'κ_RF (MD) = {kap_rf:.3f}')
    ax.axhline(kap_gh, color='k', ls='--', lw=1.5,
               label=f'κ_GH (1-exp) = {kap_gh:.3f}')
    ax.set_xlabel('t (fs)')
    ax.set_ylabel(r'$\kappa(t)$')
    ax.set_title(f'Butane GLE κ(t): anharm vs harm  (N={N_traj})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_total*1000)
    ax.set_ylim(-0.1, 1.1)

    # PMF profile + parabolic comparison around barrier
    ax = axes[1]
    phi_grid = np.linspace(0, 2*np.pi, 400)
    F_full = cs_force(phi_grid)
    fe = -np.cumsum(F_full) * (phi_grid[1] - phi_grid[0])
    fe -= fe.min()
    F_harm_arr = omega_b**2 * (phi_grid - phi_b)
    fe_harm = -0.5 * omega_b**2 * (phi_grid - phi_b)**2
    fe_harm += fe[np.argmin(abs(phi_grid - phi_b))]

    ax.plot(np.degrees(phi_grid), fe, 'C2-', lw=2, label='Full PMF (from MD)')
    ax.plot(np.degrees(phi_grid), fe_harm, 'C3--', lw=2,
            label=f'Parabolic at φ_b (ω_b={omega_b:.1f})')
    ax.axvline(np.degrees(phi_b), color='gray', ls=':', lw=1,
               label=f'φ_b={np.degrees(phi_b):.1f}°')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel('PMF (rad²/ps²)')
    ax.set_title('Anharmonic vs parabolic potential')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(fe.min()-50, fe.max()+50)

    plt.savefig('butane_gle_kappa.png', dpi=150, bbox_inches='tight')
    plt.savefig('butane_gle_kappa.pdf', bbox_inches='tight')
    print("\nSaved butane_gle_kappa.png/pdf and butane_gle_kappa.npz")
    plt.close()
