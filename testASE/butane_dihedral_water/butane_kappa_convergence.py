#!/usr/bin/env python3
"""
Convergence figure: κ_RF and κ_GLE_full vs number of MD trajectories.

Sweeps N in [50, 100, 150, 200, 300, 500, all], for each subsample
extracts the Free-LSQ kernel and runs the GLE κ simulation
(both anharmonic and parabolic) on the same data. Plots:
  (a) κ_RF, κ_GLE_full, κ_GLE_harm, κ_GH as functions of N
  (b) Free-LSQ K(t) for several N values
  (c) PMF profile vs parabolic fit at the barrier
"""

import os
import sys
import glob
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../kernel_extraction'))
from nonstationary_kernel_lsq import extract_kernel_lsq
from benchmark_extended_cases import generate_gle_arbitrary

SHOOT_DIR = os.path.join(os.path.dirname(__file__),
                         'gromacs_run/shooting_barrier')


# ── Data loading + mean force ────────────────────────────────────────────

def load_shooting(phi_tol_deg=10.0):
    files = sorted(glob.glob(os.path.join(SHOOT_DIR,
                                          'shoot_seed*/phi_data.npz')))
    dt = None
    phi_b_pl = None
    for f in files:
        d = np.load(f)
        if dt is None:
            dt = float(d['dt'])
        if phi_b_pl is None and 'phi_barrier' in d:
            phi_b_pl = float(d['phi_barrier'])
    if phi_b_pl is None:
        phi_b_pl = 2.0071
    phi_b = (-phi_b_pl) % (2*np.pi)
    phi_tol = np.radians(phi_tol_deg)
    phi_l, pdot_l, ddot_l = [], [], []
    for f in files:
        d = np.load(f)
        phi0 = d['phi'][0]
        dist = abs(phi0 - phi_b)
        if dist > np.pi:
            dist = 2*np.pi - dist
        if dist < phi_tol:
            phi_l.append(d['phi'])
            pdot_l.append(d['phidot'])
            ddot_l.append(d['ddot'])
    return phi_l, pdot_l, ddot_l, dt, phi_b


def mean_force_spline(phi_l, ddot_l, nbins=36):
    all_phi = np.concatenate(phi_l)
    all_dd = np.concatenate(ddot_l)
    edges = np.linspace(0, 2*np.pi, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    F = np.zeros(nbins)
    for i in range(nbins):
        m = (all_phi >= edges[i]) & (all_phi < edges[i+1])
        if m.sum() > 10:
            F[i] = np.mean(all_dd[m])
    phi_ext = np.concatenate([centers - 2*np.pi, centers, centers + 2*np.pi])
    F_ext = np.tile(F, 3)
    return CubicSpline(phi_ext, F_ext)


def kappa_rf(phi_arr, pdot_arr, dt, phi_b):
    v0 = pdot_arr[:, 0]
    abs_v0 = np.mean(np.abs(v0))
    T = phi_arr.shape[1]
    out = np.zeros(T)
    for t in range(T):
        out[t] = 2*np.mean(v0*(phi_arr[:, t] > phi_b).astype(float)) / abs_v0
    t = np.arange(T)*dt
    plat = (t >= 3.0) & (t <= 8.0)
    if plat.sum() < 5:
        plat = (t >= 0.5*t[-1])
    return float(np.mean(out[plat])), out, t


def kappa_gle(x_arr, v_arr, phi_b, dt_gle):
    v0 = v_arr[:, 0]
    abs_v0 = np.mean(np.abs(v0))
    T = x_arr.shape[1]
    out = np.zeros(T)
    for t in range(T):
        out[t] = 2*np.mean(v0*(x_arr[:, t] > phi_b).astype(float)) / abs_v0
    plateau = float(np.mean(out[int(0.7*T):]))
    return plateau, out


# ── Main sweep ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    phi_l, pdot_l, ddot_l, dt_md, phi_b = load_shooting()
    N_full = len(phi_l)
    print(f"Loaded {N_full} trajectories")

    cs_full = mean_force_spline(phi_l, ddot_l, nbins=60)
    F_full = lambda x: cs_full(x)
    # Robust ω_b: polyfit of PMF on a narrow window around φ_b
    phi_grid_w = np.linspace(0, 2*np.pi, 2000)
    fe_w = -np.cumsum(cs_full(phi_grid_w))*(phi_grid_w[1]-phi_grid_w[0])
    fe_w -= fe_w.min()
    narrow = abs(phi_grid_w - phi_b) < 0.15
    poly = np.polyfit(phi_grid_w[narrow] - phi_b, fe_w[narrow], 2)
    omega_b = float(np.sqrt(abs(2*poly[0])))
    print(f"ω_b (narrow polyfit) = {omega_b:.2f} rad/ps")

    v0_arr = np.array([pd[0] for pd in pdot_l])
    kBT_eff = float(np.var(v0_arr))

    N_sweep = [50, 100, 150, 200, 300, 500, N_full]
    rng_master = np.random.default_rng(0)

    rows = []
    K_by_N = {}
    kappa_rf_t_by_N = {}
    kappa_gle_full_t_by_N = {}

    stride = 5
    dt_e = dt_md*stride
    nk = max(10, int(1.0/dt_e))
    tm = max(5, int(0.5/dt_e))

    T_total = 5.0
    dt_gle = 0.001

    for N in N_sweep:
        if N >= N_full:
            idx = np.arange(N_full)
            N = N_full
        else:
            idx = rng_master.choice(N_full, N, replace=False)
        phi_sub = [phi_l[i] for i in idx]
        pdot_sub = [pdot_l[i] for i in idx]
        ddot_sub = [ddot_l[i] for i in idx]

        # κ_RF on this subsample
        phi_md = np.array(phi_sub)
        pdot_md = np.array(pdot_sub)
        kap_rf, kap_rf_t, t_rf = kappa_rf(phi_md, pdot_md, dt_md, phi_b)
        kappa_rf_t_by_N[N] = (t_rf, kap_rf_t)

        # Free-LSQ kernel on this subsample
        x_sub, v_sub, a_sub = [], [], []
        for phi, jv in zip(phi_sub, pdot_sub):
            phi_s = phi[::stride]
            jv_s = jv[::stride]
            T = len(jv_s)
            a_s = np.zeros(T)
            a_s[1:-1] = (jv_s[2:] - jv_s[:-2])/(2*dt_e)
            a_s[0] = a_s[1]; a_s[-1] = a_s[-2]
            x_sub.append(phi_s); v_sub.append(jv_s); a_sub.append(a_s)
        n_kernel_full = max(nk, int(0.5/dt_e))
        res = extract_kernel_lsq(x_sub, v_sub, a_sub, dt_e,
                                 n_kernel=n_kernel_full,
                                 t0_max_idx=0, tau_max_idx=tm,
                                 force_func=F_full)
        K_e = res['K']
        n_trunc = int(round(0.5/dt_e))
        K_e = K_e[:n_trunc]
        K_by_N[N] = (np.arange(len(K_e))*dt_e, K_e.copy())

        # Interpolate kernel onto fine grid for the GLE
        t_orig = np.arange(len(K_e))*dt_e
        n_new = int(round(t_orig[-1]/dt_gle)) + 1
        t_new = np.arange(n_new)*dt_gle
        K_fine = np.interp(t_new, t_orig, K_e)

        nsteps = int(T_total/dt_gle) + 1
        N_traj_gle = 5000

        # Anharmonic
        x_full_gle, v_full_gle = generate_gle_arbitrary(
            K_fine, dt_gle, kBT_eff, nsteps, N_traj_gle,
            x0=phi_b, rng=np.random.default_rng(42), force=F_full)
        kap_full, kap_full_t = kappa_gle(x_full_gle, v_full_gle, phi_b, dt_gle)
        kappa_gle_full_t_by_N[N] = (np.arange(nsteps)*dt_gle, kap_full_t)

        # Parabolic sanity check (less critical to repeat at every N — only at full)
        if N == N_full:
            F_harm = lambda x: omega_b**2 * (x - phi_b)
            x_h, v_h = generate_gle_arbitrary(
                K_fine, dt_gle, kBT_eff, nsteps, N_traj_gle,
                x0=phi_b, rng=np.random.default_rng(43), force=F_harm)
            kap_harm, _ = kappa_gle(x_h, v_h, phi_b, dt_gle)
        else:
            kap_harm = np.nan

        gamma_int = float(np.trapz(K_e, dx=dt_e))
        rows.append((N, kap_rf, kap_full, kap_harm, gamma_int, K_e[0]))
        print(f"N={N:4d}: κ_RF={kap_rf:.3f}  κ_GLE_full={kap_full:.3f}  "
              f"κ_GLE_harm={kap_harm}  γ_int={gamma_int:.2f}", flush=True)

    rows = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
                    dtype=float)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    # (a) κ vs N
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 1], 'o-', color='C0', lw=2, ms=7,
            label=r'$\kappa_{\mathrm{RF}}$ (MD)')
    ax.plot(rows[:, 0], rows[:, 2], 's-', color='C2', lw=2, ms=7,
            label=r'$\kappa_{\mathrm{GLE}}$ (anharmonic, free-LSQ)')
    # κ_GLE_harm only at full N
    full_idx = np.where(rows[:, 0] == N_full)[0][0]
    ax.axhline(rows[full_idx, 3], color='C3', ls='--', lw=1.5,
               label=rf'$\kappa_{{\mathrm{{GLE}}}}^{{\mathrm{{harm}}}} \approx \kappa_{{\mathrm{{GH}}}}$ = {rows[full_idx,3]:.2f}')
    ax.set_xscale('log')
    ax.set_xlabel('N trajectories (subsample)')
    ax.set_ylabel(r'$\kappa$ plateau')
    ax.set_title('(a) Convergence with sample size')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(0, 1.0)

    # (b) Free-LSQ kernel K(t) for several N
    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_sweep)))
    for c, (N, (t_k, K_k)) in zip(cmap, sorted(K_by_N.items())):
        ax.plot(t_k*1000, K_k, color=c, lw=1.6, label=f'N={N}')
    ax.axhline(0, color='gray', lw=0.7, alpha=0.6)
    ax.set_xlabel('t [fs]')
    ax.set_ylabel(r'$K(t)$ [rad²/ps²]')
    ax.set_title('(b) Free-LSQ kernel vs N')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500)

    # (c) PMF + parabolic comparison
    ax = axes[2]
    phi_grid = np.linspace(0, 2*np.pi, 400)
    F_grid = cs_full(phi_grid)
    fe = -np.cumsum(F_grid)*(phi_grid[1]-phi_grid[0])
    fe -= fe.min()
    fe_harm = -0.5*omega_b**2*(phi_grid - phi_b)**2
    fe_harm += fe[np.argmin(abs(phi_grid - phi_b))]
    ax.plot(np.degrees(phi_grid), fe, 'C2-', lw=2, label='Full PMF (MD)')
    ax.plot(np.degrees(phi_grid), fe_harm, 'C3--', lw=2,
            label=rf'Parabolic: $\omega_b$={omega_b:.1f} rad/ps')
    ax.axvline(np.degrees(phi_b), color='gray', ls=':', lw=1)
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel('PMF [rad²/ps²]')
    ax.set_title('(c) Anharmonic vs parabolic')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(fe.min()-50, fe.max()+50)

    fig.suptitle(f'Butane dihedral barrier: κ convergence  '
                 f'(N_full = {N_full} barrier-top shoots)',
                 fontsize=12)

    plt.savefig('butane_kappa_convergence.png', dpi=160, bbox_inches='tight')
    plt.savefig('butane_kappa_convergence.pdf', bbox_inches='tight')

    np.savez('butane_kappa_convergence.npz',
             rows=rows, omega_b=omega_b, kBT=kBT_eff, phi_barrier=phi_b)
    print('\nSaved butane_kappa_convergence.png/pdf/npz')
