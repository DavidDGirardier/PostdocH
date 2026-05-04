#!/usr/bin/env python3
"""Sanity checks: barrier shape (is parabola valid?) + stride sensitivity."""
import os, sys, glob
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../kernel_extraction'))
from nonstationary_kernel_lsq import extract_kernel_lsq

SHOOT = os.path.join(os.path.dirname(__file__),
                     'gromacs_run/shooting_barrier')

# Load
files = sorted(glob.glob(os.path.join(SHOOT, 'shoot_seed*/phi_data.npz')))
phi_l, pdot_l, ddot_l = [], [], []
phi_b_pl = 2.0071
phi_b = (-phi_b_pl) % (2*np.pi)
phi_tol = np.radians(10.0)
phi0_arr = []
for f in files:
    d = np.load(f)
    p0 = d['phi'][0]
    dist = abs(p0 - phi_b)
    if dist > np.pi: dist = 2*np.pi - dist
    if dist < phi_tol:
        phi_l.append(d['phi']); pdot_l.append(d['phidot']); ddot_l.append(d['ddot'])
        phi0_arr.append(p0)
phi0_arr = np.array(phi0_arr)
print(f"Accepted {len(phi_l)} trajectories")
print(f"  phi(0) distribution: mean={np.degrees(phi0_arr.mean()):.2f}°, "
      f"std={np.degrees(phi0_arr.std()):.2f}°, "
      f"range=[{np.degrees(phi0_arr.min()):.2f},{np.degrees(phi0_arr.max()):.2f}]°")
print(f"  Target phi_b = {np.degrees(phi_b):.2f}°  "
      f"(offset of mean: {np.degrees(phi0_arr.mean()-phi_b):.2f}°)")

dt_md = 0.002

# Mean force binning
all_phi = np.concatenate(phi_l)
all_dd = np.concatenate(ddot_l)
nbins = 60   # finer binning
edges = np.linspace(0, 2*np.pi, nbins+1)
centers = 0.5*(edges[:-1]+edges[1:])
F = np.zeros(nbins); cnt = np.zeros(nbins)
for i in range(nbins):
    m = (all_phi >= edges[i]) & (all_phi < edges[i+1])
    cnt[i] = m.sum()
    if m.sum() > 10: F[i] = np.mean(all_dd[m])
phi_ext = np.concatenate([centers - 2*np.pi, centers, centers + 2*np.pi])
F_ext = np.tile(F, 3)
cs = CubicSpline(phi_ext, F_ext)

# Find PMF maximum (where F crosses zero with negative slope)
phi_grid = np.linspace(phi_b - 0.4, phi_b + 0.4, 2000)
F_grid = cs(phi_grid)
# Sign change
sgn = np.sign(F_grid)
crossings = np.where(np.diff(sgn) != 0)[0]
phi_b_actual = None
for c in crossings:
    if cs(phi_grid[c], 1) < 0:  # negative slope = top of PMF
        # linear interpolate to zero
        phi_b_actual = phi_grid[c] - F_grid[c]*(phi_grid[c+1]-phi_grid[c])/(F_grid[c+1]-F_grid[c])
        break
omega_b_target = np.sqrt(abs(-cs(phi_b, 1)))
omega_b_actual = np.sqrt(abs(-cs(phi_b_actual, 1))) if phi_b_actual else None
print(f"\n  PMF maximum (zero-force, dF/dphi<0):")
print(f"    target (PLUMED)   phi_b = {np.degrees(phi_b):.3f}°,  ω_b = {omega_b_target:.3f} rad/ps")
if phi_b_actual:
    print(f"    actual (from data)        = {np.degrees(phi_b_actual):.3f}°,  ω_b = {omega_b_actual:.3f} rad/ps")
    print(f"    offset = {np.degrees(phi_b_actual - phi_b):.3f}°")

# PMF
phi_grid_full = np.linspace(0, 2*np.pi, 600)
fe = -np.cumsum(cs(phi_grid_full))*(phi_grid_full[1]-phi_grid_full[0])
fe -= fe.min()

# Fit parabola in narrow window around the actual barrier
phi_use = phi_b_actual if phi_b_actual else phi_b
narrow = abs(phi_grid_full - phi_use) < 0.15
poly = np.polyfit(phi_grid_full[narrow] - phi_use, fe[narrow], 2)
omega_b_poly = np.sqrt(abs(2*poly[0]))
print(f"    parabolic fit on |Δφ|<0.15 rad: ω_b = {omega_b_poly:.3f} rad/ps  "
      f"(vs spline {omega_b_actual or omega_b_target:.3f})")

# Plot 1: PMF + several parabolic approximations
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
ax = axes[0]
ax.plot(np.degrees(phi_grid_full), fe, 'C2-', lw=2, label='Full PMF (60 bins)')
fe_at = fe[np.argmin(abs(phi_grid_full - phi_use))]
phi_arr = phi_grid_full
fe_par_target = -0.5*omega_b_target**2*(phi_arr-phi_b)**2 + fe[np.argmin(abs(phi_arr-phi_b))]
ax.plot(np.degrees(phi_arr), fe_par_target, 'C3--', lw=1.5,
        label=f'parabola @ target ω_b={omega_b_target:.1f}')
if phi_b_actual:
    fe_par_actual = -0.5*omega_b_actual**2*(phi_arr-phi_b_actual)**2 + fe_at
    ax.plot(np.degrees(phi_arr), fe_par_actual, 'C1--', lw=1.5,
            label=f'parabola @ actual ω_b={omega_b_actual:.1f}')
fe_par_poly = -0.5*omega_b_poly**2*(phi_arr-phi_use)**2 + fe_at
ax.plot(np.degrees(phi_arr), fe_par_poly, 'C0:', lw=1.5,
        label=f'narrow polyfit ω_b={omega_b_poly:.1f}')
ax.axvline(np.degrees(phi_b), color='C3', ls=':', alpha=0.4, label='target φ_b')
if phi_b_actual:
    ax.axvline(np.degrees(phi_b_actual), color='C1', ls=':', alpha=0.6, label='actual φ_b')
# initial-position histogram
ax2 = ax.twinx()
ax2.hist(np.degrees(phi0_arr), bins=20, color='gray', alpha=0.25, label='φ(0) histogram')
ax2.set_ylabel('φ(0) count', color='gray')
ax.set_xlabel('φ [deg]'); ax.set_ylabel('PMF [rad²/ps²]')
ax.set_title('Barrier shape: spline vs parabola')
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(np.degrees(phi_use)-30, np.degrees(phi_use)+30)
ax.set_ylim(fe[abs(phi_grid_full-phi_use)<0.5].min()-50,
            fe[abs(phi_grid_full-phi_use)<0.5].max()+50)
ax.grid(True, alpha=0.3)

# Stride sweep on the kernel
F_full = lambda x: cs(x)
ax = axes[1]
strides = [1, 2, 5, 10]
for s in strides:
    dt_e = dt_md*s
    nk = max(10, int(0.5/dt_e))
    tm = max(5, int(0.5/dt_e))
    x_sub, v_sub, a_sub = [], [], []
    for phi, jv in zip(phi_l, pdot_l):
        phi_s = phi[::s]; jv_s = jv[::s]
        T = len(jv_s)
        a_s = np.zeros(T)
        a_s[1:-1] = (jv_s[2:] - jv_s[:-2])/(2*dt_e)
        a_s[0] = a_s[1]; a_s[-1] = a_s[-2]
        x_sub.append(phi_s); v_sub.append(jv_s); a_sub.append(a_s)
    res = extract_kernel_lsq(x_sub, v_sub, a_sub, dt_e,
                             n_kernel=nk, t0_max_idx=0,
                             tau_max_idx=tm, force_func=F_full)
    K = res['K']
    t_k = np.arange(len(K))*dt_e
    gamma_int = np.trapz(K, dx=dt_e)
    print(f"  stride={s:2d} (dt_e={dt_e*1000:.0f} fs): K(0)={K[0]:.1f}, "
          f"γ_int={gamma_int:.2f}, n_kernel={nk}, reg={res.get('reg', 0):.2e}")
    ax.plot(t_k*1000, K, 'o-', ms=3, lw=1.4,
            label=f"stride={s} (dt_e={dt_e*1000:.0f} fs)")
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('t [fs]'); ax.set_ylabel('K(t) [rad²/ps²]')
ax.set_title('Free-LSQ kernel vs stride')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)

plt.savefig('check_barrier_and_stride.png', dpi=160, bbox_inches='tight')
print("\nSaved check_barrier_and_stride.png")
