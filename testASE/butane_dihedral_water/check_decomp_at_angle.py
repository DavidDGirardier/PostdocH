#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Three-level closure diagnostic at fixed angles phi*:

  Level (a) -- per-component algebra (should pass to ~machine precision):
        K_tilde_1(phi*, t) ?= K11(phi*, t) + K12(phi*, t)
        K_tilde_2(phi*, t) ?= K21(phi*, t) + K22(phi*, t)
    Tests: trapz Volterra solver vs second-kind RHS form, *within one model*.

  Level (b) -- per-component-sum closure:
        K_tot(phi*, t) ?= K_tilde_1(phi*, t) + K_tilde_2(phi*, t)
    Tests: linearity of the Volterra solver across the L_obs decomposition
    (= mean-force projection F_1+F_2 == F_tot at finite spline).

  Level (c) -- global closure (the union of (a) and (b)):
        K_tot(phi*, t) ?= K11 + K12 + K21 + K22

If (a) passes but (b) fails: pool seeds, the Gram is the same and only the
mean-force projection has finite-sample noise.
If (a) fails: something inside one Pos_gle fit (smoothing, k0, conditioning).
If both pass: closure is good, finite t residual is sample noise.

Run AFTER PositionalMemoryDecompMeanForce_errorbars.py once Kdecomp_pos.npz
contains K1_xt, K2_xt as well as K0_xt and K11..K22.
"""

import numpy as np
import matplotlib.pyplot as plt

d = np.load("Kdecomp_pos.npz")
time      = d["time"]
xfine     = d["xfine"]
K0_xt     = d["K0_xt"]
K1_xt     = d["K1_xt"]
K2_xt     = d["K2_xt"]
K11_xt    = d["K11_xt"]
K22_xt    = d["K22_xt"]
K12_xt    = d["K12_xt"]
K21_xt    = d["K21_xt"]
K0_fd_pos = d["K0_fd_pos"] if "K0_fd_pos" in d.files else None
minima    = d["minima_idx"]
ts        = d["ts_idx"]

# the three closure quantities
sum_a1   = K11_xt + K12_xt                    # should == K1_xt   (level a)
sum_a2   = K21_xt + K22_xt                    # should == K2_xt   (level a)
sum_K    = K1_xt + K2_xt                      # should == K0_xt   (level b)
sum_full = K11_xt + K12_xt + K21_xt + K22_xt  # should == K0_xt   (level c)

# Pick angles to check: all FE minima + all transition states
check_idx = list(minima) + list(ts)
labels    = [f"min @ {np.rad2deg(xfine[i]):.0f}°" for i in minima] \
          + [f"TS  @ {np.rad2deg(xfine[i]):.0f}°" for i in ts]

n = len(check_idx)
fig, axes = plt.subplots(3, n, figsize=(4 * n, 11), constrained_layout=True,
                         sharex=True)
if n == 1:
    axes = axes[:, None]

print(f"{'angle':<25s}  {'|K_tot|':>10s}  "
      f"{'(a1) K1=K11+K12':>16s}  "
      f"{'(a2) K2=K21+K22':>16s}  "
      f"{'(b) K0=K1+K2':>14s}")

for col, (idx, lab) in enumerate(zip(check_idx, labels)):
    K_tot_i = K0_xt[:, idx]
    norm    = max(np.max(np.abs(K_tot_i)), 1e-12)

    # --- level (a) per-component
    ax_a = axes[0, col]
    ax_a.plot(time[1:], K1_xt[1:, idx],   lw=2,           label=r"$\tilde K_1$")
    ax_a.plot(time[1:], sum_a1[1:, idx],  lw=2, ls="--",  label=r"$K_{11}+K_{12}$")
    ax_a.plot(time[1:], K2_xt[1:, idx],   lw=2,           label=r"$\tilde K_2$")
    ax_a.plot(time[1:], sum_a2[1:, idx],  lw=2, ls="--",  label=r"$K_{21}+K_{22}$")
    ax_a.set_xscale("log"); ax_a.axhline(0, lw=1, alpha=0.5)
    ax_a.grid(True, alpha=0.25)
    ax_a.set_title(lab + r"  --  per-component (a)")
    ax_a.legend(fontsize=7)
    if col == 0:
        ax_a.set_ylabel(r"$K(\varphi^*, t)$")

    # --- level (b) per-component sum
    ax_b = axes[1, col]
    ax_b.plot(time[1:], K_tot_i[1:],          lw=2,           label=r"$K_\mathrm{tot}$")
    ax_b.plot(time[1:], sum_K[1:, idx],       lw=2, ls="--",  label=r"$\tilde K_1+\tilde K_2$")
    ax_b.set_xscale("log"); ax_b.axhline(0, lw=1, alpha=0.5)
    ax_b.grid(True, alpha=0.25)
    ax_b.set_title(r"per-component sum (b)")
    ax_b.legend(fontsize=7)
    if col == 0:
        ax_b.set_ylabel(r"$K(\varphi^*, t)$")

    # --- level (c) global closure
    ax_c = axes[2, col]
    ax_c.plot(time[1:], K_tot_i[1:],          lw=2,           label=r"$K_\mathrm{tot}$")
    ax_c.plot(time[1:], sum_full[1:, idx],    lw=2, ls="--",  label=r"$\sum_{ij} K_{ij}$")
    ax_c.set_xscale("log"); ax_c.axhline(0, lw=1, alpha=0.5)
    ax_c.grid(True, alpha=0.25)
    ax_c.set_title(r"global closure (c)")
    ax_c.set_xlabel("time [ps]")
    ax_c.legend(fontsize=7)
    if col == 0:
        ax_c.set_ylabel(r"$K(\varphi^*, t)$")

    # --- printable diagnostic
    err_a1 = np.max(np.abs(K1_xt[:, idx] - sum_a1[:, idx])) / norm
    err_a2 = np.max(np.abs(K2_xt[:, idx] - sum_a2[:, idx])) / norm
    err_b  = np.max(np.abs(K_tot_i        - sum_K[:, idx])) / norm
    print(f"{lab:<25s}  {norm:10.3e}  "
          f"{err_a1:16.2%}  {err_a2:16.2%}  {err_b:14.2%}")

plt.savefig("check_decomp_at_angle.png", dpi=130)
plt.show()
