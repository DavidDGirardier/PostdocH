#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Compare closure quality with raw vs smoothed Gram.

Expects in cwd:
    Kdecomp_pos_raw.npz       (SMOOTH_GRAM=False run)
    Kdecomp_pos_smooth.npz    (SMOOTH_GRAM=True  run)
"""

import numpy as np
import matplotlib.pyplot as plt

raw    = np.load("Kdecomp_pos_raw.npz")
smooth = np.load("Kdecomp_pos_smooth.npz")

def rel_err_per_angle(d):
    K0   = d["K0_xt"]
    Ksum = d["K11_xt"] + d["K22_xt"] + d["K12_xt"] + d["K21_xt"]
    time = d["time"]
    mask = np.ones(len(time), bool); mask[1] = False  # drop trapz spike
    err = np.max(np.abs(K0[mask] - Ksum[mask]), axis=0)
    nrm = np.maximum(np.max(np.abs(K0[mask]), axis=0), 1e-12)
    return err / nrm

xf       = raw["xfine"]
rel_raw  = rel_err_per_angle(raw)
rel_sm   = rel_err_per_angle(smooth)

print(f"{'angle [deg]':>12s}  {'raw':>10s}  {'smooth':>10s}")
for i in range(0, len(xf), 20):
    print(f"{np.rad2deg(xf[i]):>12.1f}  {rel_raw[i]:>10.1%}  {rel_sm[i]:>10.1%}")

print()
print(f"median rel error  raw   : {np.median(rel_raw):.1%}")
print(f"median rel error  smooth: {np.median(rel_sm):.1%}")
print(f"max    rel error  raw   : {np.max(rel_raw):.1%}")
print(f"max    rel error  smooth: {np.max(rel_sm):.1%}")

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.plot(np.rad2deg(xf), 100 * rel_raw, label="raw G_c (bkbkcorrw[t=0])", lw=2)
ax.plot(np.rad2deg(xf), 100 * rel_sm,  label="smoothed G_c (M_phi^-1 fit)", lw=2)
for i in raw["minima_idx"]:
    ax.axvline(np.rad2deg(xf[i]), color="g", ls="--", alpha=0.4)
for i in raw["ts_idx"]:
    ax.axvline(np.rad2deg(xf[i]), color="r", ls="--", alpha=0.4)
ax.set_xlabel(r"$\varphi$ [deg]")
ax.set_ylabel("relative closure error  max_t |K_tot - sum K_ij| / max_t |K_tot|  [%]")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig("compare_gram_methods.png", dpi=130)
plt.show()
