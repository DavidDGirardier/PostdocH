#!/usr/bin/env python3
"""
Summary figure: LSQ kernel extraction validation.

4-panel figure:
  (a) Equilibrium: Volterra vs LSQ vs true K
  (b) Out-of-equilibrium: x0=0 vs x0=2.0 (displaced 4 sigma)
  (c) Cumulative integral comparison
  (d) Error vs method/scenario bar chart
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from nonstationary_kernel import compute_kernel
from nonstationary_kernel_lsq import extract_kernel_lsq
from compare_volterra_lsq import generate_1d_gle_colored

# --- Parameters ---
K_FORCE = -4.0
GAMMA = 10.0
TAU_MEM = 0.1
KBT = 1.0
DT = 0.001
N_KERNEL = 500
TAU_MAX = 400
NSTEPS = N_KERNEL + 250
N_COMPARE = 200

K_true = lambda t: GAMMA * np.exp(-t / TAU_MEM)

def rel_l2(K_ext, t_ext, n_pts=N_COMPARE):
    Kt = K_true(t_ext[:n_pts])
    num = np.trapz((K_ext[:n_pts] - Kt)**2, t_ext[:n_pts])
    den = np.trapz(Kt**2, t_ext[:n_pts])
    return np.sqrt(num / den)


# --- Generate trajectories ---
N_trajs = 10000
print("Generating trajectories...")
rng0 = np.random.default_rng(42)
xt_eq, vt_eq, at_eq = generate_1d_gle_colored(
    K_FORCE, GAMMA, TAU_MEM, KBT, DT, NSTEPS, N_trajs, x0=0.0, rng=rng0)

rng2 = np.random.default_rng(42)
xt_dis, vt_dis, at_dis = generate_1d_gle_colored(
    K_FORCE, GAMMA, TAU_MEM, KBT, DT, NSTEPS, N_trajs, x0=2.0, rng=rng2)

# --- Extract kernels ---
print("Volterra t0=0 (equilibrium)...")
res_volt = compute_kernel(xt_eq, vt_eq, at_eq, K_FORCE, DT, N_KERNEL,
                          method="second_kind_rect")

print("LSQ t0=0 only (equilibrium)...")
res_lsq0 = extract_kernel_lsq(xt_eq, vt_eq, at_eq, DT, N_KERNEL,
                                t0_max_idx=0, tau_max_idx=TAU_MAX,
                                k_force=K_FORCE)

print("LSQ multi-t0 (equilibrium)...")
res_lsq = extract_kernel_lsq(xt_eq, vt_eq, at_eq, DT, N_KERNEL,
                               t0_max_idx=100, tau_max_idx=TAU_MAX,
                               k_force=K_FORCE)

print("LSQ multi-t0 (displaced x0=2)...")
res_lsq_dis = extract_kernel_lsq(xt_dis, vt_dis, at_dis, DT, N_KERNEL,
                                   t0_max_idx=100, tau_max_idx=TAU_MAX,
                                   k_force=K_FORCE)

# --- Compute errors ---
t_ref = np.arange(N_COMPARE) * DT
K_ref = K_true(t_ref)

errors = {
    "Volterra $t_0{=}0$": rel_l2(res_volt["K"], res_volt["time"]),
    "LSQ $t_0{=}0$": rel_l2(res_lsq0["K"], res_lsq0["time"]),
    r"LSQ $t_0{\leq}0.1$": rel_l2(res_lsq["K"], res_lsq["time"]),
    r"LSQ $t_0{\leq}0.1$" + "\n($x_0{=}2$)": rel_l2(res_lsq_dis["K"], res_lsq_dis["time"]),
}

K0_values = {
    "Volterra $t_0{=}0$": res_volt["K"][0],
    "LSQ $t_0{=}0$": res_lsq0["K"][0],
    r"LSQ $t_0{\leq}0.1$": res_lsq["K"][0],
    r"LSQ $t_0{\leq}0.1$" + "\n($x_0{=}2$)": res_lsq_dis["K"][0],
}

for name, err in errors.items():
    print(f"  {name.replace(chr(10), ' '):30s}  err={err:.4f}  K(0)={K0_values[name]:.2f}")

# --- Figure ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

# (a) Kernel: equilibrium comparison
ax = axes[0, 0]
n = N_COMPARE
ax.plot(t_ref, K_ref, "k--", lw=2.5, label=r"$K_{\rm true}(t) = \gamma\,e^{-t/\tau}$",
        zorder=10)
ax.plot(res_volt["time"][:n], res_volt["K"][:n], lw=1.5, alpha=0.8,
        color="C0", label=f"Volterra $t_0{{=}}0$ (err={errors['Volterra $t_0{=}0$']:.1%})")
ax.plot(res_lsq0["time"][:n], res_lsq0["K"][:n], lw=1.5, alpha=0.8,
        color="C1", ls="-.", label=f"LSQ $t_0{{=}}0$ (err={errors['LSQ $t_0{=}0$']:.1%})")
lsq_key = r"LSQ $t_0{\leq}0.1$"
lsq_err = errors[lsq_key]
ax.plot(res_lsq["time"][:n], res_lsq["K"][:n], lw=1.5, alpha=0.8,
        color="C2", label=f"LSQ $t_0{{\\leq}}0.1$ (err={lsq_err:.1%})")

ax.set_xlabel("time")
ax.set_ylabel(r"$K(t)$")
ax.set_title(f"(a)  Equilibrium ($x_0=0$, $N={N_trajs}$)")
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(0, 0.3)
ax.grid(True, alpha=0.3)

# (b) Out-of-equilibrium: x0=0 vs x0=2
ax = axes[0, 1]
ax.plot(t_ref, K_ref, "k--", lw=2.5, label=r"$K_{\rm true}$", zorder=10)
ax.plot(res_lsq["time"][:n], res_lsq["K"][:n], lw=1.5, alpha=0.8,
        color="C2", label=r"$x_0 = 0$")
ax.plot(res_lsq_dis["time"][:n], res_lsq_dis["K"][:n], lw=1.5, alpha=0.8,
        color="C3", ls="--", label=r"$x_0 = 2$ (4$\sigma$ displaced)")

ax.set_xlabel("time")
ax.set_ylabel(r"$K(t)$")
ax.set_title(r"(b)  LSQ $t_0{\leq}0.1$: independence of $x_0$")
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(0, 0.3)
ax.grid(True, alpha=0.3)

# (c) Cumulative integral
ax = axes[1, 0]
intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)
ax.plot(t_ref, intK_ref, "k--", lw=2.5, label="true", zorder=10)

for res, lab, color, ls in [
    (res_volt, "Volterra $t_0{=}0$", "C0", "-"),
    (res_lsq0, "LSQ $t_0{=}0$", "C1", "-."),
    (res_lsq, r"LSQ $t_0{\leq}0.1$", "C2", "-"),
    (res_lsq_dis, r"LSQ $t_0{\leq}0.1$ ($x_0{=}2$)", "C3", "--"),
]:
    Ki = res["K"][:n]
    ti = res["time"][:n]
    intK = scipy.integrate.cumulative_trapezoid(Ki, ti, initial=0)
    ax.plot(ti, intK, lw=1.5, alpha=0.8, color=color, ls=ls, label=lab)

ax.set_xlabel("time")
ax.set_ylabel(r"$\int_0^t K(s)\,\mathrm{d}s$")
ax.set_title("(c)  Cumulative kernel integral")
ax.legend(fontsize=6, loc="lower right")
ax.set_xlim(0, 0.3)
ax.grid(True, alpha=0.3)

# (d) Error bar chart
ax = axes[1, 1]
names = list(errors.keys())
vals = [errors[n] * 100 for n in names]
k0_vals = [K0_values[n] for n in names]
colors = ["C0", "C1", "C2", "C3"]
short_names = ["Volterra\n$t_0{=}0$", "LSQ\n$t_0{=}0$",
               "LSQ\n$t_0{\\leq}0.1$", "LSQ\n$t_0{\\leq}0.1$\n$x_0{=}2$"]
bars = ax.bar(short_names, vals, color=colors, alpha=0.7, edgecolor="k", lw=0.5)

for bar, v, k0 in zip(bars, vals, k0_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{v:.1f}%\n$K(0)$={k0:.1f}",
            ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Relative $L^2$ error (%)")
ax.set_title(f"(d)  Error comparison ($N={N_trajs}$)")
ax.set_ylim(0, max(vals) * 1.6)
ax.axhline(y=0, color="k", lw=0.5)
ax.grid(True, alpha=0.3, axis="y")

plt.savefig("fig_lsq_summary.png", dpi=200, bbox_inches="tight")
plt.savefig("fig_lsq_summary.pdf", bbox_inches="tight")
print("\nSaved fig_lsq_summary.png / .pdf")
plt.show()
