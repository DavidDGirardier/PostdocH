#!/usr/bin/env python3
"""
1D harmonic GLE with exponential memory kernel (colored noise).

Auxiliary variables (s, n) start in equilibrium.
Compare kernel extraction:
  - Volterra at different individual starting points t0
  - LSQ with different t0 ranges

Shows the impact on convergence with N_trajs.

System:
    dx = v dt
    dv = (-w0^2 x - gamma s + sqrt(gamma) n) dt
    ds = (v - s/tau) dt
    dn = (-n/tau) dt + sqrt(2 kBT / tau) dW

True kernel:  K(t) = gamma exp(-t/tau)

Records a_total = -w0^2 x - gamma s + sqrt(gamma) n  (total force on v,
as would be measured in MD — includes colored noise).
"""

import numpy as np
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt
from nonstationary_kernel import compute_kernel
from nonstationary_kernel_lsq import extract_kernel_lsq


# ============================================================
# Model
# ============================================================

def eq_covariance(w0sq, gamma, tau, kBT):
    """Stationary covariance of (x, v, s, n) from Lyapunov equation."""
    sg = np.sqrt(gamma)
    A = np.array([
        [0,     1,      0,      0     ],
        [-w0sq, 0,     -gamma,  sg    ],
        [0,     1,     -1/tau,  0     ],
        [0,     0,      0,     -1/tau ],
    ])
    D = np.zeros((4, 4))
    D[3, 3] = 2 * kBT / tau
    Sigma = scipy.linalg.solve_continuous_lyapunov(A, -D)
    return 0.5 * (Sigma + Sigma.T)


def generate(w0sq, gamma, tau, kBT, dt, nsteps, N,
             aux_mode="equilibrium", rng=None):
    """
    Generate 1D GLE trajectories with colored noise.
    Records a_total = -w0^2 x - gamma s + sqrt(gamma) n  (total accel).
    This is what you'd measure in MD (includes colored noise).
    """
    if rng is None:
        rng = np.random.default_rng()

    sg = np.sqrt(gamma)
    exp_dec = np.exp(-dt / tau)
    ou_std = np.sqrt(kBT * (1 - exp_dec**2))

    x = np.zeros((N, nsteps))
    v = np.zeros((N, nsteps))
    a = np.zeros((N, nsteps))   # total acceleration (incl. colored noise)
    s = np.zeros(N)
    n = np.zeros(N)

    if aux_mode == "equilibrium":
        Sigma = eq_covariance(w0sq, gamma, tau, kBT)
        L = np.linalg.cholesky(Sigma)
        init = rng.normal(size=(N, 4)) @ L.T
        x[:, 0], v[:, 0], s[:], n[:] = init.T
    elif aux_mode == "cold":
        x[:, 0] = rng.normal(0, np.sqrt(kBT / w0sq), N)
        v[:, 0] = rng.normal(0, np.sqrt(kBT), N)
    else:
        raise ValueError(aux_mode)

    k = -w0sq
    for i in range(nsteps - 1):
        a[:, i] = k * x[:, i] - gamma * s + sg * n
        v[:, i + 1] = v[:, i] + dt * a[:, i]
        x[:, i + 1] = x[:, i] + dt * v[:, i]
        s = s * exp_dec + v[:, i] * dt
        n = n * exp_dec + ou_std * rng.normal(size=N)

    a[:, -1] = k * x[:, -1] - gamma * s + sg * n
    return list(x), list(v), list(a)


# ============================================================
# Error metric
# ============================================================

def rel_l2(K_ext, t_ext, K_true, n_pts):
    nc = min(n_pts, len(K_ext))
    Kt = K_true(t_ext[:nc])
    num = np.trapz((K_ext[:nc] - Kt)**2, t_ext[:nc])
    den = np.trapz(Kt**2, t_ext[:nc])
    return np.sqrt(num / den) if den > 0 else np.inf


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --- Parameters ---
    w0sq   = 4.0
    gamma  = 10.0
    tau    = 0.1
    kBT    = 1.0
    dt     = 0.01
    k_force = -w0sq

    K_true = lambda t: gamma * np.exp(-t / tau)

    nsteps      = 300          # traj time = 3.0
    trunc       = 80           # Volterra truncation = 0.8
    n_kernel    = 50           # LSQ kernel length = 0.5
    tau_max_idx = 50
    n_compare   = 30           # error over first 0.3 = 3 tau

    # --- Starting points for Volterra ---
    t0_volterra = [0, 10, 30, 60]    # individual starting frames

    # --- t0 ranges for LSQ ---
    t0_max_lsq = [0, 10, 30]         # LSQ uses t0 = 0 .. t0_max

    # --- Convergence ---
    N_list = [500, 1000, 2000, 5000, 10000, 20000]
    n_rep  = 5

    # --- Equilibrium check ---
    Sigma = eq_covariance(w0sq, gamma, tau, kBT)
    sv = Sigma[2, 1]
    print("Equilibrium covariance:")
    print(f"  <xx>={Sigma[0,0]:.4f} (expect {kBT/w0sq:.4f})")
    print(f"  <vv>={Sigma[1,1]:.4f} (expect {kBT:.4f})")
    print(f"  <sv>={sv:.5f},  <nv>={Sigma[3,1]:.5f}")
    print(f"  Bias at tau=0 from s(0): -gamma*<sv> = {-gamma*sv:.4f}")
    print()

    # ============================================================
    # Part 1: Volterra at different starting points
    # ============================================================
    print("=== Volterra at different starting t0 ===")
    all_err_volt = {}
    for T0 in t0_volterra:
        lab = f"Volterra t0={T0}"
        errs = np.zeros((len(N_list), n_rep))
        for ni, N in enumerate(N_list):
            for r in range(n_rep):
                rng = np.random.default_rng(1000 * r + N)
                xt, vt, at = generate(w0sq, gamma, tau, kBT,
                                      dt, nsteps, N, "equilibrium", rng)
                # Slice trajectories from frame T0
                xt_s = [xi[T0:] for xi in xt]
                vt_s = [vi[T0:] for vi in vt]
                at_s = [ai[T0:] for ai in at]
                res = compute_kernel(xt_s, vt_s, at_s, k_force,
                                     dt, trunc, "second_kind_rect")
                errs[ni, r] = rel_l2(res["K"], res["time"],
                                     K_true, n_compare)
        all_err_volt[lab] = errs
        print(f"  {lab:25s}  N={N_list[-1]:5d} -> "
              f"err={np.mean(errs[-1]):.4f} +/- {np.std(errs[-1]):.4f}")
    print()

    # ============================================================
    # Part 2: LSQ with different t0 ranges
    # ============================================================
    print("=== LSQ with different t0 ranges ===")
    all_err_lsq = {}
    for t0m in t0_max_lsq:
        lab = f"LSQ t0=0..{t0m}" if t0m > 0 else "LSQ t0=0"
        errs = np.zeros((len(N_list), n_rep))
        for ni, N in enumerate(N_list):
            for r in range(n_rep):
                rng = np.random.default_rng(1000 * r + N)
                xt, vt, at = generate(w0sq, gamma, tau, kBT,
                                      dt, nsteps, N, "equilibrium", rng)
                res = extract_kernel_lsq(
                    xt, vt, at, dt, n_kernel,
                    t0m, tau_max_idx, k_force=k_force)
                errs[ni, r] = rel_l2(res["K"], res["time"],
                                     K_true, n_compare)
        all_err_lsq[lab] = errs
        print(f"  {lab:25s}  N={N_list[-1]:5d} -> "
              f"err={np.mean(errs[-1]):.4f} +/- {np.std(errs[-1]):.4f}")
    print()

    # ============================================================
    # Figure 1 — Volterra at different t0: kernel examples
    # ============================================================
    N_show = 5000
    rng = np.random.default_rng(42)
    xt, vt, at = generate(w0sq, gamma, tau, kBT, dt, nsteps,
                          N_show, "equilibrium", rng)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   constrained_layout=True)
    t_ref = np.linspace(0, n_compare * dt, 300)
    nc = n_compare

    ax1.plot(t_ref, K_true(t_ref), "k--", lw=2.5,
             label=r"$K_{\mathrm{true}}$", zorder=10)
    ax2.plot(t_ref, scipy.integrate.cumulative_trapezoid(
        K_true(t_ref), t_ref, initial=0), "k--", lw=2.5,
             label="true", zorder=10)

    for ci, T0 in enumerate(t0_volterra):
        xt_s = [xi[T0:] for xi in xt]
        vt_s = [vi[T0:] for vi in vt]
        at_s = [ai[T0:] for ai in at]
        res = compute_kernel(xt_s, vt_s, at_s, k_force,
                             dt, trunc, "second_kind_rect")
        lab = f"t0={T0}  ({T0*dt:.1f})"
        ax1.plot(res["time"][:nc], res["K"][:nc],
                 lw=1.3, alpha=0.8, color=f"C{ci}", label=lab)
        intK = scipy.integrate.cumulative_trapezoid(
            res["K"][:nc], res["time"][:nc], initial=0)
        ax2.plot(res["time"][:nc], intK,
                 lw=1.3, alpha=0.8, color=f"C{ci}", label=lab)

    ax1.set_ylabel(r"$K(t)$")
    ax1.set_title(f"Volterra at different starting points (N={N_show})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\int_0^t K(s)\,ds$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.savefig("1d_volterra_starting_points.png", dpi=150)
    plt.show()

    # ============================================================
    # Figure 2 — LSQ with different t0 ranges: kernel examples
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   constrained_layout=True)

    ax1.plot(t_ref, K_true(t_ref), "k--", lw=2.5,
             label=r"$K_{\mathrm{true}}$", zorder=10)
    ax2.plot(t_ref, scipy.integrate.cumulative_trapezoid(
        K_true(t_ref), t_ref, initial=0), "k--", lw=2.5,
             label="true", zorder=10)

    # Also plot the Volterra t0=0 reference
    res_v = compute_kernel(xt, vt, at, k_force, dt, trunc,
                           "second_kind_rect")
    ax1.plot(res_v["time"][:nc], res_v["K"][:nc],
             lw=1.3, alpha=0.6, color="gray", ls=":",
             label="Volterra t0=0")

    for ci, t0m in enumerate(t0_max_lsq):
        res = extract_kernel_lsq(xt, vt, at, dt, n_kernel,
                                  t0m, tau_max_idx, k_force=k_force)
        lab = f"LSQ t0=0..{t0m}" if t0m > 0 else "LSQ t0=0"
        ax1.plot(res["time"][:nc], res["K"][:nc],
                 lw=1.5, alpha=0.8, color=f"C{ci+1}", label=lab)
        intK = scipy.integrate.cumulative_trapezoid(
            res["K"][:nc], res["time"][:nc], initial=0)
        ax2.plot(res["time"][:nc], intK,
                 lw=1.5, alpha=0.8, color=f"C{ci+1}", label=lab)

    intK_v = scipy.integrate.cumulative_trapezoid(
        res_v["K"][:nc], res_v["time"][:nc], initial=0)
    ax2.plot(res_v["time"][:nc], intK_v,
             lw=1.3, alpha=0.6, color="gray", ls=":", label="Volterra t0=0")

    ax1.set_ylabel(r"$K(t)$")
    ax1.set_title(f"LSQ with different t0 ranges (N={N_show})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\int_0^t K(s)\,ds$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.savefig("1d_lsq_t0_ranges.png", dpi=150)
    plt.show()

    # ============================================================
    # Figure 3 — Convergence: Volterra starting points
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for ci, T0 in enumerate(t0_volterra):
        lab = f"Volterra t0={T0}"
        errs = all_err_volt[lab]
        mu, sig = np.mean(errs, axis=1), np.std(errs, axis=1)
        ax.errorbar(N_list, mu, yerr=sig, color=f"C{ci}", marker="o",
                    capsize=3, lw=1.5, ms=5, label=lab)

    N_arr = np.array(N_list, dtype=float)
    ref = np.mean(all_err_volt[f"Volterra t0={t0_volterra[0]}"][-1])
    ax.plot(N_arr, ref * np.sqrt(N_list[-1] / N_arr),
            "k:", alpha=0.4, label=r"$\propto N^{-1/2}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of trajectories")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Volterra: different starting points")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.savefig("1d_convergence_volterra.png", dpi=150)
    plt.show()

    # ============================================================
    # Figure 4 — Convergence: LSQ t0 ranges
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    markers_lsq = ["s", "D", "^"]
    for ci, t0m in enumerate(t0_max_lsq):
        lab = f"LSQ t0=0..{t0m}" if t0m > 0 else "LSQ t0=0"
        errs = all_err_lsq[lab]
        mu, sig = np.mean(errs, axis=1), np.std(errs, axis=1)
        ax.errorbar(N_list, mu, yerr=sig, color=f"C{ci+1}",
                    marker=markers_lsq[ci],
                    capsize=3, lw=1.5, ms=5, label=lab)

    # Also add Volterra t0=0 for reference
    errs_v = all_err_volt["Volterra t0=0"]
    mu_v = np.mean(errs_v, axis=1)
    sig_v = np.std(errs_v, axis=1)
    ax.errorbar(N_list, mu_v, yerr=sig_v, color="C0", marker="o",
                capsize=3, lw=1.5, ms=5, label="Volterra t0=0")

    ax.plot(N_arr, ref * np.sqrt(N_list[-1] / N_arr),
            "k:", alpha=0.4, label=r"$\propto N^{-1/2}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of trajectories")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("LSQ: different t0 ranges")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.savefig("1d_convergence_lsq.png", dpi=150)
    plt.show()

    print("Done.")
