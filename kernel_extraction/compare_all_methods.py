#!/usr/bin/env python3
"""
Compare three kernel extraction methods side by side:
  1. Volterra t0=0 only (standard)
  2. Averaged Volterra over t0=0..t0_max (with pre-memory correction)
  3. LSQ over t0=0..t0_max

All on the same 1D harmonic GLE trajectories (cold start, a_det).
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from nonstationary_kernel import compute_kernel, volterra_second_kind_rect
from nonstationary_kernel_lsq import extract_kernel_lsq
from compare_volterra_lsq import generate_1d_gle_colored


# ============================================================
# Averaged Volterra with pre-memory correction
# ============================================================

def extract_kernel_at_t0(x, v, a_orth, dt, trunc, t0, K_est):
    """Single-t0 Volterra with pre-memory correction."""
    N, T = v.shape
    trunc_eff = min(trunc, T - t0)

    v_t0 = v[:, t0] - np.mean(v[:, t0])
    Cvv = v_t0 @ v / N
    Cvv_sliced = Cvv[t0:t0 + trunc_eff]
    C_orth = v_t0 @ a_orth[:, t0:t0 + trunc_eff] / N

    if t0 > 0 and K_est is not None:
        n_need = t0 + trunc_eff
        Kp = np.zeros(n_need)
        Kp[:min(len(K_est), n_need)] = K_est[:min(len(K_est), n_need)]
        for tau in range(trunc_eff):
            for j in range(tau + 1, t0 + tau + 1):
                vidx = t0 + tau - j
                if j < len(Kp) and vidx >= 0:
                    C_orth[tau] += dt * Kp[j] * Cvv[vidx]

    Cprime = np.gradient(C_orth, dt)
    Bprime = np.gradient(Cvv_sliced, dt)
    B0 = Cvv_sliced[0]
    K, time = volterra_second_kind_rect(Cprime, Bprime, B0, dt)
    return K, time


def extract_kernel_avg_volterra(x_trajs, v_trajs, a_trajs, k_force, dt,
                                 trunc, t0_max):
    """Average corrected Volterra over t0 = 0 .. t0_max."""
    N = len(x_trajs)
    T_min = min(len(xi) for xi in x_trajs)
    x = np.array([xi[:T_min] for xi in x_trajs])
    v = np.array([vi[:T_min] for vi in v_trajs])
    a = np.array([ai[:T_min] for ai in a_trajs])
    a_orth = a - k_force * x

    # K_est from t0=0
    res0 = compute_kernel(x_trajs, v_trajs, a_trajs,
                          k_force=k_force, dt=dt, trunc=trunc,
                          method="second_kind_rect")
    K_est = res0["K"]
    n_pts = len(K_est)

    K_sum = np.zeros(n_pts)
    count = 0
    for t0 in range(0, t0_max + 1):
        K_t0, _ = extract_kernel_at_t0(x, v, a_orth, dt, trunc, t0, K_est)
        n = min(len(K_t0), n_pts)
        K_sum[:n] += K_t0[:n]
        count += 1

    return {"K": K_sum / count, "time": res0["time"][:n_pts]}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    k_force = -4.0
    gamma = 10.0
    tau_mem = 0.1
    kBT = 1.0
    dt = 0.001

    def K_true(t):
        return gamma * np.exp(-t / tau_mem)

    n_kernel = 1000
    trunc = n_kernel
    tau_max_lsq = 800
    t0_max_list = [50, 200]
    nsteps = trunc + max(t0_max_list) + 50

    N_list = [100, 500, 2000, 10000]
    n_cols = len(N_list)

    t_ref = np.arange(n_kernel) * dt
    K_ref = K_true(t_ref)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7),
                             constrained_layout=True)

    for col, N_trajs in enumerate(N_list):
        rng = np.random.default_rng(42)
        print(f"\n=== N = {N_trajs} ===")
        x_trajs, v_trajs, a_trajs = generate_1d_gle_colored(
            k_force, gamma, tau_mem, kBT, dt, nsteps, N_trajs,
            x0=0.0, rng=rng)

        ax1 = axes[0, col]
        ax2 = axes[1, col]
        ax1.plot(t_ref, K_ref, "k--", lw=2, label="true", zorder=10)
        ax2.plot(t_ref, intK_ref, "k--", lw=2, label="true", zorder=10)

        # --- 1. Volterra t0=0 ---
        print("  Volterra t0=0...")
        res_v = compute_kernel(x_trajs, v_trajs, a_trajs,
                               k_force=k_force, dt=dt, trunc=trunc,
                               method="second_kind_rect")
        n = min(len(res_v["time"]), n_kernel)
        ax1.plot(res_v["time"][:n], res_v["K"][:n],
                 lw=1.5, alpha=0.7, color="C0", label="Volterra t0=0")
        intK = scipy.integrate.cumulative_trapezoid(
            res_v["K"][:n], res_v["time"][:n], initial=0)
        ax2.plot(res_v["time"][:n], intK,
                 lw=1.5, alpha=0.7, color="C0", label="Volterra t0=0")

        # --- 1b. LSQ t0=0 only ---
        print("  LSQ t0=0...")
        res_lsq0 = extract_kernel_lsq(
            x_trajs, v_trajs, a_trajs, dt,
            n_kernel=n_kernel, t0_max_idx=0,
            tau_max_idx=tau_max_lsq, k_force=k_force)
        n_l0 = min(len(res_lsq0["time"]), n_kernel)
        ax1.plot(res_lsq0["time"][:n_l0], res_lsq0["K"][:n_l0],
                 lw=1.5, alpha=0.7, color="C0", ls="-.",
                 label="LSQ t0=0")
        intK_l0 = scipy.integrate.cumulative_trapezoid(
            res_lsq0["K"][:n_l0], res_lsq0["time"][:n_l0], initial=0)
        ax2.plot(res_lsq0["time"][:n_l0], intK_l0,
                 lw=1.5, alpha=0.7, color="C0", ls="-.",
                 label="LSQ t0=0")

        # --- 2 & 3: averaged Volterra and LSQ for each t0_max ---
        for ci, t0_max in enumerate(t0_max_list):
            # Averaged Volterra
            print(f"  Avg Volterra t0=0..{t0_max}...")
            res_av = extract_kernel_avg_volterra(
                x_trajs, v_trajs, a_trajs, k_force, dt, trunc, t0_max)
            n_av = min(len(res_av["time"]), n_kernel)
            ax1.plot(res_av["time"][:n_av], res_av["K"][:n_av],
                     lw=1.5, alpha=0.7, color=f"C{2*ci+1}",
                     label=f"avg Volt 0..{t0_max}")
            intK_av = scipy.integrate.cumulative_trapezoid(
                res_av["K"][:n_av], res_av["time"][:n_av], initial=0)
            ax2.plot(res_av["time"][:n_av], intK_av,
                     lw=1.5, alpha=0.7, color=f"C{2*ci+1}",
                     label=f"avg Volt 0..{t0_max}")

            # LSQ
            print(f"  LSQ t0=0..{t0_max}...")
            res_lsq = extract_kernel_lsq(
                x_trajs, v_trajs, a_trajs, dt,
                n_kernel=n_kernel, t0_max_idx=t0_max,
                tau_max_idx=tau_max_lsq, k_force=k_force)
            n_lsq = min(len(res_lsq["time"]), n_kernel)
            ax1.plot(res_lsq["time"][:n_lsq], res_lsq["K"][:n_lsq],
                     lw=1.5, alpha=0.7, color=f"C{2*ci+2}", ls="-.",
                     label=f"LSQ 0..{t0_max}")
            intK_lsq = scipy.integrate.cumulative_trapezoid(
                res_lsq["K"][:n_lsq], res_lsq["time"][:n_lsq], initial=0)
            ax2.plot(res_lsq["time"][:n_lsq], intK_lsq,
                     lw=1.5, alpha=0.7, color=f"C{2*ci+2}", ls="-.",
                     label=f"LSQ 0..{t0_max}")

        ax1.set_title(f"N = {N_trajs}")
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"$K(t)$")
        ax1.legend(fontsize=5)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("time")
        ax2.set_ylabel(r"$\int K$")
        ax2.legend(fontsize=5)
        ax2.grid(True, alpha=0.3)

    plt.savefig("compare_all_methods.png", dpi=150)
    plt.show()
    print("\nDone.")
