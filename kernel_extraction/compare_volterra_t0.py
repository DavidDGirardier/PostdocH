#!/usr/bin/env python3
"""
Compare Volterra kernel extraction using multiple time origins.

For each t0 = 0, 1, ..., t0_max, extract K with pre-memory correction
(using K from t0=0), then average over all t0 to reduce noise.

Shows the improvement over single-t0 Volterra as a function of N_trajs.
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from nonstationary_kernel import compute_kernel, volterra_second_kind_rect
from compare_volterra_lsq import generate_1d_gle_colored


def extract_kernel_at_t0(x, v, a_orth, Cvv_full, dt, trunc, t0, K_est):
    """
    Volterra at time origin t0 with pre-memory correction.

    x, v, a_orth : (N, T) arrays  (a_orth = a - k*x already subtracted)
    Cvv_full     : None (computed internally per t0)
    K_est        : kernel estimate for pre-memory subtraction
    """
    N, T = v.shape
    trunc_eff = min(trunc, T - t0)

    # Centered v(t0)
    v_t0 = v[:, t0] - np.mean(v[:, t0])

    # <v(t0) v(k)> for all k
    Cvv = v_t0 @ v / N
    Cvv_sliced = Cvv[t0:t0 + trunc_eff]

    # <v(t0) a_orth(t0+tau)>
    C_orth = v_t0 @ a_orth[:, t0:t0 + trunc_eff] / N

    # Pre-memory correction
    if t0 > 0 and K_est is not None:
        n_need = t0 + trunc_eff
        Kp = np.zeros(n_need)
        Kp[:min(len(K_est), n_need)] = K_est[:min(len(K_est), n_need)]

        for tau in range(trunc_eff):
            for j in range(tau + 1, t0 + tau + 1):
                vidx = t0 + tau - j
                if j < len(Kp) and vidx >= 0:
                    C_orth[tau] += dt * Kp[j] * Cvv[vidx]

    # Solve corrected Volterra (second kind, FD derivatives)
    Cprime = np.gradient(C_orth, dt)
    Bprime = np.gradient(Cvv_sliced, dt)
    B0 = Cvv_sliced[0]

    K, time = volterra_second_kind_rect(Cprime, Bprime, B0, dt)
    return K, time


def extract_kernel_multi_t0(x_trajs, v_trajs, a_trajs, k_force, dt,
                             trunc, t0_max):
    """
    Extract kernel by averaging corrected Volterra over t0 = 0 .. t0_max.

    1. Get K_est from t0=0.
    2. For each t0 = 0 .. t0_max, extract K with pre-memory correction.
    3. Average.
    """
    N = len(x_trajs)
    T_min = min(len(xi) for xi in x_trajs)

    x = np.array([xi[:T_min] for xi in x_trajs])
    v = np.array([vi[:T_min] for vi in v_trajs])
    a = np.array([ai[:T_min] for ai in a_trajs])
    a_orth = a - k_force * x

    # Step 1: K at t0=0
    res0 = compute_kernel(x_trajs, v_trajs, a_trajs,
                          k_force=k_force, dt=dt, trunc=trunc,
                          method="second_kind_rect")
    K_est = res0["K"]
    time0 = res0["time"]

    # Step 2: accumulate K from all t0
    n_pts = len(K_est)
    K_sum = np.zeros(n_pts)
    count = 0

    for t0 in range(0, t0_max + 1):
        K_t0, time_t0 = extract_kernel_at_t0(
            x, v, a_orth, None, dt, trunc, t0, K_est)
        n = min(len(K_t0), n_pts)
        K_sum[:n] += K_t0[:n]
        count += 1

    K_avg = K_sum / count
    return {"K": K_avg, "time": time0[:n_pts], "K_t0_0": K_est}


# ============================================================

if __name__ == "__main__":
    # --- Parameters ---
    k_force = -4.0
    gamma = 10.0
    tau_mem = 0.1
    kBT = 1.0
    dt = 0.001

    def K_true(t):
        return gamma * np.exp(-t / tau_mem)

    n_kernel = 1000
    trunc = n_kernel
    t0_max_list = [0, 50, 200]   # t0=0 means single origin (no averaging)
    nsteps = trunc + max(t0_max_list) + 50

    N_list = [100, 500, 2000, 10000]
    n_cols = len(N_list)

    # --- Figure ---
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7),
                             constrained_layout=True)

    t_ref = np.arange(n_kernel) * dt
    K_ref = K_true(t_ref)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)

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

        for ci, t0_max in enumerate(t0_max_list):
            if t0_max == 0:
                lab = "t0=0 only"
                print(f"  Volterra t0=0...")
                res = compute_kernel(x_trajs, v_trajs, a_trajs,
                                     k_force=k_force, dt=dt, trunc=trunc,
                                     method="second_kind_rect")
            else:
                lab = f"avg t0=0..{t0_max}"
                print(f"  Volterra avg t0=0..{t0_max}...")
                res = extract_kernel_multi_t0(
                    x_trajs, v_trajs, a_trajs, k_force, dt,
                    trunc, t0_max)

            n = min(len(res["time"]), n_kernel)
            ax1.plot(res["time"][:n], res["K"][:n], lw=1.5, alpha=0.7,
                     color=f"C{ci}", label=lab)
            intK = scipy.integrate.cumulative_trapezoid(
                res["K"][:n], res["time"][:n], initial=0)
            ax2.plot(res["time"][:n], intK, lw=1.5, alpha=0.7,
                     color=f"C{ci}", label=lab)

        ax1.set_title(f"N = {N_trajs}")
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"$K(t)$")
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("time")
        ax2.set_ylabel(r"$\int K$")
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

    plt.savefig("compare_volterra_t0.png", dpi=150)
    plt.show()
    print("\nDone.")
