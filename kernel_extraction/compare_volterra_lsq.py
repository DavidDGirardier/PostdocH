#!/usr/bin/env python3
"""
Compare Volterra (t0=0 only) vs LSQ (multiple t0) for the 1D harmonic
GLE, as a function of number of trajectories.
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from nonstationary_kernel import (
    compute_kernel, volterra_second_kind_rect,
)
from nonstationary_kernel_lsq import extract_kernel_lsq


def generate_1d_gle_colored(k_force, gamma, tau_mem, kBT, dt, nsteps,
                            N_trajs, x0=0.0, rng=None):
    """
    1D GLE with exponential kernel and correct colored noise (OU).

    V(x) = 0.5*|k|*x^2,  F(x) = k*x,  K(t) = gamma*exp(-t/tau)
    """
    if rng is None:
        rng = np.random.default_rng()

    exp_decay = np.exp(-dt / tau_mem)
    ou_noise_std = np.sqrt(kBT * (1 - exp_decay**2))
    gamma_sqrt = np.sqrt(gamma)

    x = np.zeros((N_trajs, nsteps))
    v = np.zeros((N_trajs, nsteps))
    a_det = np.zeros((N_trajs, nsteps))
    s = np.zeros(N_trajs)       # memory auxiliary
    n_ou = rng.normal(0, np.sqrt(kBT), N_trajs)  # noise auxiliary

    x[:, 0] = x0
    v[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

    for i in range(nsteps - 1):
        F = k_force * x[:, i]
        mem = gamma * s
        noise = gamma_sqrt * n_ou
        a_det[:, i] = F - mem

        v[:, i + 1] = v[:, i] + dt * (F - mem + noise)
        x[:, i + 1] = x[:, i] + dt * v[:, i]
        s = s * exp_decay + v[:, i] * dt
        n_ou = n_ou * exp_decay + ou_noise_std * rng.normal(size=N_trajs)

    a_det[:, -1] = k_force * x[:, -1] - gamma * s

    return list(x), list(v), list(a_det)


# ============================================================
# Main comparison
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

    def F_exact(x):
        return k_force * x

    n_kernel = 1000      # kernel points -> t_f = 1.0
    trunc = n_kernel
    tau_max = 800        # LSQ lags
    t0_max_list = [0, 50, 200]  # different t0 ranges
    nsteps = trunc + max(t0_max_list) + 50

    N_list = [100, 500, 2000, 10000]
    n_cols = len(N_list)

    # --- Figure 1: kernel comparison ---
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7),
                             constrained_layout=True)

    t_ref = np.arange(n_kernel) * dt
    K_ref = K_true(t_ref)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)

    for col, N_trajs in enumerate(N_list):
        rng = np.random.default_rng(42)
        print(f"\n=== N = {N_trajs} ===")
        print("Generating trajectories...")
        x_trajs, v_trajs, a_trajs = generate_1d_gle_colored(
            k_force, gamma, tau_mem, kBT, dt, nsteps, N_trajs,
            x0=0.0, rng=rng)

        ax1 = axes[0, col]
        ax2 = axes[1, col]
        ax1.plot(t_ref, K_ref, "k--", lw=2, label="true", zorder=10)
        ax2.plot(t_ref, intK_ref, "k--", lw=2, label="true", zorder=10)

        # Volterra
        print("  Volterra t0=0...")
        res_v = compute_kernel(x_trajs, v_trajs, a_trajs,
                               k_force=k_force, dt=dt, trunc=trunc,
                               method="second_kind_rect")
        n = min(len(res_v["time"]), n_kernel)
        ax1.plot(res_v["time"][:n], res_v["K"][:n], lw=1.5, alpha=0.7,
                 label="Volterra t0=0")
        intK = scipy.integrate.cumulative_trapezoid(
            res_v["K"][:n], res_v["time"][:n], initial=0)
        ax2.plot(res_v["time"][:n], intK, lw=1.5, alpha=0.7,
                 label="Volterra t0=0")

        # LSQ with different t0_max
        for t0_max in t0_max_list:
            lab = f"LSQ t0≤{t0_max*dt:.1f}" if t0_max > 0 else "LSQ t0=0"
            print(f"  {lab}...")
            res = extract_kernel_lsq(
                x_trajs, v_trajs, a_trajs, dt,
                n_kernel=n_kernel, t0_max_idx=t0_max,
                tau_max_idx=tau_max, k_force=k_force)
            n = min(len(res["time"]), len(res["K"]), n_kernel)
            ax1.plot(res["time"][:n], res["K"][:n], lw=1.5, alpha=0.7,
                     label=lab)
            intK = scipy.integrate.cumulative_trapezoid(
                res["K"][:n], res["time"][:n], initial=0)
            ax2.plot(res["time"][:n], intK, lw=1.5, alpha=0.7, label=lab)

        ax1.set_title(f"N = {N_trajs}")
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"$K(t)$")
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("time")
        ax2.set_ylabel(r"$\int K$")
        ax2.grid(True, alpha=0.3)

    plt.savefig("compare_volterra_lsq.png", dpi=150)
    plt.show()

    # --- Figure 2: position distribution at each t0_max ---
    fig2, axes2 = plt.subplots(1, len(t0_max_list),
                               figsize=(4 * len(t0_max_list), 3.5),
                               constrained_layout=True)

    # Use the last (largest) set of trajectories
    x_all = np.array([xi for xi in x_trajs])  # (N, T)
    eq_std = np.sqrt(kBT / abs(k_force))
    x_eq = np.linspace(-4 * eq_std, 4 * eq_std, 200)
    rho_eq = np.exp(-0.5 * abs(k_force) * x_eq**2 / kBT)
    rho_eq /= np.trapz(rho_eq, x_eq)

    for i, t0_max in enumerate(t0_max_list):
        ax = axes2[i]
        x_at_t0 = x_all[:, t0_max]
        ax.hist(x_at_t0, bins=50, density=True, alpha=0.7,
                label=f"sampled at t={t0_max*dt:.2f}")
        ax.plot(x_eq, rho_eq, "k--", lw=2, label=r"$\rho_\mathrm{eq}(x)$")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"t0 = {t0_max*dt:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.savefig("position_distribution_at_t0.png", dpi=150)
    plt.show()
    print("\nDone. Saved compare_volterra_lsq.png")
