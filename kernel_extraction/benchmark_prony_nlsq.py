#!/usr/bin/env python3
"""
Nonlinear LSQ with Prony-constrained kernel: K(t) = Σ aᵢ exp(-t/τᵢ).

Reuses A_mat from the free LSQ to make residual evaluation fast.
Tests different t0_max and n_exp combinations.
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nonstationary_kernel_lsq import extract_kernel_lsq
from benchmark_fd_acceleration import generate_1d_gle_all_accels


def extract_kernel_prony(x_trajs, v_trajs, a_trajs, dt,
                         t0_max_idx, tau_max_idx, n_kernel=500,
                         k_force=None, force_func=None,
                         n_exp=2, p0=None):
    """Extract Prony-parameterized kernel via nonlinear least squares."""
    res_free = extract_kernel_lsq(
        x_trajs, v_trajs, a_trajs, dt,
        n_kernel=n_kernel, t0_max_idx=t0_max_idx,
        tau_max_idx=tau_max_idx,
        k_force=k_force, force_func=force_func, reg=1e-30)
    A_mat = res_free['A_mat']
    b_vec = -res_free['G_vec']

    j_arr = np.arange(n_kernel) * dt

    def residuals(params):
        K = np.zeros(n_kernel)
        for k in range(n_exp):
            K += params[2*k] * np.exp(-j_arr / np.exp(params[2*k+1]))
        return A_mat @ K - b_vec

    if p0 is None:
        if n_exp == 1:
            p0 = [5.0, np.log(0.05)]
        elif n_exp == 2:
            p0 = [5.0, np.log(0.03), 5.0, np.log(0.15)]
        elif n_exp == 3:
            p0 = [3.0, np.log(0.02), 3.0, np.log(0.08), 3.0, np.log(0.2)]

    result = scipy.optimize.least_squares(
        residuals, p0, method='trf',
        bounds=([0, -10] * n_exp, [1000, 5] * n_exp),
        max_nfev=5000, verbose=0)

    amps = result.x[0::2]
    taus = np.exp(result.x[1::2])
    order = np.argsort(taus)
    amps, taus = amps[order], taus[order]

    K = np.zeros(n_kernel)
    for k in range(n_exp):
        K += amps[k] * np.exp(-j_arr / taus[k])

    return {
        'amps': amps, 'taus': taus,
        'K': K, 'time': np.arange(n_kernel) * dt,
        'cost': result.cost, 'success': result.success,
    }


if __name__ == "__main__":
    k_force = -4.0
    gamma_val = 10.0
    tau_mem = 0.1
    kBT = 1.0
    dt = 0.001

    def K_true(t):
        return gamma_val * np.exp(-t / tau_mem)

    n_kernel = 500
    tau_max_lsq = 400
    nsteps = n_kernel + 100 + 100
    n_plot = 400

    t_ref = np.arange(n_kernel) * dt
    K_ref = K_true(t_ref)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)

    N_list = [100, 1000, 10000]
    t0_list = [0, 5, 10, 50]

    # ── Figure: Prony 1exp from a_total, varying t0 and N ──
    fig, axes = plt.subplots(2, len(N_list),
                             figsize=(5.5 * len(N_list), 8),
                             constrained_layout=True)

    for col, N_trajs in enumerate(N_list):
        print(f"\n{'='*60}")
        print(f"N = {N_trajs}")
        print(f"{'='*60}", flush=True)

        rng = np.random.default_rng(42)
        x_trajs, v_trajs, a_det, a_total, a_fd = \
            generate_1d_gle_all_accels(
                k_force, gamma_val, tau_mem, kBT, dt, nsteps,
                N_trajs, x0=0.0, rng=rng)

        ax_k = axes[0, col]
        ax_int = axes[1, col]
        ax_k.plot(t_ref[:n_plot], K_ref[:n_plot], 'k--', lw=2.5,
                  label='true (10·exp(−t/0.1))', zorder=10)
        ax_int.plot(t_ref[:n_plot], intK_ref[:n_plot], 'k--', lw=2.5,
                    label='true', zorder=10)

        # a_det reference: Prony 1exp t0≤50
        res_det = extract_kernel_prony(
            x_trajs, v_trajs, a_det, dt,
            t0_max_idx=50, tau_max_idx=tau_max_lsq,
            k_force=k_force, n_exp=1, n_kernel=n_kernel)
        err_det = np.sqrt(np.mean(
            (res_det['K'][:n_plot] - K_ref[:n_plot])**2)) / np.mean(K_ref[:n_plot])
        ax_k.plot(res_det['time'][:n_plot], res_det['K'][:n_plot],
                  lw=1, color='gray', alpha=0.5,
                  label=f'$a_{{det}}$ Prony ref ({err_det:.1%})')
        print(f"  a_det Prony ref: amp={res_det['amps'][0]:.2f}, "
              f"tau={res_det['taus'][0]:.4f}, RMSE={err_det:.1%}", flush=True)

        # Free LSQ a_total baselines
        for t0m, ls_free in [(0, ':'), (10, '--')]:
            res_free = extract_kernel_lsq(
                x_trajs, v_trajs, a_total, dt,
                n_kernel=n_kernel, t0_max_idx=t0m,
                tau_max_idx=tau_max_lsq, k_force=k_force)
            n = min(len(res_free['K']), n_plot)
            err_free = np.sqrt(np.mean(
                (res_free['K'][:n] - K_ref[:n])**2)) / np.mean(K_ref[:n])
            ax_k.plot(res_free['time'][:n], res_free['K'][:n],
                      lw=0.8, color='C1', ls=ls_free, alpha=0.4,
                      label=f'free t0≤{t0m} ({err_free:.0%})')
            intK = scipy.integrate.cumulative_trapezoid(
                res_free['K'][:n], res_free['time'][:n], initial=0)
            ax_int.plot(res_free['time'][:n], intK,
                        lw=0.8, color='C1', ls=ls_free, alpha=0.4,
                        label=f'free t0≤{t0m}')

        # Prony 1exp from a_total: sweep t0_max
        colors = ['C0', 'C2', 'C3', 'C4']
        for i, t0m in enumerate(t0_list):
            res_p = extract_kernel_prony(
                x_trajs, v_trajs, a_total, dt,
                t0_max_idx=t0m, tau_max_idx=tau_max_lsq,
                k_force=k_force, n_exp=1, n_kernel=n_kernel)
            err_p = np.sqrt(np.mean(
                (res_p['K'][:n_plot] - K_ref[:n_plot])**2)) / np.mean(K_ref[:n_plot])
            lab = (f"Prony t0≤{t0m}: "
                   f"a={res_p['amps'][0]:.1f}, τ={res_p['taus'][0]:.3f} "
                   f"({err_p:.1%})")
            ax_k.plot(res_p['time'][:n_plot], res_p['K'][:n_plot],
                      lw=1.8, color=colors[i], alpha=0.9, label=lab)
            intK = scipy.integrate.cumulative_trapezoid(
                res_p['K'][:n_plot], res_p['time'][:n_plot], initial=0)
            ax_int.plot(res_p['time'][:n_plot], intK,
                        lw=1.8, color=colors[i], alpha=0.9,
                        label=f'Prony t0≤{t0m}')
            print(f"  Prony 1exp a_total t0≤{t0m}: "
                  f"amp={res_p['amps'][0]:.2f}, tau={res_p['taus'][0]:.4f}, "
                  f"RMSE={err_p:.1%}", flush=True)

        ax_k.set_title(f'N = {N_trajs}')
        ax_k.set_ylabel(r'$K(t)$')
        ax_k.set_xlabel('time')
        ax_k.legend(fontsize=4.5, loc='upper right')
        ax_k.grid(True, alpha=0.3)
        ax_k.set_xlim(0, 0.4)

        ax_int.set_ylabel(r'$\int_0^t K$')
        ax_int.set_xlabel('time')
        ax_int.legend(fontsize=5.5)
        ax_int.grid(True, alpha=0.3)
        ax_int.set_xlim(0, 0.4)

    fig.suptitle(r'Prony 1-exp NLSQ from $a_\mathrm{total}$ — varying $t_0^\max$ and $N$',
                 fontsize=13, fontweight='bold')
    plt.savefig('benchmark_prony_nlsq.png', dpi=150, bbox_inches='tight')
    print("\nSaved benchmark_prony_nlsq.png", flush=True)
    plt.close()
