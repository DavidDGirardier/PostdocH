#!/usr/bin/env python3
"""
Extended benchmark: RF vs Prony vs Free LSQ across different kernel types.

Cases:
  A. Single exponential (reference, favors Prony 1-exp)
  B. Bi-exponential (Prony 1-exp fails, 2-exp should work)
  C. Oscillatory kernel (fundamentally non-Prony)
  D. Stretched exponential (no finite Prony sum)

All use V(x) = x⁴ - 2x², ω_b = 2, kBT = 1, cold start at barrier.
"""

import numpy as np
import scipy.optimize
import scipy.integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nonstationary_kernel_lsq import extract_kernel_lsq
from benchmark_prony_nlsq import extract_kernel_prony
from benchmark_kernel_vs_reactiveflux import (
    compute_reactive_flux, kappa_plateau,
    kappa_from_free_kernel, subsample_fd)


# ── GLE integrators ─────────────────────────────────────────────────────

def generate_gle_multiexp(amps, taus, kBT, dt, nsteps, N_trajs,
                          x0=0.0, rng=None, force='doublewell'):
    """GLE with multi-exponential kernel: K(t) = Σ aᵢ exp(-t/τᵢ).

    Uses one auxiliary variable per exponential (efficient).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_exp = len(amps)
    exp_decays = np.exp(-dt / np.array(taus))
    ou_stds = np.sqrt(kBT * (1 - exp_decays**2))

    x = np.zeros((N_trajs, nsteps))
    v = np.zeros((N_trajs, nsteps))
    s = np.zeros((n_exp, N_trajs))  # memory auxiliaries
    n_ou = rng.normal(0, np.sqrt(kBT), (n_exp, N_trajs))

    x[:, 0] = x0
    v[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

    for i in range(nsteps - 1):
        if force == 'doublewell':
            F = 4 * x[:, i] - 4 * x[:, i]**3
        else:
            F = force(x[:, i])

        mem = sum(amps[k] * s[k] for k in range(n_exp))
        noise = sum(np.sqrt(amps[k]) * n_ou[k] for k in range(n_exp))

        v[:, i + 1] = v[:, i] + dt * (F - mem + noise)
        x[:, i + 1] = x[:, i] + dt * v[:, i]

        for k in range(n_exp):
            s[k] = s[k] * exp_decays[k] + v[:, i] * dt
            n_ou[k] = n_ou[k] * exp_decays[k] + ou_stds[k] * rng.normal(size=N_trajs)

    return x, v


def generate_gle_oscillatory(gamma, tau, omega_osc, kBT, dt, nsteps,
                              N_trajs, x0=0.0, rng=None):
    """GLE with oscillatory kernel: K(t) = γ exp(-t/τ) cos(ωt).

    Uses two auxiliary variables for sin/cos decomposition.
    Noise: colored noise with same power spectrum (FDT).
    """
    if rng is None:
        rng = np.random.default_rng()

    exp_dec = np.exp(-dt / tau)
    cos_w = np.cos(omega_osc * dt)
    sin_w = np.sin(omega_osc * dt)
    ou_std = np.sqrt(kBT * (1 - exp_dec**2))

    x = np.zeros((N_trajs, nsteps))
    v = np.zeros((N_trajs, nsteps))
    s_r = np.zeros(N_trajs)  # real part of memory aux
    s_i = np.zeros(N_trajs)  # imaginary part
    n_r = rng.normal(0, np.sqrt(kBT), N_trajs)
    n_i = rng.normal(0, np.sqrt(kBT), N_trajs)

    x[:, 0] = x0
    v[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

    for i in range(nsteps - 1):
        F = 4 * x[:, i] - 4 * x[:, i]**3
        mem = gamma * s_r
        noise = np.sqrt(gamma) * n_r

        v[:, i + 1] = v[:, i] + dt * (F - mem + noise)
        x[:, i + 1] = x[:, i] + dt * v[:, i]

        # Rotate and decay auxiliary variables
        sr_new = exp_dec * (s_r * cos_w + s_i * sin_w) + v[:, i] * dt
        si_new = exp_dec * (-s_r * sin_w + s_i * cos_w)
        s_r, s_i = sr_new, si_new

        nr_new = exp_dec * (n_r * cos_w + n_i * sin_w) + ou_std * rng.normal(size=N_trajs)
        ni_new = exp_dec * (-n_r * sin_w + n_i * cos_w) + ou_std * rng.normal(size=N_trajs)
        n_r, n_i = nr_new, ni_new

    return x, v


def generate_gle_arbitrary(K_array, dt, kBT, nsteps, N_trajs,
                           x0=0.0, rng=None, force='doublewell'):
    """GLE with arbitrary kernel via direct convolution.

    K_array: kernel values at t=0, dt, 2dt, ...
    force: 'doublewell' (4x-4x³) or callable F(x_array) -> array.
    Colored noise generated via spectral filter to satisfy FDT.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_k = len(K_array)

    x = np.zeros((N_trajs, nsteps))
    v = np.zeros((N_trajs, nsteps))

    x[:, 0] = x0
    v[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

    # Precompute noise with correct correlation structure
    # For FDT: <η(t)η(t')> = kBT K(|t-t'|)
    # Generate via Cholesky of the correlation matrix (expensive but correct)
    # For large nsteps, use circulant embedding
    n_noise = min(nsteps, 2000)
    K_corr = kBT * K_array[:n_noise]
    # Use approximate method: filter white noise through sqrt(spectrum)
    nfft = 2 ** int(np.ceil(np.log2(2 * n_noise)))
    K_padded = np.zeros(nfft)
    K_padded[:n_noise] = K_corr
    K_padded[nfft-n_noise+1:] = K_corr[1:][::-1]  # symmetric
    S = np.abs(np.fft.rfft(K_padded))
    S_sqrt = np.sqrt(np.maximum(S, 0) * dt)

    white = rng.normal(size=(N_trajs, nfft))
    white_fft = np.fft.rfft(white, axis=1)
    noise = np.fft.irfft(S_sqrt[None, :] * white_fft, n=nfft, axis=1)[:, :nsteps]

    # Store velocity history for convolution
    for i in range(nsteps - 1):
        if force == 'doublewell':
            F = 4 * x[:, i] - 4 * x[:, i]**3
        else:
            F = force(x[:, i])

        # Memory integral: ∫₀ᵗ K(s) v(t-s) ds ≈ dt Σⱼ K(j) v(i-j)
        if i == 0:
            mem = 0.0
        else:
            j_max = min(i, n_k)
            # v at times i, i-1, ..., i-j_max+1
            idx = np.arange(i, i - j_max, -1)
            v_hist = v[:, idx]  # shape (N, j_max)
            mem = dt * v_hist @ K_array[:j_max]

        v[:, i + 1] = v[:, i] + dt * (F - mem + noise[:, i])
        x[:, i + 1] = x[:, i] + dt * v[:, i]

    return x, v


# ── Grote-Hynes with general kernel ─────────────────────────────────────

def grote_hynes_general(omega_b, K_hat_func):
    """Solve GH equation with general K̂(s).

    λ = ω_b² / (λ + K̂(λ)),  returns κ = λ/ω_b.
    """
    def eq(lam):
        return lam - omega_b**2 / (lam + K_hat_func(lam))
    try:
        lam_r = scipy.optimize.brentq(eq, 1e-10, omega_b - 1e-10)
        return lam_r / omega_b
    except (ValueError, RuntimeError):
        return np.nan


# ── Case definitions ────────────────────────────────────────────────────

def define_cases():
    kBT = 1.0
    omega_b = 2.0
    dt_base = 0.001

    cases = {}

    # Case A: Single exponential (reference)
    gamma_A, tau_A = 20.0, 0.1
    cases['A: 1-exp'] = {
        'K_func': lambda t: gamma_A * np.exp(-t / tau_A),
        'K_hat': lambda s: gamma_A * tau_A / (1 + s * tau_A),
        'gen': lambda N, rng: generate_gle_multiexp(
            [gamma_A], [tau_A], kBT, dt_base, 2001, N, x0=0.0, rng=rng),
        'dt': dt_base,
        'gamma_total': gamma_A * tau_A,  # friction = ∫K dt
    }

    # Case B: Bi-exponential (fast + slow)
    g1, t1, g2, t2 = 15.0, 0.02, 5.0, 0.3
    cases['B: 2-exp'] = {
        'K_func': lambda t: g1 * np.exp(-t / t1) + g2 * np.exp(-t / t2),
        'K_hat': lambda s: g1*t1/(1+s*t1) + g2*t2/(1+s*t2),
        'gen': lambda N, rng: generate_gle_multiexp(
            [g1, g2], [t1, t2], kBT, dt_base, 2001, N, x0=0.0, rng=rng),
        'dt': dt_base,
        'gamma_total': g1*t1 + g2*t2,
    }

    # Case C: Oscillatory kernel
    gamma_C, tau_C, omega_C = 20.0, 0.15, 40.0
    cases['C: oscillatory'] = {
        'K_func': lambda t: gamma_C * np.exp(-t / tau_C) * np.cos(omega_C * t),
        'K_hat': lambda s: gamma_C * tau_C * (1 + s*tau_C) / ((1+s*tau_C)**2 + (omega_C*tau_C)**2),
        'gen': lambda N, rng: generate_gle_oscillatory(
            gamma_C, tau_C, omega_C, kBT, dt_base, 2001, N, x0=0.0, rng=rng),
        'dt': dt_base,
        'gamma_total': gamma_C * tau_C / (1 + (omega_C*tau_C)**2),
    }

    # Case D: Stretched exponential
    gamma_D, tau_D, beta_D = 15.0, 0.08, 0.5
    K_D_array = gamma_D * np.exp(-(np.arange(2001) * dt_base / tau_D)**beta_D)
    cases['D: stretched'] = {
        'K_func': lambda t: gamma_D * np.exp(-(t / tau_D)**beta_D),
        'K_hat': None,  # compute numerically
        'gen': lambda N, rng: generate_gle_arbitrary(
            K_D_array, dt_base, kBT, 2001, N, x0=0.0, rng=rng),
        'dt': dt_base,
        'gamma_total': None,
        '_K_array': K_D_array,
    }

    # Compute κ_GH for each case
    for name, case in cases.items():
        if case['K_hat'] is not None:
            case['kappa_GH'] = grote_hynes_general(omega_b, case['K_hat'])
        else:
            # Numerical Laplace for stretched exp
            t_arr = np.arange(2001) * dt_base
            K_vals = case['K_func'](t_arr)
            def K_hat_num(s, t=t_arr, K=K_vals):
                return np.trapz(K * np.exp(-s * t), t)
            case['kappa_GH'] = grote_hynes_general(omega_b, K_hat_num)
            case['K_hat'] = K_hat_num
        print(f"{name}: κ_GH = {case['kappa_GH']:.4f}")

    return cases


# ── Main benchmark ──────────────────────────────────────────────────────

if __name__ == "__main__":
    omega_b = 2.0
    kBT = 1.0

    cases = define_cases()
    N_list = [50, 100, 200, 500, 1000, 2000, 5000]
    n_seeds = 8
    stride = 5
    t0_max = 5

    force_func_dw = lambda x_arr: 4 * x_arr - 4 * x_arr**3

    # Storage
    all_results = {}

    for case_name, case in cases.items():
        print(f"\n{'#'*70}")
        print(f"# {case_name}")
        print(f"{'#'*70}", flush=True)

        dt = case['dt']
        kappa_exact = case['kappa_GH']

        res = {N: {'rf': [], 'prony1': [], 'prony2': [], 'free': []}
               for N in N_list}

        for N_trajs in N_list:
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                x, v = case['gen'](N_trajs, rng)

                # Reactive flux
                kappa_t = compute_reactive_flux(x, v, dt)
                kap_rf = kappa_plateau(kappa_t)

                # Subsample for kernel extraction
                x_sub, v_sub, a_sub, dt_eff = subsample_fd(x, dt, stride)
                n_kernel_eff = max(10, int(0.5 / dt_eff))
                tau_max_eff = max(5, int(0.2 / dt_eff))

                # Prony 1-exp
                try:
                    res_p1 = extract_kernel_prony(
                        x_sub, v_sub, a_sub, dt_eff,
                        t0_max_idx=t0_max, tau_max_idx=tau_max_eff,
                        force_func=force_func_dw, n_exp=1,
                        n_kernel=n_kernel_eff)
                    kap_p1 = kappa_from_free_kernel(res_p1['K'], dt_eff, omega_b)
                except Exception:
                    kap_p1 = np.nan

                # Prony 2-exp
                try:
                    res_p2 = extract_kernel_prony(
                        x_sub, v_sub, a_sub, dt_eff,
                        t0_max_idx=t0_max, tau_max_idx=tau_max_eff,
                        force_func=force_func_dw, n_exp=2,
                        n_kernel=n_kernel_eff)
                    kap_p2 = kappa_from_free_kernel(res_p2['K'], dt_eff, omega_b)
                except Exception:
                    kap_p2 = np.nan

                # Free LSQ
                try:
                    res_f = extract_kernel_lsq(
                        x_sub, v_sub, a_sub, dt_eff,
                        n_kernel=n_kernel_eff, t0_max_idx=t0_max,
                        tau_max_idx=tau_max_eff,
                        force_func=force_func_dw)
                    kap_f = kappa_from_free_kernel(res_f['K'], dt_eff, omega_b)
                except Exception:
                    kap_f = np.nan

                res[N_trajs]['rf'].append(kap_rf)
                res[N_trajs]['prony1'].append(kap_p1)
                res[N_trajs]['prony2'].append(kap_p2)
                res[N_trajs]['free'].append(kap_f)

            # Summary
            for method in ['rf', 'prony1', 'prony2', 'free']:
                vals = np.array(res[N_trajs][method])
                m = np.nanmean(vals)
                err = abs(m - kappa_exact) / kappa_exact if not np.isnan(m) else np.nan
                res[N_trajs][f'{method}_err'] = err

            print(f"  N={N_trajs:5d}: "
                  f"RF={res[N_trajs]['rf_err']:.0%}, "
                  f"P1={res[N_trajs]['prony1_err']:.0%}, "
                  f"P2={res[N_trajs]['prony2_err']:.0%}, "
                  f"Free={res[N_trajs]['free_err']:.0%}",
                  flush=True)

        all_results[case_name] = res

    # ════════════════════════════════════════════════════════════════════
    # Figure
    # ════════════════════════════════════════════════════════════════════

    n_cases = len(cases)
    fig, axes = plt.subplots(2, n_cases, figsize=(5 * n_cases, 9),
                             constrained_layout=True)

    for ci, (case_name, case) in enumerate(cases.items()):
        res = all_results[case_name]
        kappa_exact = case['kappa_GH']

        # Top: kernel shape
        ax = axes[0, ci]
        t_arr = np.linspace(0, 0.5, 500)
        K_true = case['K_func'](t_arr)
        ax.plot(t_arr, K_true, 'k-', lw=2, label='true K(t)')
        ax.set_xlabel('time')
        ax.set_ylabel('K(t)')
        ax.set_title(f'{case_name}\n(κ_GH={kappa_exact:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)

        # Bottom: convergence
        ax = axes[1, ci]
        methods = [
            ('rf', 'RF', 'o-', 'C0'),
            ('prony1', 'Prony 1exp', 's--', 'C1'),
            ('prony2', 'Prony 2exp', 'D-.', 'C2'),
            ('free', 'Free LSQ', '^:', 'C3'),
        ]
        for method, label, ls, color in methods:
            errs = []
            for N in N_list:
                vals = np.array(res[N][method])
                err = np.nanmean(np.abs(vals - kappa_exact) / kappa_exact)
                errs.append(err * 100)
            ax.plot(N_list, errs, ls, color=color, lw=1.5, ms=4, label=label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N trajectories')
        ax.set_ylabel('Error on κ (%)')
        ax.axhline(10, color='gray', ls=':', lw=1, alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1, 200)

    plt.savefig('benchmark_extended_cases.png', dpi=150, bbox_inches='tight')
    print("\nSaved benchmark_extended_cases.png", flush=True)
    plt.close()
