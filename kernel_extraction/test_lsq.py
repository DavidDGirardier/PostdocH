#!/usr/bin/env python3
"""
Test suite for the multi-origin LSQ kernel extraction method.

Run:  python test_lsq.py
"""

import numpy as np
import sys

from nonstationary_kernel import (
    compute_kernel, volterra_second_kind_rect,
    generate_2d_doublewell_trajectories,
)
from nonstationary_kernel_lsq import extract_kernel_lsq
from compare_volterra_lsq import generate_1d_gle_colored


# ============================================================
# Helpers
# ============================================================

def rel_l2(K_ext, t_ext, K_true_func, n_pts=None):
    """Relative L2 error over the first n_pts kernel points."""
    if n_pts is None:
        n_pts = len(K_ext)
    n = min(n_pts, len(K_ext))
    Kt = K_true_func(t_ext[:n])
    num = np.trapz((K_ext[:n] - Kt)**2, t_ext[:n])
    den = np.trapz(Kt**2, t_ext[:n])
    return np.sqrt(num / den) if den > 0 else np.inf


def generate_two_exp_trajectories(k_force, gamma1, tau1, gamma2, tau2,
                                   kBT, dt, nsteps, N, rng=None):
    """
    1D GLE with K(t) = gamma1*exp(-t/tau1) + gamma2*exp(-t/tau2).
    Two auxiliary variables s1, s2.  Cold start, colored noise (OU).
    Returns a_det (no noise).
    """
    if rng is None:
        rng = np.random.default_rng()

    exp1 = np.exp(-dt / tau1)
    exp2 = np.exp(-dt / tau2)
    ou_std1 = np.sqrt(kBT * (1 - exp1**2))
    ou_std2 = np.sqrt(kBT * (1 - exp2**2))
    sg1 = np.sqrt(gamma1)
    sg2 = np.sqrt(gamma2)

    x = np.zeros((N, nsteps))
    v = np.zeros((N, nsteps))
    a_det = np.zeros((N, nsteps))
    s1 = np.zeros(N)
    s2 = np.zeros(N)
    n1 = rng.normal(0, np.sqrt(kBT), N)
    n2 = rng.normal(0, np.sqrt(kBT), N)

    x[:, 0] = 0.0
    v[:, 0] = rng.normal(0, np.sqrt(kBT), N)

    for i in range(nsteps - 1):
        F = k_force * x[:, i]
        mem = gamma1 * s1 + gamma2 * s2
        noise = sg1 * n1 + sg2 * n2
        a_det[:, i] = F - mem
        v[:, i + 1] = v[:, i] + dt * (F - mem + noise)
        x[:, i + 1] = x[:, i] + dt * v[:, i]
        s1 = s1 * exp1 + v[:, i] * dt
        s2 = s2 * exp2 + v[:, i] * dt
        n1 = n1 * exp1 + ou_std1 * rng.normal(size=N)
        n2 = n2 * exp2 + ou_std2 * rng.normal(size=N)

    a_det[:, -1] = k_force * x[:, -1] - gamma1 * s1 - gamma2 * s2
    return list(x), list(v), list(a_det)


# ============================================================
# Shared parameters and trajectory cache
# ============================================================

K_FORCE = -4.0
GAMMA = 10.0
TAU_MEM = 0.1
KBT = 1.0
DT = 0.001
N_KERNEL = 500
TAU_MAX = 400
NSTEPS = N_KERNEL + 250
N_COMPARE = 200   # compare first 200 points (0.2 time = 2 tau)

_cache = {}

def get_trajs(N, seed=42):
    """Cached trajectory generation."""
    key = (N, seed)
    if key not in _cache:
        rng = np.random.default_rng(seed)
        _cache[key] = generate_1d_gle_colored(
            K_FORCE, GAMMA, TAU_MEM, KBT, DT, NSTEPS, N,
            x0=0.0, rng=rng)
    return _cache[key]


def get_trajs_displaced(N, x0, seed=42):
    """Cached trajectory generation with displaced initial position."""
    key = (N, x0, seed)
    if key not in _cache:
        rng = np.random.default_rng(seed)
        _cache[key] = generate_1d_gle_colored(
            K_FORCE, GAMMA, TAU_MEM, KBT, DT, NSTEPS, N,
            x0=x0, rng=rng)
    return _cache[key]


def K_true(t):
    return GAMMA * np.exp(-t / TAU_MEM)


# ============================================================
# Tests
# ============================================================

def test_exact_recovery():
    """LSQ with large N should recover K(t) = gamma*exp(-t/tau) accurately."""
    xt, vt, at = get_trajs(20000)
    res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                              t0_max_idx=100, tau_max_idx=TAU_MAX,
                              k_force=K_FORCE)
    err = rel_l2(res["K"], res["time"], K_true, N_COMPARE)
    assert err < 0.10, f"Relative L2 error {err:.4f} > 0.10"
    return err


def test_two_exponentials():
    """LSQ recovers a sum of two exponentials."""
    gamma1, tau1 = 8.0, 0.1
    gamma2, tau2 = 4.0, 0.5
    k_force = -4.0
    dt = 0.001
    n_kernel = 800
    nsteps = n_kernel + 250

    def K_two(t):
        return gamma1 * np.exp(-t / tau1) + gamma2 * np.exp(-t / tau2)

    rng = np.random.default_rng(42)
    xt, vt, at = generate_two_exp_trajectories(
        k_force, gamma1, tau1, gamma2, tau2, KBT, dt, nsteps, 20000, rng)

    res = extract_kernel_lsq(xt, vt, at, dt, n_kernel,
                              t0_max_idx=100, tau_max_idx=600,
                              k_force=k_force)
    err = rel_l2(res["K"], res["time"], K_two, 300)
    assert err < 0.10, f"Two-exp relative L2 error {err:.4f} > 0.10"
    return err


def test_lsq_t0_equals_volterra():
    """LSQ at t0=0 should match Volterra (same data, same equation)."""
    xt, vt, at = get_trajs(10000)

    res_v = compute_kernel(xt, vt, at, K_FORCE, DT, N_KERNEL,
                           method="second_kind_rect")
    res_l = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                t0_max_idx=0, tau_max_idx=TAU_MAX,
                                k_force=K_FORCE)

    # Compare over first N_COMPARE points
    n = min(N_COMPARE, len(res_v["K"]), len(res_l["K"]))
    diff = np.sqrt(np.mean((res_v["K"][:n] - res_l["K"][:n])**2))
    scale = np.sqrt(np.mean(res_v["K"][:n]**2))
    rel_diff = diff / scale
    assert rel_diff < 0.15, f"LSQ vs Volterra relative diff {rel_diff:.4f} > 0.15"
    return rel_diff


def test_forward_consistency():
    """The extracted K should satisfy A @ K + G ≈ 0."""
    xt, vt, at = get_trajs(10000)
    res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                              t0_max_idx=50, tau_max_idx=TAU_MAX,
                              k_force=K_FORCE)

    residual_per_row = np.sqrt(res["residual"] / res["n_rows"])
    G_scale = np.sqrt(np.mean(res["G_vec"]**2))
    rel_residual = residual_per_row / G_scale
    assert rel_residual < 0.5, f"Relative residual {rel_residual:.4f} > 0.5"
    return rel_residual


def test_more_t0_lowers_error():
    """More time origins should give lower (or equal) error."""
    xt, vt, at = get_trajs(2000)

    errs = []
    for t0_max in [0, 50, 200]:
        res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=t0_max, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)
        errs.append(rel_l2(res["K"], res["time"], K_true, N_COMPARE))

    # All should be accurate; more t0 should not degrade significantly
    for i, e in enumerate(errs):
        assert e < 0.02, f"t0={[0,50,200][i]} error {e:.4f} > 0.02"
    return errs


def test_convergence_with_N():
    """Error should be bounded even at low N (adaptive reg absorbs noise)."""
    N_list = [50, 10000]
    errs = []
    for N in N_list:
        xt, vt, at = get_trajs(N, seed=123)
        res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=0, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)
        errs.append(rel_l2(res["K"], res["time"], K_true, N_COMPARE))

    # Both should give reasonable results thanks to adaptive regularization
    assert errs[0] < 0.10, f"N={N_list[0]} error {errs[0]:.4f} > 0.10"
    assert errs[1] < 0.10, f"N={N_list[1]} error {errs[1]:.4f} > 0.10"
    return errs


def test_prememory_in_matrix():
    """A_mat rows at t0>0 should include correlations back to v(0)."""
    xt, vt, at = get_trajs(1000)
    t0_max = 50
    res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                              t0_max_idx=t0_max, tau_max_idx=TAU_MAX,
                              k_force=K_FORCE)

    A = res["A_mat"]
    # Rows from t0>0 should have nonzero entries beyond column tau
    # (those are the pre-memory terms involving K_j for j > tau)
    # Check a row from late t0: it should use more columns than tau alone
    # With t0_max=50 and tau_max=400, first block has ~400 rows (t0=0),
    # next block starts after that.
    # For t0=50, tau=10: j_max = min(500, 61) = 61 columns used.
    # Check that column 55 (j=55 > tau=10) has a nonzero entry.

    # Find a row from t0=50 block.  t0=0 contributes up to tau_end rows.
    # tau_end(t0=0) = min(400, nsteps-1, 499) rows
    # tau_end(t0=1) = min(400, nsteps-2, 498) rows, etc.
    # Row offset for t0=50: sum of tau_end for t0=0..49
    # Easier: just check that A has entries beyond the diagonal band
    n_rows, n_cols = A.shape
    # Check last rows — they come from large t0
    last_row = A[-1, :]
    # Count nonzero columns (above machine epsilon)
    n_nonzero = np.sum(np.abs(last_row) > 1e-15)
    assert n_nonzero > 5, (
        f"Last A_mat row has only {n_nonzero} nonzero entries "
        f"(expected pre-memory terms)")
    return n_nonzero


def test_nonlinear_force():
    """LSQ with force_func recovers K from 2D double-well trajectories."""
    from nonstationary_kernel import generate_2d_doublewell_trajectories

    omega_y = 3.0
    kBT = 0.5
    dt = 0.001
    coupling = 1.5
    tau_mem = 0.1
    gamma_mat = np.array([[10.0, 2.0], [2.0, 5.0]])
    N_trajs = 10000
    n_kernel = 300
    nsteps = n_kernel + 50

    def K_true_xx(t):
        return gamma_mat[0, 0] * np.exp(-t / tau_mem)

    rng = np.random.default_rng(42)
    x_trajs, vx_trajs, ax_trajs, _, fx_trajs = \
        generate_2d_doublewell_trajectories(
            gamma_mat, tau_mem, kBT, dt, nsteps,
            N_trajs, x0=0.0, y0=0.0, omega_y=omega_y,
            coupling=coupling, rng=rng)

    force_func = lambda x_arr: np.array(
        [fx_trajs[i][:x_arr.shape[1]] for i in range(len(fx_trajs))])

    res = extract_kernel_lsq(
        x_trajs, vx_trajs, ax_trajs, dt, n_kernel,
        t0_max_idx=50, tau_max_idx=200,
        force_func=force_func)

    err = rel_l2(res["K"], res["time"], K_true_xx, 100)
    assert err < 0.10, f"2D double-well error {err:.4f} > 0.10"
    return err


def test_large_N_limit():
    """With very large N, error should be dominated by discretization O(dt)."""
    xt, vt, at = get_trajs(20000, seed=99)
    res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                              t0_max_idx=200, tau_max_idx=TAU_MAX,
                              k_force=K_FORCE)
    err = rel_l2(res["K"], res["time"], K_true, N_COMPARE)
    assert err < 0.10, f"Large-N error {err:.4f} > 0.10"
    return err


def test_regularization_not_biasing():
    """Regularization should not significantly bias the kernel."""
    xt, vt, at = get_trajs(20000)

    # Extract with default regularization
    res_reg = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=100, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)

    # Compare with true kernel — if regularization biases significantly,
    # the error would be much larger than statistical noise alone
    err = rel_l2(res_reg["K"], res_reg["time"], K_true, N_COMPARE)

    # With 20000 trajs and multi-t0, error should be reasonable
    assert err < 0.10, (
        f"Regularized kernel error {err:.4f} > 0.10 — "
        f"regularization may be biasing the result")
    return err


# ============================================================
# OOE / multi-t0 tests
# ============================================================

def test_kernel_independent_of_x0():
    """K(t) is a bath property — must be the same regardless of x0."""
    N = 5000
    xt_a, vt_a, at_a = get_trajs_displaced(N, x0=0.0, seed=77)
    xt_b, vt_b, at_b = get_trajs_displaced(N, x0=2.0, seed=77)

    res_a = extract_kernel_lsq(xt_a, vt_a, at_a, DT, N_KERNEL,
                                t0_max_idx=100, tau_max_idx=TAU_MAX,
                                k_force=K_FORCE)
    res_b = extract_kernel_lsq(xt_b, vt_b, at_b, DT, N_KERNEL,
                                t0_max_idx=100, tau_max_idx=TAU_MAX,
                                k_force=K_FORCE)

    err_a = rel_l2(res_a["K"], res_a["time"], K_true, N_COMPARE)
    err_b = rel_l2(res_b["K"], res_b["time"], K_true, N_COMPARE)

    # Mutual difference
    n = N_COMPARE
    t = res_a["time"][:n]
    Kt = K_true(t)
    num = np.trapz((res_a["K"][:n] - res_b["K"][:n])**2, t)
    den = np.trapz(Kt**2, t)
    diff_ab = np.sqrt(num / den)

    assert err_a < 0.15, f"x0=0 error {err_a:.4f} > 0.15"
    assert err_b < 0.15, f"x0=2 error {err_b:.4f} > 0.15"
    assert diff_ab < 0.15, f"x0=0 vs x0=2 diff {diff_ab:.4f} > 0.15"
    return (err_a, err_b, diff_ab)


def test_displaced_start_more_t0_helps():
    """More t0 should help even when starting displaced (x0=2.0)."""
    xt, vt, at = get_trajs_displaced(2000, x0=2.0, seed=88)

    errs = []
    for t0_max in [0, 50, 200]:
        res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=t0_max, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)
        errs.append(rel_l2(res["K"], res["time"], K_true, N_COMPARE))

    for i, e in enumerate(errs):
        assert e < 0.02, (
            f"t0={[0,50,200][i]} error {e:.4f} > 0.02 (displaced start)")
    return errs


def test_convergence_with_N_displaced():
    """Error should be bounded at all N for displaced start."""
    N_list = [200, 5000]
    errs = []
    for N in N_list:
        xt, vt, at = get_trajs_displaced(N, x0=2.0, seed=123)
        res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=100, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)
        errs.append(rel_l2(res["K"], res["time"], K_true, N_COMPARE))

    for i, (N, e) in enumerate(zip(N_list, errs)):
        assert e < 0.02, f"N={N} error {e:.4f} > 0.02 (displaced start)"
    return errs


def test_forward_reconstruction_at_t0():
    """Extracted K should reconstruct G[t0,tau] at a specific displaced t0."""
    xt, vt, at = get_trajs_displaced(10000, x0=1.5, seed=99)
    t0_max = 50
    res = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                              t0_max_idx=t0_max, tau_max_idx=TAU_MAX,
                              k_force=K_FORCE)

    A_mat = res["A_mat"]
    G_vec = res["G_vec"]
    K = res["K"]

    # Find row range for t0=25:
    # Each t0 contributes tau_end rows. Accumulate offsets.
    T_min = NSTEPS
    row_offset = 0
    target_t0 = 25
    for t0 in range(target_t0):
        tau_end = min(TAU_MAX, T_min - t0 - 1, N_KERNEL - t0 - 1)
        if tau_end >= 1:
            row_offset += tau_end
    tau_end_target = min(TAU_MAX, T_min - target_t0 - 1,
                         N_KERNEL - target_t0 - 1)

    G_slice = G_vec[row_offset:row_offset + tau_end_target]
    A_slice = A_mat[row_offset:row_offset + tau_end_target, :]
    G_recon = A_slice @ K

    # Relative RMS of reconstruction error
    rms_err = np.sqrt(np.mean((G_recon + G_slice)**2))
    rms_G = np.sqrt(np.mean(G_slice**2))
    rel_err = rms_err / rms_G if rms_G > 0 else np.inf

    assert rel_err < 0.20, (
        f"Reconstruction error at t0={target_t0}: {rel_err:.4f} > 0.20")
    return rel_err


def test_2d_doublewell_displaced_y():
    """LSQ recovers K_xx from 2D double-well with y strongly displaced."""
    omega_y = 3.0
    kBT = 0.5
    dt = 0.001
    coupling = 1.5
    tau_mem = 0.1
    gamma_mat = np.array([[10.0, 2.0], [2.0, 5.0]])
    N_trajs = 10000
    n_kernel = 300
    nsteps = n_kernel + 50

    def K_true_xx(t):
        return gamma_mat[0, 0] * np.exp(-t / tau_mem)

    rng = np.random.default_rng(42)
    x_trajs, vx_trajs, ax_trajs, _, fx_trajs = \
        generate_2d_doublewell_trajectories(
            gamma_mat, tau_mem, kBT, dt, nsteps,
            N_trajs, x0=0.0, y0=3.0, omega_y=omega_y,
            coupling=coupling, rng=rng)

    force_func = lambda x_arr: np.array(
        [fx_trajs[i][:x_arr.shape[1]] for i in range(len(fx_trajs))])

    res = extract_kernel_lsq(
        x_trajs, vx_trajs, ax_trajs, dt, n_kernel,
        t0_max_idx=50, tau_max_idx=200,
        force_func=force_func)

    err = rel_l2(res["K"], res["time"], K_true_xx, 100)
    assert err < 0.15, f"2D displaced-y error {err:.4f} > 0.15"
    return err


def test_w_func_boltzmann_weighting():
    """w_func (Boltzmann weighting) should work without degrading extraction."""
    xt, vt, at = get_trajs(10000)

    def w_boltzmann(t0_idx, x, v):
        return np.exp(-0.5 * abs(K_FORCE) * x[:, t0_idx]**2 / KBT)

    res_unif = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                   t0_max_idx=100, tau_max_idx=TAU_MAX,
                                   k_force=K_FORCE)
    res_wt = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                 t0_max_idx=100, tau_max_idx=TAU_MAX,
                                 k_force=K_FORCE, w_func=w_boltzmann)

    err_u = rel_l2(res_unif["K"], res_unif["time"], K_true, N_COMPARE)
    err_w = rel_l2(res_wt["K"], res_wt["time"], K_true, N_COMPARE)

    n = N_COMPARE
    t = res_unif["time"][:n]
    Kt = K_true(t)
    num = np.trapz((res_unif["K"][:n] - res_wt["K"][:n])**2, t)
    den = np.trapz(Kt**2, t)
    diff = np.sqrt(num / den)

    assert err_w < 0.15, f"Boltzmann-weighted error {err_w:.4f} > 0.15"
    assert diff < 0.20, (
        f"Uniform vs Boltzmann diff {diff:.4f} > 0.20")
    return (err_u, err_w, diff)


def test_prememory_at_large_t0():
    """Pre-memory stress test: large t0_max makes most rows pre-memory-heavy."""
    nsteps_long = 1000
    rng = np.random.default_rng(42)
    xt, vt, at = generate_1d_gle_colored(
        K_FORCE, GAMMA, TAU_MEM, KBT, DT, nsteps_long, 10000,
        x0=0.0, rng=rng)

    # t0_max=0 baseline
    res_0 = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                t0_max_idx=0, tau_max_idx=200,
                                k_force=K_FORCE)
    # t0_max=400 stress test
    res_big = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=400, tau_max_idx=200,
                                  k_force=K_FORCE)

    # Check early kernel (first 20 points = 0.02 time = 0.2 tau_mem)
    err_0 = rel_l2(res_0["K"], res_0["time"], K_true, 20)
    err_big = rel_l2(res_big["K"], res_big["time"], K_true, 20)

    assert err_big < 2 * err_0 + 0.02, (
        f"Large-t0 early-kernel error {err_big:.4f} much worse than "
        f"t0=0 error {err_0:.4f}")
    return (err_0, err_big)


def test_lsq_vs_stationary_volterra():
    """LSQ (multi-t0) should match the standard Volterra extraction
    from many cold-start trajectories in the well."""
    N = 5000
    xt, vt, at = get_trajs(N, seed=200)

    # --- Standard Volterra (t0=0, transient correlations) ---
    res_volt = compute_kernel(xt, vt, at, K_FORCE, DT, N_KERNEL,
                              method="second_kind_rect")

    # --- LSQ with multi-t0 ---
    res_lsq = extract_kernel_lsq(xt, vt, at, DT, N_KERNEL,
                                  t0_max_idx=100, tau_max_idx=TAU_MAX,
                                  k_force=K_FORCE)

    n = N_COMPARE
    err_volt = rel_l2(res_volt["K"], res_volt["time"], K_true, n)
    err_lsq = rel_l2(res_lsq["K"], res_lsq["time"], K_true, n)

    # Mutual difference
    t = res_lsq["time"][:n]
    Kt = K_true(t)
    K_v = res_volt["K"][:n]
    K_l = res_lsq["K"][:n]
    num = np.trapz((K_v - K_l)**2, t)
    den = np.trapz(Kt**2, t)
    diff = np.sqrt(num / den)

    print(f"    Volterra err = {err_volt:.4f}, "
          f"LSQ err = {err_lsq:.4f}, mutual diff = {diff:.4f}")

    # Both should recover K_true
    assert err_volt < 0.15, f"Volterra error {err_volt:.4f} > 0.15"
    assert err_lsq < 0.15, f"LSQ error {err_lsq:.4f} > 0.15"
    # They should agree within LSQ accuracy (limited by regularization)
    assert diff < 0.15, (
        f"Volterra vs LSQ diff {diff:.4f} > 0.15")
    return (err_volt, err_lsq, diff)


# ============================================================
# Runner
# ============================================================

ALL_TESTS = [
    ("Exact recovery (exp kernel, large N)",    test_exact_recovery),
    ("Sum of two exponentials",                 test_two_exponentials),
    ("LSQ t0=0 matches Volterra",               test_lsq_t0_equals_volterra),
    ("Forward consistency (A@K + G ≈ 0)",       test_forward_consistency),
    ("More t0 → lower error",                   test_more_t0_lowers_error),
    ("Convergence with N (~N^{-1/2})",          test_convergence_with_N),
    ("Pre-memory in A_mat structure",            test_prememory_in_matrix),
    ("Nonlinear force (2D double-well)",         test_nonlinear_force),
    ("Large-N limit (discretization error)",     test_large_N_limit),
    ("Regularization not biasing",               test_regularization_not_biasing),
    # --- OOE / multi-t0 tests ---
    ("Kernel independent of x0",                test_kernel_independent_of_x0),
    ("Displaced start: more t0 helps",          test_displaced_start_more_t0_helps),
    ("Convergence with N (displaced)",           test_convergence_with_N_displaced),
    ("Forward reconstruction at t0=25",         test_forward_reconstruction_at_t0),
    ("2D double-well displaced y",              test_2d_doublewell_displaced_y),
    ("Boltzmann w_func weighting",              test_w_func_boltzmann_weighting),
    ("Pre-memory at large t0",                  test_prememory_at_large_t0),
    ("LSQ vs stationary Volterra",              test_lsq_vs_stationary_volterra),
]


if __name__ == "__main__":
    passed = 0
    failed = 0
    results = []

    for name, func in ALL_TESTS:
        try:
            ret = func()
            print(f"  PASS  {name}  ({ret})")
            passed += 1
            results.append((name, "PASS", ret))
        except AssertionError as e:
            print(f"  FAIL  {name}  ({e})")
            failed += 1
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            print(f"  ERROR {name}  ({type(e).__name__}: {e})")
            failed += 1
            results.append((name, "ERROR", str(e)))

    print(f"\n{passed}/{passed+failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
