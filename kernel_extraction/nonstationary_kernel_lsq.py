#!/usr/bin/env python3
"""
Non-stationary memory kernel extraction via least-squares.

Uses multiple time origins t0 and lags tau to build an overdetermined
linear system from the Volterra equation, then solves for K by
weighted least squares.

For each (t0, tau):
  G[t0,tau] = -dt * sum_j K_j * A[t0,tau,j]

where:
  G[t0,tau] = <w(t0) v(t0) [a(t0+tau) - f_pmf(x(t0+tau))]>
  A[t0,tau,j] = <w(t0) v(t0) v(t0+tau - j*dt)>
  j = 0 ... (t0+tau)/dt
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def _smoothness_matrix(n):
    """First-difference matrix D (n-1 x n): (D @ K)_i = K_{i+1} - K_i."""
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D


def _lcurve_smooth_reg(A_mat, b_vec, DTD, n_grid=80):
    """L-curve corner detection for smoothness-penalized Tikhonov.

    Sweeps lambda on a log grid, computing (log||AK-b||, log||DK||)
    at each point, then returns the lambda at maximum curvature.
    """
    ATA = A_mat.T @ A_mat
    ATb = A_mat.T @ b_vec

    lams = np.logspace(-8, 3, n_grid)
    log_res = np.empty(n_grid)
    log_pen = np.empty(n_grid)

    for i, lam in enumerate(lams):
        K = np.linalg.solve(ATA + lam * DTD, ATb)
        log_res[i] = np.log10(max(np.sum((A_mat @ K - b_vec) ** 2), 1e-30))
        log_pen[i] = np.log10(max(K @ DTD @ K, 1e-30))

    # Curvature of (log_res, log_pen) vs log(lambda)
    log_lams = np.log10(lams)
    dx = np.gradient(log_res, log_lams)
    dy = np.gradient(log_pen, log_lams)
    d2x = np.gradient(dx, log_lams)
    d2y = np.gradient(dy, log_lams)
    kappa = (dx * d2y - d2x * dy) / (dx ** 2 + dy ** 2) ** 1.5

    margin = max(3, n_grid // 10)
    best_idx = margin + np.argmax(kappa[margin:-margin])

    return lams[best_idx]


def extract_kernel_lsq(x_trajs, v_trajs, a_trajs, dt, n_kernel,
                       t0_max_idx, tau_max_idx, force_func=None,
                       k_force=None, w_func=None, W_func=None,
                       reg=None):
    """
    Extract the memory kernel by least-squares over multiple time
    origins and lags.

    Parameters
    ----------
    x_trajs, v_trajs, a_trajs : list of 1D arrays (N_trajs,)
        Position, velocity, deterministic acceleration trajectories.
    dt : float
    n_kernel : int
        Number of kernel points to extract (K_0 ... K_{n_kernel-1}).
    t0_max_idx : int
        Maximum time origin index (t0 = 0, dt, ..., t0_max_idx*dt).
    tau_max_idx : int
        Maximum lag index.
    force_func : callable or None
        F(x) -> mean force. If None, uses k_force * x.
    k_force : float or None
        Linear force coefficient (used if force_func is None).
    w_func : callable or None
        w(t0_idx, x_trajs, v_trajs) -> weight array (N_trajs,).
        If None, uniform weights.
    W_func : callable or None
        W(t0_idx, tau_idx) -> scalar weight for the least-squares row.
        If None, uniform.
    reg : float, "gcv", or None
        Tikhonov regularization parameter.
        - float: use this value directly as lambda.
        - "gcv": select automatically via Generalized Cross-Validation.
        - None (default): heuristic 1e-5 * max(diag(A^T A)).

    Returns
    -------
    result : dict
        "K" : kernel array (n_kernel,)
        "time" : time array for kernel
        "residual" : least-squares residual
        "reg" : regularization parameter used
        "n_rows", "A_mat", "G_vec" : system details
    """
    N = len(x_trajs)
    T_min = min(len(x) for x in x_trajs)

    # Stack trajectories
    x = np.array([xi[:T_min] for xi in x_trajs])   # (N, T)
    v = np.array([vi[:T_min] for vi in v_trajs])
    a = np.array([ai[:T_min] for ai in a_trajs])

    # Compute mean force at all points
    if force_func is not None:
        Fx = force_func(x)  # (N, T)
    elif k_force is not None:
        Fx = k_force * x
    else:
        raise ValueError("Provide force_func or k_force")

    # Precompute a_orth = a - F(x)
    a_orth_all = a - Fx  # (N, T)

    # Build linear system vectorized: loop over t0, vectorize over tau
    rows_G = []
    rows_A = []

    for t0 in range(0, t0_max_idx + 1):
        v_t0 = v[:, t0].copy()
        v_t0 -= np.mean(v_t0)

        if w_func is not None:
            w = w_func(t0, x, v)
            wv = w * v_t0
            denom = np.sum(w)
        else:
            wv = v_t0
            denom = N

        # Max usable tau
        tau_end = min(tau_max_idx, T_min - t0 - 1, n_kernel - t0 - 1)
        if tau_end < 1:
            continue

        # G block: <wv * a_orth(t0+tau)> for tau=1..tau_end
        t_ends = t0 + np.arange(1, tau_end + 1)
        G_block = np.dot(wv, a_orth_all[:, t_ends]) / denom

        # Precompute wv @ v for all time indices
        wv_dot_v = np.dot(wv, v) / denom  # (T,)

        # A block: for each tau, A[j] = wv_dot_v[t_end - j]
        # Trapezoidal quadrature: half-weight at integration endpoints
        # (j=0 and j=j_max-1) to reduce collinearity and improve K(0).
        A_block = np.zeros((tau_end, n_kernel))
        for ri, tau in enumerate(range(1, tau_end + 1)):
            t_end = t0 + tau
            j_max = min(n_kernel, t_end + 1)
            # indices t_end, t_end-1, ..., t_end-j_max+1
            idx = np.arange(t_end, t_end - j_max, -1)
            A_block[ri, :j_max] = wv_dot_v[idx]
            A_block[ri, 0] *= 0.5
            if j_max > 1:
                A_block[ri, j_max - 1] *= 0.5

        rows_G.append(G_block)
        rows_A.append(dt * A_block)

    G_vec = np.concatenate(rows_G)
    A_mat = np.vstack(rows_A)
    b_vec = -G_vec  # solve A @ K = b

    # Smoothness penalty: penalize ||D @ K||^2 (first differences)
    # instead of ||K||^2, to avoid shrinking K(0) toward zero.
    D = _smoothness_matrix(n_kernel)
    DTD = D.T @ D
    ATA = A_mat.T @ A_mat

    # --- Select regularization parameter ---
    if reg == "gcv":
        reg_value = _lcurve_smooth_reg(A_mat, b_vec, DTD)
    elif reg is None:
        reg_value = _lcurve_smooth_reg(A_mat, b_vec, DTD)
    else:
        reg_value = float(reg)

    # Smoothness-regularized least squares:
    # K = argmin |A @ K - b|^2 + reg * |D @ K|^2
    # Solution: K = (A^T A + reg * D^T D)^{-1} A^T b
    ATb = A_mat.T @ b_vec
    K = np.linalg.solve(ATA + reg_value * DTD, ATb)
    residual = np.sum((A_mat @ K - b_vec)**2)

    time = np.arange(n_kernel) * dt

    return {
        "K": K,
        "time": time,
        "residual": residual,
        "reg": reg_value,
        "n_rows": len(G_vec),
        "A_mat": A_mat,
        "G_vec": G_vec,
    }


# ============================================================
# Test with 2D double-well
# ============================================================

if __name__ == "__main__":
    # Import the trajectory generator from the other script
    from nonstationary_kernel import (
        generate_2d_doublewell_trajectories,
        compute_kernel,
        volterra_second_kind_rect,
    )

    # --- Parameters ---
    omega_y = 3.0
    kBT = 0.5
    dt = 0.001
    N_trajs = 20000
    coupling = 1.5
    tau_mem = 0.1
    gamma_mat = np.array([[10.0, 2.0],
                          [2.0,  5.0]])

    def K_true_xx(t):
        return gamma_mat[0, 0] * np.exp(-t / tau_mem)

    def F_x_exact(x, y):
        return -4 * x**3 + 4 * x - 2 * coupling * x * y

    # --- Generate trajectories from barrier top, y displaced ---
    rng = np.random.default_rng(42)
    T_traj = 2.0
    nsteps = int(T_traj / dt) + 10
    print(f"Generating {N_trajs} trajectories from barrier top...")
    x_trajs, vx_trajs, ax_trajs, xy_trajs, fx_trajs = \
        generate_2d_doublewell_trajectories(
            gamma_mat, tau_mem, kBT, dt, nsteps,
            N_trajs, x0=0.0, y0=3.0, omega_y=omega_y,
            coupling=coupling, rng=rng)

    # --- Method 1: standard Volterra (t0=0 only) ---
    trunc = 2000
    print("\nMethod 1: Volterra from t0=0...")
    result_v = compute_kernel(
        x_trajs, vx_trajs, ax_trajs,
        k_force=None, dt=dt, trunc=trunc,
        method="second_kind_rect",
        force_func=lambda x_arr: np.array(
            [f[:x_arr.shape[1]] for f in fx_trajs]))

    # --- Method 2: Least-squares with multiple t0 ---
    n_kernel = 200   # kernel points (0.2 time units)
    t0_max = 50      # use t0 = 0 to 0.05 time units
    tau_max = 200    # lags up to 0.2 time units

    print(f"\nMethod 2: Least-squares (n_kernel={n_kernel}, "
          f"t0_max={t0_max}, tau_max={tau_max})...")
    print(f"  Building {(t0_max+1)*tau_max} equations "
          f"for {n_kernel} unknowns...")

    result_lsq = extract_kernel_lsq(
        x_trajs, vx_trajs, ax_trajs, dt,
        n_kernel=n_kernel,
        t0_max_idx=t0_max,
        tau_max_idx=tau_max,
        force_func=lambda x: np.array(
            [fx_trajs[i][:x.shape[1]] for i in range(len(fx_trajs))]))

    print(f"  System: {result_lsq['n_rows']} rows x {n_kernel} cols")

    # --- Method 3: LSQ with t0=0 only (for comparison) ---
    print("\nMethod 3: LSQ with t0=0 only...")
    result_lsq0 = extract_kernel_lsq(
        x_trajs, vx_trajs, ax_trajs, dt,
        n_kernel=n_kernel,
        t0_max_idx=0,
        tau_max_idx=tau_max,
        force_func=lambda x: np.array(
            [fx_trajs[i][:x.shape[1]] for i in range(len(fx_trajs))]))

    # --- Equilibrium reference (stationary, from other script) ---
    print("\nGenerating equilibrium reference...")
    gamma_sqrt = np.linalg.cholesky(gamma_mat)
    exp_dec = np.exp(-dt / tau_mem)
    ou_std = np.sqrt(kBT * (1 - exp_dec**2))

    N_eq = 50
    nsteps_eq = 200000
    rng_eq = np.random.default_rng(123)

    def fft_corr(a, b, trunc):
        N = len(a)
        nfft = 2 ** int(np.ceil(np.log2(2 * N)))
        fa = np.fft.rfft(a, n=nfft)
        fb = np.fft.rfft(b, n=nfft)
        corr = np.fft.irfft(np.conj(fa) * fb, n=nfft)[:trunc]
        norm = np.arange(N, N - trunc, -1, dtype=float)
        return corr / norm

    C_vv_eq = np.zeros(trunc)
    C_av_eq = np.zeros(trunc)
    C_Fv_eq = np.zeros(trunc)

    for n in range(N_eq):
        xx = np.zeros(nsteps_eq); yy = np.zeros(nsteps_eq)
        vx = np.zeros(nsteps_eq); vy = np.zeros(nsteps_eq)
        ax_det = np.zeros(nsteps_eq); fxarr = np.zeros(nsteps_eq)
        sx, sy, nx, ny = 0., 0., rng_eq.normal(0, np.sqrt(kBT)), rng_eq.normal(0, np.sqrt(kBT))
        xx[0] = 1.0 + rng_eq.normal(0, 0.1)
        yy[0] = rng_eq.normal(0, np.sqrt(kBT / omega_y**2))
        vx[0] = rng_eq.normal(0, np.sqrt(kBT))
        vy[0] = rng_eq.normal(0, np.sqrt(kBT))

        for i in range(nsteps_eq - 1):
            ffx = -4*xx[i]**3 + 4*xx[i] - 2*coupling*xx[i]*yy[i]
            ffy = -omega_y**2*yy[i] - coupling*xx[i]**2
            fxarr[i] = ffx
            mx = gamma_mat[0,0]*sx + gamma_mat[0,1]*sy
            my = gamma_mat[1,0]*sx + gamma_mat[1,1]*sy
            nfx = gamma_sqrt[0,0]*nx + gamma_sqrt[0,1]*ny
            nfy = gamma_sqrt[1,0]*nx + gamma_sqrt[1,1]*ny
            ax_det[i] = ffx - mx
            vx[i+1] = vx[i] + dt*(ffx - mx + nfx)
            vy[i+1] = vy[i] + dt*(ffy - my + nfy)
            xx[i+1] = xx[i] + dt*vx[i]
            yy[i+1] = yy[i] + dt*vy[i]
            sx = sx*exp_dec + vx[i]*dt
            sy = sy*exp_dec + vy[i]*dt
            nx = nx*exp_dec + ou_std*rng_eq.normal()
            ny = ny*exp_dec + ou_std*rng_eq.normal()

        fxarr[-1] = -4*xx[-1]**3 + 4*xx[-1] - 2*coupling*xx[-1]*yy[-1]
        ax_det[-1] = fxarr[-1] - (gamma_mat[0,0]*sx + gamma_mat[0,1]*sy)

        eq_start = 10000
        C_vv_eq += fft_corr(vx[eq_start:], vx[eq_start:], trunc) / N_eq
        C_av_eq += fft_corr(vx[eq_start:], ax_det[eq_start:], trunc) / N_eq
        C_Fv_eq += fft_corr(vx[eq_start:], fxarr[eq_start:], trunc) / N_eq

    C_orth_eq = C_av_eq - C_Fv_eq
    Cprime_eq = np.gradient(C_orth_eq, dt)
    K_eq, t_eq = volterra_second_kind_rect(Cprime_eq, C_av_eq, C_vv_eq[0], dt)
    print(f"  <vx^2>_eq = {C_vv_eq[0]:.4f}")

    # --- Plot: zoomed to kernel range ---
    t_max_plot = n_kernel * dt * 1.5  # show a bit beyond kernel range

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   constrained_layout=True)

    t_plot = np.arange(n_kernel) * dt
    K_ref = K_true_xx(t_plot)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_plot, initial=0)

    ax1.plot(t_plot, K_ref, "k--", lw=2.5,
             label=r"$K_{xx}^{\mathrm{true}}$", zorder=10)
    ax2.plot(t_plot, intK_ref, "k--", lw=2.5, label="true", zorder=10)

    cases = [
        (t_eq, K_eq, "eq stationary"),
        (result_v["time"], result_v["K"], "Volterra t0=0"),
        (result_lsq0["time"], result_lsq0["K"], "LSQ t0=0 only"),
        (result_lsq["time"], result_lsq["K"],
         f"LSQ t0=0..{t0_max*dt:.2f}"),
    ]

    for t, K, lab in cases:
        n = min(len(t), len(K), n_kernel)
        intK = scipy.integrate.cumulative_trapezoid(K[:n], t[:n], initial=0)
        ax1.plot(t[:n], K[:n], lw=1.5, alpha=0.8, label=lab)
        ax2.plot(t[:n], intK, lw=1.5, alpha=0.8, label=lab)

    ax1.set_xlim(0, t_max_plot)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$K(t)$")
    ax1.set_title(f"Kernel extraction: Volterra vs LSQ (N={N_trajs})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlim(0, t_max_plot)
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\int_0^t K(s)\,ds$")
    ax2.set_title("Cumulative integral")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.savefig("nonstationary_kernel_lsq_test.png", dpi=150)
    plt.show()

    print(f"\nDone. LSQ used {result_lsq['n_rows']} equations "
          f"for {n_kernel} unknowns.")
