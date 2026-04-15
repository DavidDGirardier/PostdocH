#!/usr/bin/env python3
"""
Non-stationary memory kernel extraction from the GLE.

Computes the memory kernel K(t) from multiple independent trajectories
using non-stationary (t=0) correlation functions and a user-provided
linear mean force F(x) = k * x.

No VolterraBasis dependency -- uses only numpy/scipy/matplotlib.

GLE:  a(t) = k*x(t) - int_0^t K(t-s) v(s) ds + eta(t)

Multiply by v(0), ensemble average (eta orthogonal to v(0)):
  <a(t) v(0)> = k <x(t) v(0)> - int_0^t K(t-s) <v(s) v(0)> ds

Define:
  C(t) = <a(t) v(0)> - k <x(t) v(0)>    (orthogonalized RHS)
  B(t) = <v(t) v(0)>                      (velocity autocorrelation)

Volterra equation (first kind):  C(t) = -int_0^t K(t-s) B(s) ds
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


# ============================================================
# Non-stationary correlation functions
# ============================================================

def compute_nonstat_corrs(x_trajs, v_trajs, a_trajs, trunc,
                         force_func=None):
    """
    Compute non-stationary correlation functions from t=0,
    ensemble-averaged over independent trajectories.

    Parameters
    ----------
    x_trajs : list of 1D arrays
        Position trajectories, one per independent run.
    v_trajs : list of 1D arrays
        Velocity trajectories.
    a_trajs : list of 1D arrays
        Acceleration trajectories.
    trunc : int
        Maximum lag in number of frames.
    force_func : callable or None
        If provided, F(x) -> force. Used to compute <F(x(t)) v(0)>
        directly instead of relying on k*<x(t) v(0)>.

    Returns
    -------
    C_av : ndarray (trunc,)
        <a(t) * v(0)>
    C_xv : ndarray (trunc,)
        <x(t) * v(0)>
    C_vv : ndarray (trunc,)
        <v(t) * v(0)>
    """
    N = len(x_trajs)
    T_min = min(len(x) for x in x_trajs)
    trunc = min(trunc, T_min)

    # Stack to (N, T) and truncate
    x = np.array([xi[:trunc] for xi in x_trajs])
    v = np.array([vi[:trunc] for vi in v_trajs])
    a = np.array([ai[:trunc] for ai in a_trajs])

    v0 = v[:, 0]  # (N,)
    # Center v(0) so correlations = covariances.
    # This removes the mean-response contribution and makes the
    # method work for any v(0) distribution, not just zero-mean.
    v0 = v0 - np.mean(v0)

    C_av = np.mean(a * v0[:, None], axis=0)
    C_xv = np.mean(x * v0[:, None], axis=0)
    C_vv = np.mean(v * v0[:, None], axis=0)

    # If a force function or pre-computed force arrays are provided,
    # compute <F(x(t)) * v(0)> for exact mean force subtraction.
    if force_func is not None:
        Fx = force_func(x)  # (N, trunc)
        C_Fv = np.mean(Fx * v0[:, None], axis=0)
        return C_av, C_xv, C_vv, C_Fv

    return C_av, C_xv, C_vv


# ============================================================
# Reweighted stationary correlation functions
# ============================================================

def compute_reweighted_corrs(x_trajs, v_trajs, a_trajs, trunc,
                             k_force, kBT, n_bins=50):
    """
    Compute stationary correlation functions using all time origins,
    reweighted by rho_eq(q(t)) / rho_neq(q(t), t) to correct for
    non-equilibrium sampling at each time t.

    C(tau) = <w(q(t), t) * a(t) * v(t+tau)>_{t, trajs}
    where w(q, t) = rho_eq(q) / rho_neq(q, t)

    Parameters
    ----------
    x_trajs, v_trajs, a_trajs : lists of 1D arrays
    trunc : int
    k_force : float
    kBT : float
    n_bins : int

    Returns
    -------
    C_av, C_xv, C_vv : ndarray (trunc,)
    """
    N = len(x_trajs)
    T = min(len(xi) for xi in x_trajs)

    # Stack trajectories: (N, T)
    x = np.array([xi[:T] for xi in x_trajs])
    v = np.array([vi[:T] for vi in v_trajs])
    a = np.array([ai[:T] for ai in a_trajs])

    # Compute weights w(q, t) = rho_eq(q) / rho_neq(q, t) for each frame
    # rho_eq(q) ∝ exp(-|k|q²/(2kBT))
    log_rho_eq = -0.5 * abs(k_force) * x**2 / kBT  # unnormalized log

    # Estimate rho_neq(q, t) at each time t from histogram over trajectories
    w = np.ones_like(x)
    for t in range(T):
        x_t = x[:, t]  # N samples at time t
        counts, bin_edges = np.histogram(x_t, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # rho_eq normalized on the same bins
        rho_eq = np.exp(-0.5 * abs(k_force) * bin_centers**2 / kBT)
        rho_eq /= np.trapz(rho_eq, bin_centers)

        # Assign weight to each trajectory at this time
        bin_idx = np.digitize(x_t, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        rho_neq = np.maximum(counts[bin_idx], 1e-10)
        rho_eq_at_x = rho_eq[bin_idx]
        w[:, t] = rho_eq_at_x / rho_neq

    # Compute reweighted correlations
    C_av = np.zeros(trunc)
    C_xv = np.zeros(trunc)
    C_vv = np.zeros(trunc)
    W = np.zeros(trunc)

    for tau in range(trunc):
        n_origins = T - tau
        if n_origins <= 0:
            break
        # w[:, :n_origins] weights the time origin t
        wt = w[:, :n_origins]  # (N, n_origins)
        C_av[tau] = np.sum(wt * a[:, :n_origins] * v[:, tau:tau + n_origins])
        C_xv[tau] = np.sum(wt * x[:, :n_origins] * v[:, tau:tau + n_origins])
        C_vv[tau] = np.sum(wt * v[:, :n_origins] * v[:, tau:tau + n_origins])
        W[tau] = np.sum(wt)

    C_av /= W
    C_xv /= W
    C_vv /= W

    return C_av, C_xv, C_vv


# ============================================================
# Volterra solvers (scalar, first kind)
# ============================================================

def volterra_rect(C, B, dt):
    """
    Solve the Volterra equation of the first kind (rectangular rule):
        C(t) = -int_0^t K(t-s) B(s) ds

    Translated from fkernel.f90 kernel_first_kind_rect (scalar case).

    Parameters
    ----------
    C : ndarray (M,)
        RHS of the Volterra equation, indexed from t=0.
    B : ndarray (M,)
        Velocity autocorrelation, indexed from t=0.
    dt : float

    Returns
    -------
    K : ndarray (M-2,)
        Memory kernel from t=dt to t=(M-2)*dt.
    time : ndarray (M-2,)
        Time array for the kernel.
    """
    M = len(C)
    N = M - 2  # usable kernel points
    K = np.zeros(N)

    inv_dtB1 = 1.0 / (dt * B[1])
    K[0] = -inv_dtB1 * C[1]

    for i in range(1, N):
        # integral = dt * sum_{j=0}^{i-1} B[i+1-j] * K[j]
        integral = dt * np.dot(B[i + 1:1:-1][:i], K[:i])
        K[i] = -inv_dtB1 * (integral + C[i + 1])

    time = np.arange(1, N + 1) * dt
    return K, time


def volterra_trapz(C, B, dt, k0=None):
    """
    Solve the Volterra equation of the first kind (trapezoidal rule):
        C(t) = -int_0^t K(t-s) B(s) ds

    Translated from fkernel.f90 kernel_first_kind_trapz (scalar case).
    Includes smoothing step matching gle_estimation.py line 331.

    Parameters
    ----------
    C : ndarray (M,)
        RHS of the Volterra equation, indexed from t=0.
    B : ndarray (M,)
        Velocity autocorrelation, indexed from t=0.
    dt : float
    k0 : float or None
        Initial value K(0). If None, estimated from finite differences.

    Returns
    -------
    K : ndarray
        Memory kernel (after smoothing), starting at t=0.
    time : ndarray
        Time array for the kernel.
    """
    M = len(C)
    K_raw = np.zeros(M)

    if k0 is None:
        # Estimate K(0) from the derivative of C at t=0
        k0 = -(C[1] - C[0]) / (dt * B[0])

    inv_halfdt_B0 = 1.0 / (0.5 * dt * B[0])
    K_raw[0] = k0

    for i in range(1, M):
        # Trapezoidal: half-weight at j=0, full weight for j=1..i-1
        s = 0.5 * dt * B[i] * K_raw[0]
        if i > 1:
            s += dt * np.dot(B[i - 1:0:-1], K_raw[1:i])
        K_raw[i] = -inv_halfdt_B0 * (s + C[i])

    # Smoothing (matching VolterraBasis)
    K = 0.5 * (K_raw[1:-1] + 0.5 * (K_raw[:-2] + K_raw[2:]))
    K = np.insert(K, 0, k0)

    time = np.arange(len(K)) * dt
    return K, time


def volterra_deconv(C, B, dt, reg=1e-2):
    """
    Solve the Volterra equation via Fourier deconvolution:
        C(t) = -int_0^t K(t-s) B(s) ds  <==>  C = -(K * B)

    In Fourier space: K_hat = -C_hat / B_hat, regularized with
    a Wiener-like filter to suppress noise amplification.

    Parameters
    ----------
    C : ndarray (M,)
        Orthogonalized RHS, from t=0.
    B : ndarray (M,)
        Velocity autocorrelation, from t=0.
    dt : float
    reg : float
        Regularization parameter (fraction of max |B_hat|^2).

    Returns
    -------
    K : ndarray (M,)
        Memory kernel from t=0.
    time : ndarray (M,)
    """
    M = len(C)
    # Zero-pad to avoid circular convolution artifacts
    N_fft = 2 * M
    C_hat = np.fft.rfft(C, n=N_fft)
    B_hat = np.fft.rfft(B, n=N_fft)

    # Wiener-regularized deconvolution
    B2 = np.abs(B_hat) ** 2
    lam = reg * np.max(B2)
    K_hat = -C_hat * np.conj(B_hat) / (B2 + lam) / dt

    K = np.fft.irfft(K_hat, n=N_fft)[:M]
    time = np.arange(M) * dt
    return K, time


def volterra_second_kind_rect(Cprime, Bprime, B0, dt, k0=None):
    """
    Solve the Volterra equation of the second kind:
        K(t) = -[C'(t) + int_0^t K(u) B'(t-u) du] / B(0)

    Derived by differentiating the first-kind equation
        C(t) = -int_0^t K(t-s) B(s) ds
    This is much better conditioned than first-kind (no 1/dt factor).

    Translated from fkernel.f90 kernel_second_kind_rect (scalar case).

    Parameters
    ----------
    Cprime : ndarray (M,)
        Time derivative of the orthogonalized RHS, d/dt C_orth.
    Bprime : ndarray (M,)
        Time derivative of the velocity autocorrelation, d/dt C_vv.
    B0 : float
        C_vv(0) = <v(0)^2>.
    dt : float
    k0 : float or None
        Initial value K(0). If None, estimated as -Cprime[0] / B0.

    Returns
    -------
    K : ndarray (M,)
        Memory kernel from t=0.
    time : ndarray (M,)
        Time array.
    """
    M = len(Cprime)
    K = np.zeros(M)

    if k0 is None:
        k0 = -Cprime[0] / B0

    inv_B0 = 1.0 / B0
    K[0] = k0

    for i in range(1, M):
        # integral = dt * sum_{j=0}^{i-1} Bprime[i-j] * K[j]
        jj = np.arange(i)
        integral = dt * np.sum(Bprime[i - jj] * K[jj])
        K[i] = -inv_B0 * (integral + Cprime[i])

    time = np.arange(M) * dt
    return K, time


# ============================================================
# Main entry point
# ============================================================

def compute_kernel(x_trajs, v_trajs, a_trajs, k_force, dt, trunc,
                   method="second_kind_rect", corr_mode="nonstat",
                   kBT=None, force_func=None):
    """
    Compute the memory kernel from trajectory data.

    Parameters
    ----------
    x_trajs, v_trajs, a_trajs : lists of 1D arrays
    k_force : float or None
        Linear mean force coefficient: F(x) = k_force * x.
        Ignored if force_func is provided.
    dt : float
    trunc : int
        Maximum lag in frames.
    method : str
        "rect", "trapz", "second_kind_rect", or "deconv:REG".
    corr_mode : str
        "nonstat" : non-stationary correlations from t=0 (default)
        "reweighted" : stationary correlations reweighted by rho_eq/rho_sampled
    kBT : float or None
        Thermal energy (required for corr_mode="reweighted").
    force_func : callable or None
        If provided, F(x) evaluated directly. Overrides k_force.

    Returns
    -------
    result : dict with "time", "K", "C_vv", "C_xv", "C_av", "dt"
    """
    if corr_mode == "nonstat":
        corr_out = compute_nonstat_corrs(
            x_trajs, v_trajs, a_trajs, trunc, force_func=force_func)
        if force_func is not None:
            C_av, C_xv, C_vv, C_Fv = corr_out
        else:
            C_av, C_xv, C_vv = corr_out
            C_Fv = None
    elif corr_mode == "reweighted":
        if kBT is None:
            raise ValueError("kBT required for reweighted correlations")
        C_av, C_xv, C_vv = compute_reweighted_corrs(
            x_trajs, v_trajs, a_trajs, trunc, k_force, kBT)
        C_Fv = None
    else:
        raise ValueError(f"Unknown corr_mode '{corr_mode}'")

    # Orthogonalized RHS: C(t) = <a(t) v(0)> - <F(x(t)) v(0)>
    if C_Fv is not None:
        C_orth = C_av - C_Fv
    else:
        C_orth = C_av - k_force * C_xv

    if method == "rect":
        K, time = volterra_rect(C_orth, C_vv, dt)
    elif method == "trapz":
        K, time = volterra_trapz(C_orth, C_vv, dt)
    elif method == "second_kind_rect":
        # B'(t) = d/dt <v(t)v(0)> = <a(t)v(0)> = C_av (exact, no FD)
        Bprime = C_av
        B0 = C_vv[0]
        # C_orth'(t): use FD of C_orth
        if C_Fv is not None:
            # Exact force: C_orth = C_av - C_Fv, take FD
            Cprime = np.gradient(C_orth, dt)
        else:
            # Linear force: C_orth' = d/dt C_av - k * C_vv (less FD needed)
            Cprime = np.gradient(C_av, dt) - k_force * C_vv
        K, time = volterra_second_kind_rect(Cprime, Bprime, B0, dt)
    elif method.startswith("deconv"):
        # Parse optional reg parameter: "deconv" or "deconv:1e-3"
        if ":" in method:
            reg = float(method.split(":")[1])
        else:
            reg = 1e-2
        K, time = volterra_deconv(C_orth, C_vv, dt, reg=reg)
    else:
        raise ValueError(f"Unknown method '{method}', use "
                         "'rect', 'trapz', 'second_kind_rect', or 'deconv'")

    return {
        "time": time,
        "K": K,
        "C_vv": C_vv,
        "C_xv": C_xv,
        "C_av": C_av,
        "dt": dt,
    }


# ============================================================
# Verification
# ============================================================

def check_volterra(K, B, C, dt, method="rect"):
    """
    Reconstruct C(t) from K and B via numerical integration and compare
    to the input C. Returns the reconstructed C and the residual.
    """
    if method == "rect":
        # K is defined from t=dt, C and B from t=0
        N = len(K)
        C_recon = np.zeros(N)
        for i in range(N):
            # C[i+1] = -dt * sum_{j=0}^{i} B[i+1-j] * K[j]
            jj = np.arange(i + 1)
            C_recon[i] = -dt * np.sum(B[i + 1 - jj] * K[jj])
        residual = C[1:N + 1] - C_recon
    else:
        N = len(K)
        C_recon = np.zeros(N)
        for i in range(N):
            s = 0.5 * dt * B[i] * K[0]
            if i > 1:
                jj = np.arange(1, i)
                s += dt * np.sum(B[i - jj] * K[jj])
            C_recon[i] = -s
        residual = C[:N] - C_recon

    return C_recon, residual


# ============================================================
# Plotting
# ============================================================

def plot_kernel(result, title="Memory kernel", savefig=None):
    """Plot the memory kernel and its cumulative integral."""
    time = result["time"]
    K = result["K"]
    intK = scipy.integrate.cumulative_trapezoid(K, time, initial=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                   constrained_layout=True)

    ax1.plot(time, K, lw=2)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$K(t)$")
    ax1.set_title(title)
    ax1.axhline(0, lw=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, intK, lw=2)
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\int_0^t K(s)\,ds$")
    ax2.set_title("Cumulative integral")
    ax2.axhline(0, lw=1, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_correlations(result, savefig=None):
    """Diagnostic plot of the non-stationary correlation functions."""
    dt = result["dt"]
    C_vv = result["C_vv"]
    C_xv = result["C_xv"]
    C_av = result["C_av"]
    time = np.arange(len(C_vv)) * dt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(time, C_vv, lw=2)
    axes[0].set_title(r"$\langle v(t)\, v(0) \rangle$")

    axes[1].plot(time, C_xv, lw=2)
    axes[1].set_title(r"$\langle x(t)\, v(0) \rangle$")

    axes[2].plot(time, C_av, lw=2)
    axes[2].set_title(r"$\langle a(t)\, v(0) \rangle$")

    for ax in axes:
        ax.set_xlabel("time")
        ax.axhline(0, lw=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)
    plt.show()


# ============================================================
# Example: synthetic test with exponential kernel
# ============================================================

def generate_gle_trajectories(k_force, gamma, tau_mem, kBT, dt, nsteps,
                              N_trajs, rng=None):
    """
    Generate equilibrium GLE trajectories with exponential memory kernel.

    Uses the auxiliary variable trick: for K(t) = gamma * exp(-t/tau),
    define s(t) = int_0^t exp(-(t-u)/tau) v(u) du  =>  ds/dt = v - s/tau.
    The GLE becomes a Markovian system:
        dx/dt = v
        dv/dt = k*x - gamma*s + eta(t)
        ds/dt = v - s/tau
    with white noise <eta(t) eta(t')> = 2*kBT*gamma*delta(t-t').

    Parameters
    ----------
    k_force : float
        Linear force coefficient F(x) = k*x (negative for restoring).
    gamma : float
        Memory kernel amplitude.
    tau_mem : float
        Memory timescale.
    kBT : float
        Thermal energy.
    dt : float
        Timestep.
    nsteps : int
        Number of integration steps per trajectory.
    N_trajs : int
        Number of independent trajectories.
    rng : numpy Generator or None

    Returns
    -------
    x_trajs, v_trajs, a_det_trajs : lists of 1D arrays (length nsteps each)
        a_det is the deterministic acceleration k*x - gamma*s (no noise),
        analogous to F_total/m in real MD data.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_std = np.sqrt(2 * gamma * kBT / dt)
    exp_decay = np.exp(-dt / tau_mem)

    x_all = np.zeros((N_trajs, nsteps))
    v_all = np.zeros((N_trajs, nsteps))
    a_det_all = np.zeros((N_trajs, nsteps))  # deterministic part only
    s_all = np.zeros(N_trajs)

    # Initial conditions (can be non-equilibrium)
    x_all[:, 0] = rng.normal(0, np.sqrt(kBT / abs(k_force)), N_trajs)
    v_all[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

    for i in range(nsteps - 1):
        eta = noise_std * rng.normal(size=N_trajs)
        a_det_all[:, i] = k_force * x_all[:, i] - gamma * s_all
        a_full = a_det_all[:, i] + eta
        v_half = v_all[:, i] + 0.5 * dt * a_full
        x_all[:, i + 1] = x_all[:, i] + dt * v_half
        s_all = s_all * exp_decay + v_all[:, i] * dt
        eta_new = noise_std * rng.normal(size=N_trajs)
        a_det_all[:, i + 1] = (k_force * x_all[:, i + 1]
                                - gamma * s_all)
        v_all[:, i + 1] = v_half + 0.5 * dt * (a_det_all[:, i + 1]
                                                 + eta_new)

    x_trajs = list(x_all)
    v_trajs = list(v_all)
    a_det_trajs = list(a_det_all)
    return x_trajs, v_trajs, a_det_trajs


def run_test(k_force, gamma, tau_mem, kBT, dt, N_trajs, trunc,
             x0_init=None, v0_init=None, label="equilibrium"):
    """
    Run kernel extraction test.

    Parameters
    ----------
    x0_init : float or None
        If None, draw x(0) from equilibrium. If float, fixed x(0).
    v0_init : float, tuple(float, float), or None
        If None, draw v(0) from Boltzmann. If float, fixed v(0).
        If tuple (mean, std), draw v(0) ~ N(mean, std).
    label : str
    """
    nsteps = trunc + 10

    def K_true(t):
        return gamma * np.exp(-t / tau_mem)

    rng = np.random.default_rng(42)
    print(f"\n=== {label} ===")
    print(f"Generating {N_trajs} trajectories ({nsteps} steps, dt={dt})...")

    need_custom = (x0_init is not None) or (v0_init is not None)

    if not need_custom:
        x_trajs, v_trajs, a_trajs = generate_gle_trajectories(
            k_force, gamma, tau_mem, kBT, dt, nsteps, N_trajs, rng)
    else:
        noise_std = np.sqrt(2 * gamma * kBT / dt)
        exp_decay = np.exp(-dt / tau_mem)

        x_all = np.zeros((N_trajs, nsteps))
        v_all = np.zeros((N_trajs, nsteps))
        a_det_all = np.zeros((N_trajs, nsteps))
        s_all = np.zeros(N_trajs)

        if x0_init is not None:
            x_all[:, 0] = x0_init
        else:
            x_all[:, 0] = rng.normal(0, np.sqrt(kBT / abs(k_force)), N_trajs)

        if isinstance(v0_init, tuple):
            v_all[:, 0] = rng.normal(v0_init[0], v0_init[1], N_trajs)
        elif v0_init is not None:
            v_all[:, 0] = v0_init
        else:
            v_all[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)

        for i in range(nsteps - 1):
            eta = noise_std * rng.normal(size=N_trajs)
            a_det_all[:, i] = k_force * x_all[:, i] - gamma * s_all
            a_full = a_det_all[:, i] + eta
            v_half = v_all[:, i] + 0.5 * dt * a_full
            x_all[:, i + 1] = x_all[:, i] + dt * v_half
            s_all = s_all * exp_decay + v_all[:, i] * dt
            eta_new = noise_std * rng.normal(size=N_trajs)
            a_det_all[:, i + 1] = (k_force * x_all[:, i + 1]
                                    - gamma * s_all)
            v_all[:, i + 1] = v_half + 0.5 * dt * (a_det_all[:, i + 1]
                                                     + eta_new)

        x_trajs = list(x_all)
        v_trajs = list(v_all)
        a_trajs = list(a_det_all)

    x0 = np.array([x[0] for x in x_trajs])
    v0 = np.array([v[0] for v in v_trajs])
    print(f"  <x(0)> = {np.mean(x0):.4f},  <x(0)^2> = {np.mean(x0**2):.4f}")
    print(f"  <v(0)> = {np.mean(v0):.4f},  <v(0)^2> = {np.mean(v0**2):.4f}")

    # --- Extract kernel ---
    method = "second_kind_rect"
    print(f"Computing memory kernel ({method})...")
    result = compute_kernel(x_trajs, v_trajs, a_trajs,
                            k_force, dt, trunc, method=method)

    result["_x_trajs"] = x_trajs
    result["_v_trajs"] = v_trajs
    result["_a_trajs"] = a_trajs

    # --- Forward check ---
    C_vv = result["C_vv"]
    C_orth = result["C_av"] - k_force * result["C_xv"]
    N_check = min(trunc, len(C_vv))
    t_check = np.arange(N_check) * dt
    C_orth_predicted = np.zeros(N_check)
    for i in range(N_check):
        s_arr = np.arange(i + 1) * dt
        integrand = K_true(t_check[i] - s_arr) * C_vv[:i + 1]
        C_orth_predicted[i] = -np.trapz(integrand, dx=dt)
    fwd_err = np.max(np.abs(C_orth[:N_check] - C_orth_predicted))
    print(f"Forward check max error: {fwd_err:.4e}")

    return result, K_true


# ============================================================
# 2D double-well GLE trajectory generator
# ============================================================

def generate_2d_doublewell_trajectories(gamma_mat, tau_mem, kBT, dt, nsteps,
                                        N_trajs, x0, y0, omega_y=2.0,
                                        coupling=0.0, rng=None):
    """
    Generate 2D Langevin trajectories (with exponential memory) in:
        V(x,y) = x^4 - 2x^2 + 0.5*omega_y^2*y^2 + coupling*x^2*y

    Parameters
    ----------
    gamma_mat : (2,2) array
        Memory kernel amplitude matrix.
    tau_mem, kBT, dt : float
    nsteps, N_trajs : int
    x0, y0 : float
        Initial position.
    omega_y : float
        Frequency of harmonic y potential.
    coupling : float
        Coupling strength for the x^2*y term.
    rng : numpy Generator

    Returns
    -------
    x_trajs, v_x_trajs, a_x_det_trajs : lists of 1D arrays
        Projected onto x coordinate only.
    xy_trajs : list of (nsteps, 2) arrays
        Full 2D positions.
    fx_trajs : lists of 1D arrays
        F_x(x(t), y(t)) at each frame (for exact mean force subtraction).
    """
    if rng is None:
        rng = np.random.default_rng()

    gamma_mat = np.asarray(gamma_mat)
    noise_cov = 2 * kBT * gamma_mat / dt  # discrete noise covariance
    # Cholesky for correlated noise
    L_noise = np.linalg.cholesky(noise_cov)
    exp_decay = np.exp(-dt / tau_mem)

    # Allocate
    x_all = np.zeros((N_trajs, nsteps))   # x coordinate
    y_all = np.zeros((N_trajs, nsteps))   # y coordinate
    vx_all = np.zeros((N_trajs, nsteps))
    vy_all = np.zeros((N_trajs, nsteps))
    ax_det_all = np.zeros((N_trajs, nsteps))
    fx_all = np.zeros((N_trajs, nsteps))  # F_x(x,y) for mean force
    sx_all = np.zeros(N_trajs)  # auxiliary for memory (x)
    sy_all = np.zeros(N_trajs)  # auxiliary for memory (y)

    def compute_forces(xx, yy):
        # V = x^4 - 2x^2 + 0.5*omega_y^2*y^2 + coupling*x^2*y
        ffx = -4*xx**3 + 4*xx - 2*coupling*xx*yy
        ffy = -omega_y**2*yy - coupling*xx**2
        return ffx, ffy

    # Correct augmented GLE with colored noise (OU process):
    #   dv = (F - Gamma @ s + Gamma^{1/2} @ n) dt
    #   ds = (v - s/tau) dt
    #   dn = (-n/tau) dt + sqrt(2 kBT / tau) dW
    # where n is the noise auxiliary variable, <n^2>_eq = kBT * I
    gamma_sqrt = np.linalg.cholesky(gamma_mat)  # Gamma^{1/2}
    ou_decay = np.exp(-dt / tau_mem)
    ou_noise_std = np.sqrt(kBT * (1 - ou_decay**2))  # exact OU step

    # Noise auxiliary variables
    nx_all = np.zeros(N_trajs)
    ny_all = np.zeros(N_trajs)

    # Initial conditions
    x_all[:, 0] = x0
    y_all[:, 0] = y0
    vx_all[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)
    vy_all[:, 0] = rng.normal(0, np.sqrt(kBT), N_trajs)
    # Initialize noise from equilibrium
    nx_all = rng.normal(0, np.sqrt(kBT), N_trajs)
    ny_all = rng.normal(0, np.sqrt(kBT), N_trajs)

    for i in range(nsteps - 1):
        fx, fy = compute_forces(x_all[:, i], y_all[:, i])
        fx_all[:, i] = fx

        # Memory: Gamma @ s
        mem_x = gamma_mat[0, 0] * sx_all + gamma_mat[0, 1] * sy_all
        mem_y = gamma_mat[1, 0] * sx_all + gamma_mat[1, 1] * sy_all

        # Colored noise force: Gamma^{1/2} @ n
        noise_x = gamma_sqrt[0, 0] * nx_all + gamma_sqrt[0, 1] * ny_all
        noise_y = gamma_sqrt[1, 0] * nx_all + gamma_sqrt[1, 1] * ny_all

        # Deterministic acceleration (recorded, without noise)
        ax_det_all[:, i] = fx - mem_x

        # Full acceleration (for integration)
        ax_full = fx - mem_x + noise_x
        ay_full = fy - mem_y + noise_y

        # Euler-Maruyama integration
        vx_all[:, i + 1] = vx_all[:, i] + dt * ax_full
        vy_all[:, i + 1] = vy_all[:, i] + dt * ay_full
        x_all[:, i + 1] = x_all[:, i] + dt * vx_all[:, i]
        y_all[:, i + 1] = y_all[:, i] + dt * vy_all[:, i]

        # Update memory auxiliary (exact exponential)
        sx_all = sx_all * exp_decay + vx_all[:, i] * dt
        sy_all = sy_all * exp_decay + vy_all[:, i] * dt

        # Update noise auxiliary (exact OU step)
        nx_all = nx_all * ou_decay + ou_noise_std * rng.normal(size=N_trajs)
        ny_all = ny_all * ou_decay + ou_noise_std * rng.normal(size=N_trajs)

    # Last frame forces
    fx_last, _ = compute_forces(x_all[:, -1], y_all[:, -1])
    fx_all[:, -1] = fx_last
    ax_det_all[:, -1] = fx_last - (gamma_mat[0, 0] * sx_all
                                     + gamma_mat[0, 1] * sy_all)

    x_trajs = list(x_all)
    vx_trajs = list(vx_all)
    ax_trajs = list(ax_det_all)
    fx_trajs = list(fx_all)
    xy_trajs = [np.column_stack([x_all[n], y_all[n]]) for n in range(N_trajs)]

    return x_trajs, vx_trajs, ax_trajs, xy_trajs, fx_trajs


if __name__ == "__main__":
    # --- 2D double-well parameters ---
    # V(x,y) = x^4 - 2x^2 + 0.5*omega_y^2*y^2
    # F_x = -4x^3 + 4x,  at x=0: F_x ≈ 4x  (linear, repulsive)
    # F_y = -omega_y^2 * y

    omega_y = 3.0
    kBT = 0.5        # low-ish T so barrier matters (barrier height = 1)
    dt = 0.001
    N_trajs = 20000
    trunc = 2000
    tau_mem = 0.1

    # Memory kernel matrix (with x-y coupling)
    gamma_mat = np.array([[10.0, 2.0],
                          [2.0,  5.0]])

    # The scalar kernel projected on x is K_xx(t) = gamma_mat[0,0] * exp(-t/tau)
    # (plus coupling contributions)
    def K_true_xx(t):
        return gamma_mat[0, 0] * np.exp(-t / tau_mem)

    # x-y potential coupling
    coupling = 1.5  # V += coupling * x^2 * y

    rng = np.random.default_rng(42)

    # --- Generate two sets: y equilibrium vs y displaced ---
    rng_eq = np.random.default_rng(42)
    print(f"Generating trajectories: x0=0, y0=0 (eq), coupling={coupling}...")
    x_eq, vx_eq, ax_eq, xy_eq, fx_eq = \
        generate_2d_doublewell_trajectories(
            gamma_mat, tau_mem, kBT, dt, trunc + 10,
            N_trajs, x0=0.0, y0=0.0, omega_y=omega_y,
            coupling=coupling, rng=rng_eq)

    y0 = 3.0
    rng_neq = np.random.default_rng(42)
    print(f"Generating trajectories: x0=0, y0={y0} (non-eq), "
          f"coupling={coupling}...")
    x_trajs, vx_trajs, ax_trajs, xy_trajs, fx_trajs = \
        generate_2d_doublewell_trajectories(
            gamma_mat, tau_mem, kBT, dt, trunc + 10,
            N_trajs, x0=0.0, y0=y0, omega_y=omega_y,
            coupling=coupling, rng=rng_neq)

    print(f"  N={N_trajs}, trunc={trunc}, dt={dt}")
    x0_arr = np.array([x[0] for x in x_trajs])
    vx0_arr = np.array([v[0] for v in vx_trajs])
    print(f"  <x(0)> = {np.mean(x0_arr):.4f}")
    print(f"  <vx(0)> = {np.mean(vx0_arr):.4f}, "
          f"<vx(0)^2> = {np.mean(vx0_arr**2):.4f}")

    # --- PMF force ---
    alpha = 1 - coupling**2 / (2 * omega_y**2)
    print(f"\nPMF: A(x) = {alpha:.3f}*x^4 - 2*x^2")
    def F_pmf(x):
        return -4 * alpha * x**3 + 4 * x

    # --- Extract kernels for all 4 combinations ---
    def make_fx_func(fx_list):
        return lambda x_arr: np.array([f[:x_arr.shape[1]] for f in fx_list])

    cases = {}

    print("y0=0, exact F_x(x,y)...")
    cases["y=0, exact F"] = compute_kernel(
        x_eq, vx_eq, ax_eq, k_force=None, dt=dt, trunc=trunc,
        method="second_kind_rect", force_func=make_fx_func(fx_eq))

    print("y0=0, PMF...")
    cases["y=0, PMF"] = compute_kernel(
        x_eq, vx_eq, ax_eq, k_force=None, dt=dt, trunc=trunc,
        method="second_kind_rect", force_func=F_pmf)

    print(f"y0={y0}, exact F_x(x,y)...")
    cases[f"y={y0}, exact F"] = compute_kernel(
        x_trajs, vx_trajs, ax_trajs, k_force=None, dt=dt, trunc=trunc,
        method="second_kind_rect", force_func=make_fx_func(fx_trajs))

    print(f"y0={y0}, PMF...")
    cases[f"y={y0}, PMF"] = compute_kernel(
        x_trajs, vx_trajs, ax_trajs, k_force=None, dt=dt, trunc=trunc,
        method="second_kind_rect", force_func=F_pmf)

    # --- Ground truth: long equilibrium trajectories (Euler-Maruyama) ---
    print("\n--- Equilibrium reference (stationary correlations) ---")
    N_eq_long = 50
    nsteps_eq = 200000
    dt_eq = dt
    rng_long = np.random.default_rng(123)
    print(f"Generating {N_eq_long} long eq trajectories "
          f"({nsteps_eq} steps, Euler-Maruyama)...")

    # Same correct colored-noise integrator as the generator
    gamma_sqrt_eq = np.linalg.cholesky(gamma_mat)
    exp_dec = np.exp(-dt_eq / tau_mem)
    ou_std = np.sqrt(kBT * (1 - exp_dec**2))

    x_long, vx_long, ax_long, fx_long = [], [], [], []
    for n in range(N_eq_long):
        xx = np.zeros(nsteps_eq)
        yy = np.zeros(nsteps_eq)
        vx = np.zeros(nsteps_eq)
        vy = np.zeros(nsteps_eq)
        ax_det = np.zeros(nsteps_eq)
        fxarr = np.zeros(nsteps_eq)
        sx, sy = 0.0, 0.0
        nx = rng_long.normal(0, np.sqrt(kBT))
        ny = rng_long.normal(0, np.sqrt(kBT))

        xx[0] = 1.0 + rng_long.normal(0, 0.1)
        yy[0] = rng_long.normal(0, np.sqrt(kBT / omega_y**2))
        vx[0] = rng_long.normal(0, np.sqrt(kBT))
        vy[0] = rng_long.normal(0, np.sqrt(kBT))

        for i in range(nsteps_eq - 1):
            ffx = -4*xx[i]**3 + 4*xx[i] - 2*coupling*xx[i]*yy[i]
            ffy = -omega_y**2*yy[i] - coupling*xx[i]**2
            fxarr[i] = ffx
            mem_x = gamma_mat[0,0]*sx + gamma_mat[0,1]*sy
            mem_y = gamma_mat[1,0]*sx + gamma_mat[1,1]*sy
            noise_fx = gamma_sqrt_eq[0,0]*nx + gamma_sqrt_eq[0,1]*ny
            noise_fy = gamma_sqrt_eq[1,0]*nx + gamma_sqrt_eq[1,1]*ny
            ax_det[i] = ffx - mem_x

            vx[i+1] = vx[i] + dt_eq * (ffx - mem_x + noise_fx)
            vy[i+1] = vy[i] + dt_eq * (ffy - mem_y + noise_fy)
            xx[i+1] = xx[i] + dt_eq * vx[i]
            yy[i+1] = yy[i] + dt_eq * vy[i]
            sx = sx * exp_dec + vx[i] * dt_eq
            sy = sy * exp_dec + vy[i] * dt_eq
            nx = nx * exp_dec + ou_std * rng_long.normal()
            ny = ny * exp_dec + ou_std * rng_long.normal()

        fxarr[-1] = -4*xx[-1]**3 + 4*xx[-1] - 2*coupling*xx[-1]*yy[-1]
        ax_det[-1] = fxarr[-1] - (gamma_mat[0,0]*sx + gamma_mat[0,1]*sy)

        x_long.append(xx)
        vx_long.append(vx)
        ax_long.append(ax_det)
        fx_long.append(fxarr)

    # Stationary (time-averaged) correlations via FFT
    def fft_corr(a, b, trunc):
        """Stationary correlation <a(t) b(t+tau)> via FFT."""
        N = len(a)
        nfft = 2 ** int(np.ceil(np.log2(2 * N)))
        fa = np.fft.rfft(a, n=nfft)
        fb = np.fft.rfft(b, n=nfft)
        corr = np.fft.irfft(np.conj(fa) * fb, n=nfft)[:trunc]
        # Normalize by number of pairs at each lag
        norm = np.arange(N, N - trunc, -1, dtype=float)
        return corr / norm

    print("Computing stationary correlations...")
    C_vv_eq = np.zeros(trunc)
    C_av_eq = np.zeros(trunc)
    C_Fv_eq = np.zeros(trunc)
    for i in range(N_eq_long):
        # Discard first 10000 steps for equilibration
        eq_start = 10000
        vxi = vx_long[i][eq_start:]
        axi = ax_long[i][eq_start:]
        fxi = fx_long[i][eq_start:]
        # corr(v(t), v(t+tau)) with v at origin, v at lag
        C_vv_eq += fft_corr(vxi, vxi, trunc) / N_eq_long
        # corr(v(t), a(t+tau)) -- v at origin, a_det at lag
        C_av_eq += fft_corr(vxi, axi, trunc) / N_eq_long
        # corr(v(t), F_x(t+tau))
        C_Fv_eq += fft_corr(vxi, fxi, trunc) / N_eq_long

    C_orth_eq = C_av_eq - C_Fv_eq
    Cprime_eq = np.gradient(C_orth_eq, dt)
    Bprime_eq = C_av_eq
    B0_eq = C_vv_eq[0]

    print(f"  <vx^2>_eq = {B0_eq:.4f}")
    K_eq, t_eq = volterra_second_kind_rect(Cprime_eq, Bprime_eq, B0_eq, dt)
    cases["eq stationary"] = {"time": t_eq, "K": K_eq}

    result = cases[f"y={y0}, exact F"]  # for correlation plot

    # --- Plot trajectories ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Sample trajectories in x-y plane
    for n in range(min(50, N_trajs)):
        axes[0].plot(xy_trajs[n][:500, 0], xy_trajs[n][:500, 1],
                     lw=0.3, alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Trajectories (x-y plane)")
    axes[0].axvline(0, ls=":", color="k", alpha=0.3)

    # x(t) for a few trajectories
    t_arr = np.arange(min(500, trunc)) * dt
    for n in range(min(20, N_trajs)):
        axes[1].plot(t_arr, x_trajs[n][:len(t_arr)], lw=0.5, alpha=0.5)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("x")
    axes[1].set_title("x(t) from barrier top")
    axes[1].axhline(1, ls=":", color="k", alpha=0.3, label=r"$\pm 1$ (minima)")
    axes[1].axhline(-1, ls=":", color="k", alpha=0.3)

    # Potential
    xp = np.linspace(-1.8, 1.8, 200)
    Vx = xp**4 - 2 * xp**2
    axes[2].plot(xp, Vx, "k-", lw=2)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$V(x) = x^4 - 2x^2$")
    axes[2].set_title("Double-well potential")
    axes[2].axhline(0, ls=":", color="gray", alpha=0.3)

    plt.savefig("doublewell_trajectories.png", dpi=150)
    plt.show()

    # --- Plot kernel ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   constrained_layout=True)

    t_ref = list(cases.values())[0]["time"]
    K_ref = K_true_xx(t_ref)
    intK_ref = scipy.integrate.cumulative_trapezoid(K_ref, t_ref, initial=0)

    ax1.plot(t_ref, K_ref, "k--", lw=2.5, label=r"$K_{xx}^{\mathrm{true}}$",
             zorder=10)
    ax2.plot(t_ref, intK_ref, "k--", lw=2.5, label="true", zorder=10)

    for lab, res in cases.items():
        t = res["time"]
        K = res["K"]
        intK = scipy.integrate.cumulative_trapezoid(K, t, initial=0)
        ax1.plot(t, K, lw=1.5, alpha=0.8, label=lab)
        ax2.plot(t, intK, lw=1.5, alpha=0.8, label=lab)

    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$K(t)$")
    ax1.set_title(f"2D double-well, coupling={coupling} (N={N_trajs})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\int_0^t K(s)\,ds$")
    ax2.set_title("Cumulative integral")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.savefig("nonstationary_kernel_test.png", dpi=150)
    plt.show()

    # --- Correlation diagnostics ---
    plot_correlations(result, savefig="nonstationary_corrs_test.png")
