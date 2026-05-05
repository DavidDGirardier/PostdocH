#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Position-dependent memory kernel decomposition (internal/water) with error bars.

Decomposition basis C_k = h_k(phi) (B-splines on the dihedral):
  L_1 C_k = h_k''(phi) phi_dot^2 + h_k'(phi) (intra+constr+hess)        [internal]
  L_2 C_k = h_k'(phi) (water)                                            [water]

Cross kernel (Mori-Zwanzig with position-resolved projector):
  K_ij(t, o_1) = - sum_{k,k'} (G_c^{-1})_{k,k'}
                   <L_i C_{k'}, eta_{j,t}>  nabla h_k(o_1)
where G_c = bkbkcorrw[t=0] = (1/beta) <nabla h_k, M_phi^{-1} nabla h_{k'}>_phi
and  <L_i C_{k'}, eta_{j,t}> is obtained via compute_projected_corrs after
fitting Pos_gle with L_obs = a_j  (so the Volterra-projected residual is eta_j).

Data file: cv_accel_data.npz
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf

truncation = 1
center = True
n_jobs = 4
Nsplines = 10
str1 = "internal"
str2 = "water"
# === ADDED: number of grid points for position evaluation ===========
n_grid = 200
# === ADDED: toggle Gram smoothing via compute_pos_effective_mass ====
# False -> G_c from bkbkcorrw[t=0]                   (raw, default)
# True  -> G_c built from a smoothed M_phi^{-1}(phi) (less Gram noise)
SMOOTH_GRAM = False
# === ADDED: Volterra solver for compute_kernel ======================
# Methods compatible with compute_projected_corrs / compute_corrs_w_noise
# (see VolterraBasis/models.py:202-207):
#   "trapz"               first-kind, trapz quadrature, 3-point post-smoothing
#                         -> introduces a spike at t=dt
#   "rect" / "rectangular"  first-kind, rectangular quadrature, no smoothing
#   "second_kind_rect"    second-kind form, rect quadrature, no smoothing
#                         -> matches the K_ij formula's numerics best
#   "second_kind_trapz"   second-kind form, trapz quadrature, no smoothing
# (midpoint / midpoint_w_richardson are NOT supported by compute_corrs_w_noise)
KERNEL_METHOD = "rect"
# === ADDED: patch the trapz spike at t=dt =============================
# Choose how (or whether) to remove the trapz solver's t=dt artifact:
#   None    : no patch (default -- keep the spike, mask in analysis if needed)
#   "K1=K2" : kernel[1] := kernel[2]   (replace t=dt frame with t=2dt)
#   "K0=K1" : kernel[0] := kernel[1]   (replace t=0 with t=dt smoothed value)
#   "K1=K0" : kernel[1] := kernel[0]   (collapse spike to k0)
SPIKE_PATCH = None
# ====================================================================

# --- Load data and split into 5 runs --- (UNCHANGED)
data = np.load("cv_accel_data.npz")
nruns = int(data["nruns"])
dt = float(data["dt"])
phi_all = data["phi"]
hess_all = data["hess"]
intra_all = data["intra"]
water_all = data["water"]
constr_all = data["constr"]
ddot_fd_all = data["ddot_fd"]

n_per_run = len(phi_all) // nruns

# Split into per-run arrays (UNCHANGED)
runs = []
for i in range(nruns):
    s = slice(i * n_per_run, (i + 1) * n_per_run)
    runs.append({
        "phi": phi_all[s],
        "hess": hess_all[s],
        "intra": intra_all[s],
        "water": water_all[s],
        "constr": constr_all[s],
        "ddot_fd": ddot_fd_all[s],
    })

n_seeds = nruns


def _apply_spike_patch(est):
    """In-place patch of est.model.kernel to mitigate the trapz t=dt spike."""
    if SPIKE_PATCH is None:
        return
    K = est.model.kernel.values
    if SPIKE_PATCH == "K1=K2":
        K[1, :, :] = K[2, :, :]
    elif SPIKE_PATCH == "K0=K1":
        K[0, :, :] = K[1, :, :]
    elif SPIKE_PATCH == "K1=K0":
        K[1, :, :] = K[0, :, :]
    else:
        raise ValueError(f"Unknown SPIKE_PATCH value: {SPIKE_PATCH!r}")


def compute_kernels_single(run):
    """Compute position-dependent kernel components and mean force for one run."""
    nf = len(run["phi"])
    time_arr = np.arange(nf) * dt  # time in ps

    # accelerations (UNCHANGED logic, both grouped exactly as before)
    a_internal = run["intra"] + run["constr"] + run["hess"]   # = L_1 O^v
    a_water    = run["water"]                                  # = L_2 O^v
    atot       = a_internal + a_water

    phi = run["phi"][:, None]
    phidot = np.zeros_like(phi)
    phidot[1:-1, 0] = (run["phi"][2:] - run["phi"][:-2]) / (2 * dt)
    phidot[0, 0]    = (run["phi"][1]  - run["phi"][0])   / dt
    phidot[-1, 0]   = (run["phi"][-1] - run["phi"][-2])  / dt

    # === MODIFIED: no manual centering --- Pos_gle handles the mean force
    # ===   via compute_mean_force / projection on the spline basis. ====
    a1 = a_internal
    a2 = a_water
    a0 = atot
    a0_2d = a0[:, None]
    a1_2d = a1[:, None]
    a2_2d = a2[:, None]

    # ================================================================
    # === MODIFIED: Pos_gle on a_total --- position-dependent kernel ==
    # ===  gives K_tot(phi,t), force(phi), and the kernel-Gram G_c   ==
    # ================================================================
    xf0 = vb.xframe(phi, time_arr, v=phidot, a=a0_2d, fix_time=True)
    xf0 = xf0.assign({"Lobs": (["time", "dim_x"], a0_2d)})
    est0 = vb.Estimator_gle([xf0], vb.Pos_gle,
                            bf.BSplineFeatures(Nsplines),
                            trunc=truncation, saveall=False,
                            L_obs="Lobs", n_jobs=n_jobs, verbose=False)
    est0.compute_mean_force()
    est0.compute_corrs()
    model0 = est0.compute_kernel(method=KERNEL_METHOD)
    _apply_spike_patch(est0)
    basis  = est0.model.basis

    # === ADDED: kernel Gram (raw or smoothed via M_phi^{-1}(phi)) ====
    if not SMOOTH_GRAM:
        # raw: G_c = bkbkcorrw[t=0] = <(nabla h_k . v)(nabla h_k' . v)>
        G_c = est0.bkbkcorrw.isel(time_trunc=0).values     # (Kb, Kb)
    else:
        # smoothed: G_c = (1/T) sum_t  M_phi^{-1}(phi_t) nabla h_k(phi_t) nabla h_k'(phi_t)
        # with M_phi^{-1}(phi) fitted as a basis-coefficient expansion via equipartition.
        # NOTE: model.inv_mass_eval is buggy for Pos_gle (uses N_basis_elt=Kb+1 while
        # inv_mass_coeff is fit in the gradient basis of size Kb). We contract manually.
        est0.compute_pos_effective_mass()
        inv_m_coeff = np.asarray(est0.model.inv_mass_coeff).reshape(-1)   # (Kb,)
        dbk_t  = est0.model.basis.deriv(phi)[:, :, 0]                     # (T, Kb)
        M_inv  = dbk_t @ inv_m_coeff                                       # (T,)  M_phi^{-1}(phi_t)
        G_c    = np.einsum("tk,t,tl->kl", dbk_t, M_inv, dbk_t) / len(phi)
    G_c_inv = np.linalg.inv(G_c)
    # ================================================================

    # evaluation grid
    xfine    = np.linspace(phi.min(), phi.max(), n_grid)[:, None]
    force_xt = model0.force_eval(xfine)                       # (n_grid, 1)

    # ================================================================
    # === MODIFIED: train Pos_gle on a_1 and a_2 separately, so that =
    # ===           compute_projected_corrs returns <left_op, eta_j> =
    # ================================================================
    def fit_component(a_comp_2d):
        xf = vb.xframe(phi, time_arr, v=phidot, a=a0_2d, fix_time=True)
        xf = xf.assign({"Lobs": (["time", "dim_x"], a_comp_2d)})
        est = vb.Estimator_gle([xf], vb.Pos_gle,
                               bf.BSplineFeatures(Nsplines),
                               trunc=truncation, saveall=False,
                               L_obs="Lobs", n_jobs=n_jobs, verbose=False)
        est.compute_mean_force()
        est.compute_corrs()
        est.compute_kernel(method=KERNEL_METHOD)
        _apply_spike_patch(est)
        return est

    est1 = fit_component(a1_2d)
    est2 = fit_component(a2_2d)
    # ================================================================

    # ================================================================
    # === ADDED: per-basis derivatives along the trajectory ==========
    # ===   G_k[t,k]  = h_k'(phi(t))      (basis Jacobian)            =
    # ===   H_kk[t,k] = h_k''(phi(t))     (basis Hessian)             =
    # ================================================================
    G_k  = basis.deriv(phi)[:, :, 0]            # (T, Kb)
    H_kk = basis.hessian(phi)[:, :, 0, 0]       # (T, Kb)
    v    = phidot[:, 0]                          # (T,)
    Kb   = G_k.shape[1]
    # ================================================================

    # ================================================================
    # === ADDED: L_i C_k along the trajectory ========================
    # ===   L_1 C_k = h_k'' phi_dot^2 + h_k' (intra+constr+hess)
    # ===   L_2 C_k = h_k'           (water)
    # ================================================================
    L1Ck = H_kk * (v * v)[:, None] + G_k * a1[:, None]   # (T, Kb)
    L2Ck =                            G_k * a2[:, None]  # (T, Kb)
    # ================================================================

    # ================================================================
    # === MODIFIED: 4 projected correlations <L_j C_k, eta_i> per k ==
    # ===   K_ij convention: first index i = which eta (which model);
    # ===                    second index j = which left_op L_jC.    =
    # ===   So  K11, K12 share eta_1 (=> est1, g~^(1))               =
    # ===       K21, K22 share eta_2 (=> est2, g~^(2))               =
    # ================================================================
    # probe length once
    est1.xva_list[0] = est1.xva_list[0].assign(
        {"a_comp": (["time", "dim_x"], L1Ck[:, 0:1])}
    )
    time_proj, _ = est1.compute_projected_corrs(left_op="a_comp")
    trunc_n = len(time_proj)

    # Notation: C_ij_k = <L_jC_k, eta_i>(t), so first index = eta-index = model
    C11_k = np.zeros((Kb, trunc_n))   # est1 (eta_1), left_op = L_1C
    C12_k = np.zeros((Kb, trunc_n))   # est1 (eta_1), left_op = L_2C
    C21_k = np.zeros((Kb, trunc_n))   # est2 (eta_2), left_op = L_1C
    C22_k = np.zeros((Kb, trunc_n))   # est2 (eta_2), left_op = L_2C

    for k in range(Kb):
        # eta_1 (model trained on a_1) -- K11, K12
        est1.xva_list[0] = est1.xva_list[0].assign(
            {"a_comp": (["time", "dim_x"], L1Ck[:, k:k+1])}
        )
        _, c = est1.compute_projected_corrs(left_op="a_comp")
        C11_k[k] = np.asarray(c).ravel()[:trunc_n]                # <L_1C, eta_1>

        est1.xva_list[0] = est1.xva_list[0].assign(
            {"a_comp": (["time", "dim_x"], L2Ck[:, k:k+1])}
        )
        _, c = est1.compute_projected_corrs(left_op="a_comp")
        C12_k[k] = np.asarray(c).ravel()[:trunc_n]                # <L_2C, eta_1>

        # eta_2 (model trained on a_2) -- K21, K22
        est2.xva_list[0] = est2.xva_list[0].assign(
            {"a_comp": (["time", "dim_x"], L1Ck[:, k:k+1])}
        )
        _, c = est2.compute_projected_corrs(left_op="a_comp")
        C21_k[k] = np.asarray(c).ravel()[:trunc_n]                # <L_1C, eta_2>

        est2.xva_list[0] = est2.xva_list[0].assign(
            {"a_comp": (["time", "dim_x"], L2Ck[:, k:k+1])}
        )
        _, c = est2.compute_projected_corrs(left_op="a_comp")
        C22_k[k] = np.asarray(c).ravel()[:trunc_n]                # <L_2C, eta_2>
    # ================================================================

    # ================================================================
    # === ADDED: position projection                                  =
    # ===   K_ij(t, o_1) = - inv_ker_gram                             =
    # ===                  * sum_k <L_i C_k, eta_{j,t}> nabla h_k(o_1)=
    # ================================================================
    dpsi = basis.deriv(xfine)[:, :, 0]                # (n_grid, Kb)  = nabla h_k(o_g)
    # K_ij(t, o) = sum_{k,k'} G_c^{-1}_{k k'} <L_i C_{k'}, eta_{j,t}> nabla h_k(o)
    # (sign convention matches Pos_gle.kernel_eval; closure check verified at t=0)
    # einsum: gk,kl,lt->tg  sums BOTH k and l (only g and t survive)
    K11_xt = np.einsum("gk,kl,lt->tg", dpsi, G_c_inv, C11_k)   # (trunc_n, n_grid)
    K12_xt = np.einsum("gk,kl,lt->tg", dpsi, G_c_inv, C12_k)
    K21_xt = np.einsum("gk,kl,lt->tg", dpsi, G_c_inv, C21_k)
    K22_xt = np.einsum("gk,kl,lt->tg", dpsi, G_c_inv, C22_k)
    # ================================================================

    # K_tot from Pos_gle on a_total (truncated to match)
    K0_xt = model0.kernel_eval(xfine).values[:, 0, :, 0]  # (T_ker, n_grid)
    K0_xt = K0_xt[:trunc_n]

    # === ADDED: per-component kernels K_tilde_1, K_tilde_2 from the
    # ===        same trapz Volterra solve. Used to check
    # ===        K_tilde_i ?= K_i1 + K_i2  (per-component closure)
    K1_xt = est1.model.kernel_eval(xfine).values[:, 0, :, 0][:trunc_n]
    K2_xt = est2.model.kernel_eval(xfine).values[:, 0, :, 0][:trunc_n]
    # ================================================================

    # FD-based total kernel (UNCHANGED in spirit -- const-kernel sanity check)
    xf_fd = vb.xframe(phi, time_arr, v=phidot)
    xvaf  = vb.compute_va(xf_fd)
    est_fd = vb.Estimator_gle([xvaf], vb.Pos_gle_const_kernel,
                              bf.LinearFeatures(to_center=True),
                              trunc=truncation, saveall=False,
                              n_jobs=n_jobs, verbose=False)
    est_fd.compute_mean_force()
    est_fd.compute_corrs()
    est_fd.compute_kernel(method=KERNEL_METHOD)
    K0_fd = est_fd.model.kernel[:, 0, 0].values[:trunc_n]

    # === ADDED: position-resolved K_tot from FD acceleration =========
    # ===   Pos_gle on vb.compute_va output (FD a) with same B-spline =
    # ===   basis. For unbiased runs this should ~= K0_xt.            =
    est_fd_pos = vb.Estimator_gle([xvaf], vb.Pos_gle,
                                  bf.BSplineFeatures(Nsplines),
                                  trunc=truncation, saveall=False,
                                  n_jobs=n_jobs, verbose=False)
    est_fd_pos.compute_mean_force()
    est_fd_pos.compute_corrs()
    est_fd_pos.compute_kernel(method=KERNEL_METHOD)
    K0_fd_pos = est_fd_pos.model.kernel_eval(xfine).values[:, 0, :, 0][:trunc_n]
    # ================================================================

    time = np.arange(trunc_n) * dt

    return {
        "time":      time,
        "xfine":     xfine.ravel(),
        "force_xt":  force_xt,
        "K0_xt":     K0_xt,
        "K1_xt":     K1_xt,
        "K2_xt":     K2_xt,
        "K11_xt":    K11_xt,
        "K22_xt":    K22_xt,
        "K12_xt":    K12_xt,
        "K21_xt":    K21_xt,
        "K0_fd":     K0_fd,
        "K0_fd_pos": K0_fd_pos,
    }


# --- Compute for all seeds --- (UNCHANGED loop pattern)
print(f"Computing position-dependent kernel decomposition for {n_seeds} seeds...")
all_results = []
for s in range(n_seeds):
    print(f"  Seed {s+1}/{n_seeds}")
    all_results.append(compute_kernels_single(runs[s]))

# ====================================================================
# === MODIFIED: aggregation --- arrays are now (T, n_grid) ===========
# ====================================================================
kernel_keys = ["K0_xt", "K1_xt", "K2_xt", "K11_xt", "K22_xt", "K12_xt", "K21_xt",
               "K0_fd", "K0_fd_pos"]
time  = all_results[0]["time"]
xfine = all_results[0]["xfine"]

# truncate to common length (defensive)
nk = min(len(time), *(all_results[0][key].shape[0] for key in kernel_keys))
time = time[:nk]
for r in all_results:
    r["time"] = r["time"][:nk]
    for key in kernel_keys:
        r[key] = r[key][:nk]

kernels_mean = {}
kernels_sem  = {}
for key in kernel_keys:
    stk = np.array([r[key] for r in all_results])
    kernels_mean[key] = stk.mean(axis=0)
    if n_seeds > 1:
        kernels_sem[key] = stk.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    else:
        kernels_sem[key] = np.zeros_like(kernels_mean[key])

force_stack = np.array([r["force_xt"] for r in all_results])
force_mean  = force_stack.mean(axis=0)
force_sem   = (force_stack.std(axis=0, ddof=1) / np.sqrt(n_seeds)
               if n_seeds > 1 else np.zeros_like(force_mean))
# ====================================================================

# ====================================================================
# === MODIFIED: detect FE minima / midpoints / TS for plotting =======
# ====================================================================
fe = -scipy.integrate.cumulative_trapezoid(force_mean[:, 0], xfine, initial=0)
fe -= fe.min()
from scipy.signal import argrelmin, argrelmax
minima_idx = argrelmin(fe, order=5)[0]
ts_idx     = argrelmax(fe, order=5)[0]
special    = np.sort(np.concatenate([minima_idx, ts_idx]))
mid_idx    = np.array([(special[i] + special[i+1]) // 2
                       for i in range(len(special)-1)], dtype=int)
print("FE minima:", np.rad2deg(xfine[minima_idx]).round(1), "deg")
print("FE TS    :", np.rad2deg(xfine[ts_idx]).round(1),     "deg")
if len(mid_idx):
    print("midpts   :", np.rad2deg(xfine[mid_idx]).round(1), "deg")
# ====================================================================

# --- save .dat files (phi-averaged, parity with original layout) ---
fnames = {"K0_xt":     "kerneltot.dat",
          "K1_xt":     "kernel1.dat",
          "K2_xt":     "kernel2.dat",
          "K11_xt":    "kernel11.dat",
          "K22_xt":    "kernel22.dat",
          "K12_xt":    "kernel12.dat",
          "K21_xt":    "kernel21.dat",
          "K0_fd":     "kerneltot_fd.dat",
          "K0_fd_pos": "kerneltot_fd_pos.dat"}
for key, fname in fnames.items():
    if key == "K0_fd":                      # const-kernel: 1-D in time
        m, s = kernels_mean[key], kernels_sem[key]
    else:                                    # position-dependent: average over phi
        m = kernels_mean[key].mean(axis=1)
        s = kernels_sem[key].mean(axis=1)
    np.savetxt(fname, np.column_stack((time, m, s)),
               header="time  mean  sem")

# cumulative integrals (UNCHANGED structure, phi-averaged)
for key in kernel_keys:
    m = kernels_mean[key] if key == "K0_fd" else kernels_mean[key].mean(axis=1)
    intK = scipy.integrate.cumulative_trapezoid(m, time, initial=0)
    if key == "K0_xt":
        fname = "intkerneltot.dat"
    elif key == "K1_xt":
        fname = "intkernel1.dat"
    elif key == "K2_xt":
        fname = "intkernel2.dat"
    elif key == "K0_fd":
        fname = "intkerneltot_fd.dat"
    elif key == "K0_fd_pos":
        fname = "intkerneltot_fd_pos.dat"
    else:
        fname = f"intkernel{key.replace('_xt','')[1:]}.dat"
    np.savetxt(fname, np.column_stack((time, intK)), header="time  intK")

# ====================================================================
# === MODIFIED: plot --- 4 rows (K_ij) x 4 cols (minima/mid/TS/avg) ==
# ====================================================================
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True,
                         constrained_layout=True)
groups = [("minima",   minima_idx),
          ("midpoints", mid_idx),
          ("TS",       ts_idx),
          (r"$\langle\cdot\rangle_\varphi$", None)]
rows = [("K11_xt", f"{str1}-{str1}"),
        ("K22_xt", f"{str2}-{str2}"),
        ("K12_xt", f"{str1}-{str2}"),
        ("K21_xt", f"{str2}-{str1}")]

for r, (key, lab) in enumerate(rows):
    K = kernels_mean[key]
    for c, (gname, idxs) in enumerate(groups):
        ax = axes[r, c]
        if idxs is None:
            ax.plot(time[1:], K.mean(axis=1)[1:], lw=1.5)
        else:
            for idx in idxs:
                ax.plot(time[1:], K[1:, idx], lw=1.5,
                        label=f"{np.rad2deg(xfine[idx]):.0f}°")
            if len(idxs):
                ax.legend(fontsize=7)
        ax.set_xscale("log")
        ax.axhline(0, lw=1, alpha=0.5)
        ax.grid(True, alpha=0.25)
        if c == 0:
            ax.set_ylabel(rf"$K_{{{lab}}}(\varphi,t)$")
        if r == 3:
            ax.set_xlabel("time [ps]")
        if r == 0:
            ax.set_title(gname)

plt.savefig(f"Kdecomp_pos_{str1}{str2}_errorbars.png")
plt.show()
# ====================================================================

# ====================================================================
# === MODIFIED: closure check --- K_tot vs Sum_ij K_ij vs K_FD =======
# ====================================================================
Ksum = (kernels_mean["K11_xt"] + kernels_mean["K22_xt"]
      + kernels_mean["K12_xt"] + kernels_mean["K21_xt"])

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 7),
                                constrained_layout=True,
                                gridspec_kw={"height_ratios": [2.2, 1.4]})

ax3.plot(time[1:], kernels_mean["K0_xt"].mean(1)[1:],     label="tot (Pos_gle, atot)",  lw=2)
ax3.plot(time[1:], Ksum.mean(1)[1:],                       label=r"$\sum_{ij} K_{ij}$",  lw=2, linestyle="--")
ax3.plot(time[1:], kernels_mean["K0_fd_pos"].mean(1)[1:],  label="tot (Pos_gle, FD a)",  lw=2, linestyle="-.")
ax3.plot(time[1:], kernels_mean["K0_fd"][1:],              label="tot (const, FD a)",    lw=2, linestyle=":")
ax3.set_xscale("log")
ax3.set_xlabel("time [ps]")
ax3.set_ylabel(r"$\langle K(\varphi,t)\rangle_\varphi$ [ps$^{-2}$]")
ax3.axhline(0, lw=1, alpha=0.5)
ax3.grid(True, which="major", alpha=0.25)
ax3.grid(True, which="minor", alpha=0.12)
ax3.legend(title="components")

err = kernels_mean["K0_xt"].mean(1) - Ksum.mean(1)
ax4.plot(time[1:], err[1:], lw=2)
ax4.set_xlabel("time [ps]")
ax4.set_ylabel(r"$K_{\mathrm{tot}}-\sum_{ij}K_{ij}$")
ax4.set_xscale("log")
ax4.axhline(0, lw=1, alpha=0.5)
ax4.grid(True, which="major", alpha=0.25)
ax4.grid(True, which="minor", alpha=0.12)

plt.savefig(f"{str1}{str2}_closure_errorbars.png")
plt.show()
# ====================================================================

# --- save full position-resolved arrays ---
np.savez("Kdecomp_pos.npz",
         time=time, xfine=xfine, fe=fe,
         force=force_mean, force_sem=force_sem,
         minima_idx=minima_idx, ts_idx=ts_idx, mid_idx=mid_idx,
         **{k: kernels_mean[k] for k in kernel_keys},
         **{f"{k}_sem": kernels_sem[k] for k in kernel_keys})
print("Done.")
