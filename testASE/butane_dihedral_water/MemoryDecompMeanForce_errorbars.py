#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Memory kernel decomposition (internal/water) with error bars from 5 independent seeds.

Decomposition: a1 = internal (intra + constr + hess), a2 = water.
Data file: gromacs_run/cv_accel_data.npz
Arrays: phi, hess, intra, water, constr, ddot_fd (all concatenated over 5 runs of 99 frames)
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf

truncation = 10000
center = True
n_jobs = 8
str1 = "internal"
str2 = "water"

# --- Load data and split into 5 runs ---
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

# Split into per-run arrays
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


def compute_kernels_single(run):
    """Compute all kernel components and mean forces for one run."""
    nf = len(run["phi"])
    time_arr = np.arange(nf) * dt  # time in ps

    # Build acceleration components
    a_internal = run["intra"] + run["constr"] + run["hess"]
    a_water = run["water"]
    atot = a_internal + a_water  # = ddot_fd approximately

    # Reshape to (nf, 1) for VolterraBasis
    phi = run["phi"][:, None]
    # Estimate velocity by FD of phi
    phidot = np.zeros_like(phi)
    phidot[1:-1, 0] = (run["phi"][2:] - run["phi"][:-2]) / (2 * dt)
    phidot[0, 0] = (run["phi"][1] - run["phi"][0]) / dt
    phidot[-1, 0] = (run["phi"][-1] - run["phi"][-2]) / dt

    atot_2d = atot[:, None]
    a1 = a_internal[:, None]
    a2 = a_water[:, None]

    # Center force components
    a1 = a1 - np.mean(a1)
    a2 = a2 - np.mean(a2)
    atot_2d = atot_2d - np.mean(atot_2d)

    acc = [atot_2d, a1, a2]

    results = {}

    for i in range(3):
        xva_list = []
        xf = vb.xframe(phi, time_arr, v=phidot, a=atot_2d, fix_time=True)
        xf = xf.assign({"Lobs": (["time", "dim_x"], acc[i])})
        xva_list.append(xf)

        estimator = vb.Estimator_gle(xva_list, vb.Pos_gle_const_kernel,
                                     bf.LinearFeatures(to_center=center),
                                     trunc=truncation, saveall=False,
                                     L_obs="Lobs", n_jobs=n_jobs, verbose=False)

        model = estimator.compute_mean_force()
        estimator.compute_corrs()
        inv_ker_gram = estimator.compute_effective_mass().eff_mass.values
        model = estimator.compute_kernel(method="trapz")

        estimator.xva_list[0].update({"a_comp": (["time", "dim_x"], acc[i])})
        time, kernel_part = estimator.compute_projected_corrs(left_op="a_comp")

        bins = np.histogram_bin_edges(xf["x"], bins=15)
        xbin = (bins[1:] + bins[:-1]) / 2.0
        force = model.force_eval(xbin)

        if i == 0:
            results["K0"] = kernel_part.ravel()
            results["F0"] = force
            results["xbin"] = xbin
        elif i == 1:
            results["K11"] = kernel_part.ravel()
            results["F1"] = force
        elif i == 2:
            results["K22"] = kernel_part.ravel()
            results["F2"] = force

        # Cross terms
        if i == 1:
            xva_list2 = []
            xf2 = vb.xframe(phi, time_arr, v=phidot, a=atot_2d, fix_time=True)
            xf2 = xf2.assign({"Lobs": (["time", "dim_x"], a1)})
            xva_list2.append(xf2)

            est2 = vb.Estimator_gle(xva_list2, vb.Pos_gle_const_kernel,
                                    bf.LinearFeatures(to_center=center),
                                    trunc=truncation, saveall=False,
                                    L_obs="Lobs", n_jobs=n_jobs, verbose=False)
            est2.compute_mean_force()
            est2.compute_corrs()
            inv_ker_gram = est2.compute_effective_mass().eff_mass.values
            est2.compute_kernel(method="trapz")
            est2.xva_list[0].update({"a_comp": (["time", "dim_x"], a2)})
            time, kernel_part = est2.compute_projected_corrs(left_op="a_comp")
            results["K12"] = kernel_part.ravel()

        elif i == 2:
            xva_list2 = []
            xf2 = vb.xframe(phi, time_arr, v=phidot, a=atot_2d, fix_time=True)
            xf2 = xf2.assign({"Lobs": (["time", "dim_x"], a2)})
            xva_list2.append(xf2)

            est2 = vb.Estimator_gle(xva_list2, vb.Pos_gle_const_kernel,
                                    bf.LinearFeatures(to_center=center),
                                    trunc=truncation, saveall=False,
                                    L_obs="Lobs", n_jobs=n_jobs, verbose=False)
            est2.compute_mean_force()
            est2.compute_corrs()
            inv_ker_gram = est2.compute_effective_mass().eff_mass.values
            est2.compute_kernel(method="trapz")
            est2.xva_list[0].update({"a_comp": (["time", "dim_x"], a1)})
            time, kernel_part = est2.compute_projected_corrs(left_op="a_comp")
            results["K21"] = kernel_part.ravel()

    # Finite-difference kernel
    xva_list = []
    xf = vb.xframe(phi, time_arr, v=phidot)
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)

    est_fd = vb.Estimator_gle(xva_list, vb.Pos_gle_const_kernel,
                              bf.LinearFeatures(to_center=True),
                              trunc=truncation, saveall=False, n_jobs=n_jobs, verbose=False)
    est_fd.compute_mean_force()
    est_fd.compute_corrs()
    est_fd.compute_kernel(method="trapz")
    K0_fd_raw = est_fd.model.kernel[:, 0, 0]
    # Truncate to match time array length (projected corrs may drop a frame)
    results["K0_fd"] = K0_fd_raw[:len(time)]

    # Apply inv_ker_gram scaling
    ikg = inv_ker_gram.ravel()
    for key in ["K0", "K11", "K22", "K12", "K21"]:
        results[key] = results[key] * ikg

    results["time"] = time

    return results


# --- Compute for all seeds ---
print(f"Computing kernel decomposition for {n_seeds} seeds...")
all_results = []
for s in range(n_seeds):
    print(f"  Seed {s+1}/{n_seeds}")
    all_results.append(compute_kernels_single(runs[s]))

# --- Aggregate: mean and standard error ---
kernel_keys = ["K0", "K11", "K22", "K12", "K21", "K0_fd"]
force_keys = ["F0", "F1", "F2"]
time = all_results[0]["time"]
xbin = all_results[0]["xbin"]

# Truncate all kernel arrays to common minimum length (K0_fd may be shorter)
nk = min(len(time), *(len(all_results[0][key]) for key in kernel_keys))
time = time[:nk]
for r in all_results:
    r["time"] = r["time"][:nk]
    for key in kernel_keys:
        r[key] = r[key][:nk]

kernels_mean = {}
kernels_sem = {}
for key in kernel_keys:
    stacked = np.array([r[key] for r in all_results])
    kernels_mean[key] = np.mean(stacked, axis=0)
    kernels_sem[key] = np.std(stacked, axis=0, ddof=1) / np.sqrt(n_seeds)

forces_mean = {}
forces_sem = {}
for key in force_keys:
    stacked = np.array([r[key] for r in all_results])
    forces_mean[key] = np.mean(stacked, axis=0)
    forces_sem[key] = np.std(stacked, axis=0, ddof=1) / np.sqrt(n_seeds)

# --- Save kernels ---
for key in kernel_keys:
    if key == "K0":
        fname = "kerneltot.dat"
    elif key == "K0_fd":
        fname = "kerneltot_fd.dat"
    else:
        fname = f"kernel{key[1:]}.dat"
    n = min(len(time), len(kernels_mean[key]))
    np.savetxt(fname, np.column_stack((time[:n], kernels_mean[key][:n], kernels_sem[key][:n])),
               header="time  mean  sem")

for key, name in zip(force_keys, ["meanFtot.dat", f"meanF{str1}.dat", f"meanF{str2}.dat"]):
    np.savetxt(name, np.column_stack((xbin, forces_mean[key], forces_sem[key])),
               header="xbin  mean  sem")

# --- Save cumulative integrals ---
for key in kernel_keys:
    n = min(len(time), len(kernels_mean[key]))
    intK = scipy.integrate.cumulative_trapezoid(kernels_mean[key][:n], time[:n], initial=0)
    if key == "K0":
        fname = "intkerneltot.dat"
    elif key == "K0_fd":
        fname = "intkerneltot_fd.dat"
    else:
        fname = f"intkernel{key[1:]}.dat"
    np.savetxt(fname, np.column_stack((time[:n], intK)), header="time  intK")

# --- Plot ---
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10,
    "axes.labelpad": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})

K11 = kernels_mean["K11"]
K22 = kernels_mean["K22"]
K12 = kernels_mean["K12"]
K21 = kernels_mean["K21"]
K0 = kernels_mean["K0"]
K0_fd = kernels_mean["K0_fd"]
Ksum = K11 + K22 + K12 + K21

# --- Figure 1: Kernel decomposition ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 7), constrained_layout=True,
    gridspec_kw={"height_ratios": [2.2, 1.4]}
)

plot_items = [
    ("K11",  f"{str1}-{str1}", "-"),
    ("K22",  f"{str2}-{str2}", "-"),
    ("K12",  f"{str1}-{str2}", "-"),
    ("K21",  f"{str2}-{str1}", "-"),
    ("K0",   "tot",            "--"),
    ("K0_fd", "tot_fd",        "--"),
]

for key, lab, ls in plot_items:
    m = kernels_mean[key]
    s = kernels_sem[key]
    ax1.plot(time[1:], m[1:], label=lab, lw=2, linestyle=ls)
    ax1.fill_between(time[1:], (m - s)[1:], (m + s)[1:], alpha=0.25)

ax1.plot(time[1:], Ksum[1:], label="sum", lw=2, linestyle="--")
Ksum_sem = np.sqrt(kernels_sem["K11"]**2 + kernels_sem["K22"]**2 +
                   kernels_sem["K12"]**2 + kernels_sem["K21"]**2)
ax1.fill_between(time[1:], (Ksum - Ksum_sem)[1:], (Ksum + Ksum_sem)[1:], alpha=0.25)

ax1.set_title(f"Kernel decomposition ({str1}/{str2})")
ax1.set_xlabel("time [ps]")
ax1.set_ylabel(r"$K_{xy}\ [\mathrm{ps}^{-2}]$")
ax1.set_xscale("log")
ax1.axhline(0, lw=1, alpha=0.5)
ax1.grid(True, which="major", alpha=0.25)
ax1.grid(True, which="minor", alpha=0.12)
ax1.legend(ncol=2, title="components")

err = K0 - Ksum
err_sem = np.sqrt(kernels_sem["K0"]**2 + Ksum_sem**2)

ax2.plot(time[1:], err[1:], lw=2)
ax2.fill_between(time[1:], (err - err_sem)[1:], (err + err_sem)[1:], alpha=0.3)
ax2.set_title("error")
ax2.set_xlabel("time [ps]")
ax2.set_ylabel(r"$(K_{\mathrm{tot}} - \sum K_{xy})$")
ax2.set_xscale("log")
ax2.axhline(0, lw=1, alpha=0.5)
ax2.grid(True, which="major", alpha=0.25)
ax2.grid(True, which="minor", alpha=0.12)

plt.savefig(f"Kdecomp_{str1}{str2}_errorbars.png")
plt.show()

# --- Figure 2: Cumulative integral of kernels ---
fig2, (ax3, ax4) = plt.subplots(
    2, 1, figsize=(8, 7), constrained_layout=True,
    gridspec_kw={"height_ratios": [2.2, 1.4]}
)

styles = {
    "tot":    "--",
    "tot_fd": "--",
    "sum":    "--",
}

intK0 = scipy.integrate.cumulative_trapezoid(K0, time, initial=0)
intK11 = scipy.integrate.cumulative_trapezoid(K11, time, initial=0)
intK22 = scipy.integrate.cumulative_trapezoid(K22, time, initial=0)
intK12 = scipy.integrate.cumulative_trapezoid(K12, time, initial=0)
intK21 = scipy.integrate.cumulative_trapezoid(K21, time, initial=0)
intK0_fd = scipy.integrate.cumulative_trapezoid(K0_fd, time, initial=0)
intKsum = intK11 + intK22 + intK12 + intK21

for K, lab in [(intK11, f"{str1}-{str1}"), (intK22, f"{str2}-{str2}"),
               (intK12, f"{str1}-{str2}"), (intK21, f"{str2}-{str1}"),
               (intK0, "tot"), (intKsum, "sum"), (intK0_fd, "tot_fd")]:
    ax3.plot(time[1:], K[1:], label=lab, lw=2, linestyle=styles.get(lab, "-"))

ax3.set_title(r"Cumulative integral $\int_0^t K_{xy}(s)\,ds$")
ax3.set_xlabel("time [ps]")
ax3.set_ylabel(r"$\int_0^t K_{xy}(s)\,ds$")
ax3.set_xscale("log")
ax3.axhline(0, lw=1, alpha=0.5)
ax3.grid(True, which="major", alpha=0.25)
ax3.grid(True, which="minor", alpha=0.12)
ax3.legend(ncol=2, title="components")

int_err = intK0 - intKsum
ax4.plot(time[1:], int_err[1:], lw=2)
ax4.set_title("error on integral")
ax4.set_xlabel("time [ps]")
ax4.set_ylabel(r"$\int_0^t (K_{\mathrm{tot}} - \sum K_{xy})\,ds$")
ax4.set_xscale("log")
ax4.axhline(0, lw=1, alpha=0.5)
ax4.grid(True, which="major", alpha=0.25)
ax4.grid(True, which="minor", alpha=0.12)

plt.savefig(f"{str1}{str2}_integral_errorbars.png")
plt.show()
