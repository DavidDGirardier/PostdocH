#!/usr/bin/env python3
"""
Aggregate per-seed cv_accel_data.npz files into one combined file.

Usage:
  python3 aggregate_seeds.py [output_name]
  default output: cv_accel_data.npz

Automatically finds all run_seed*/cv_accel_data.npz in the script's directory.
"""
import numpy as np
import os
import glob
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
outname = sys.argv[1] if len(sys.argv) > 1 else 'cv_accel_data.npz'

files = sorted(glob.glob(os.path.join(base_dir, 'run_seed*', 'cv_accel_data.npz')))
if not files:
    print("No run_seed*/cv_accel_data.npz found.")
    sys.exit(1)

all_phi, all_hess, all_intra, all_water, all_constr, all_ddot_fd = [], [], [], [], [], []
seeds = []
dt = None

for f in files:
    seed_dir = os.path.basename(os.path.dirname(f))
    seed = int(seed_dir.replace('run_seed', ''))
    seeds.append(seed)
    rd = np.load(f)
    dt = float(rd['dt'])
    s = slice(1, len(rd['phi']) - 1)  # interior frames only
    all_phi.append(rd['phi'][s])
    all_hess.append(rd['hess'][s])
    all_intra.append(rd['intra'][s])
    all_water.append(rd['water'][s])
    all_constr.append(rd['constr'][s])
    all_ddot_fd.append(rd['ddot_fd'][s])
    print(f"  {seed_dir}: {len(rd['phi'])} frames (seed={seed})")

phi = np.concatenate(all_phi)
hess = np.concatenate(all_hess)
intra = np.concatenate(all_intra)
water = np.concatenate(all_water)
constr = np.concatenate(all_constr)
ddot_fd = np.concatenate(all_ddot_fd)
nruns = len(seeds)

outfile = os.path.join(base_dir, outname)
np.savez(outfile, phi=phi, hess=hess, intra=intra, water=water,
         constr=constr, ddot_fd=ddot_fd, dt=dt, seeds=seeds, nruns=nruns)
print(f"\nAggregated {nruns} seeds {seeds}: {len(phi)} interior frames")
print(f"Saved to {outfile}")
