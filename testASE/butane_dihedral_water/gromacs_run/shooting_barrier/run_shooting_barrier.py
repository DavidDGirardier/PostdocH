#!/usr/bin/env python3
"""
Shooting trajectories from the eclipsed barrier (phi ≈ 120°) for
kernel extraction vs reactive flux comparison.

Simplified: only records phi(t), phidot(t) = J·v, ddot_fd = d(J·v)/dt.
No force decomposition, no vacuum rerun.

Usage:
  python run_shooting_barrier.py --equilibrate
  python run_shooting_barrier.py --shoot --nshoot 200 --seed-start 1
  python run_shooting_barrier.py --equilibrate --shoot --nshoot 200
"""

import numpy as np
import subprocess
import os
import sys
import argparse
import time

GROMACS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GROMACS_DIR)
from cv_acceleration_rerun_wall_analytical import (
    phi_numpy, get_jacobian_hessian, read_trr, MASSES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GMX = 'gmx_mpi'
FULL_TOP = os.path.join(GROMACS_DIR, 'topol.top')
FF_DIR = os.path.join(GROMACS_DIR, 'gromos53a6.ff')
GRO_START = os.path.join(GROMACS_DIR, 'npt.gro')
DT = 0.002  # ps

# Barrier position: eclipsed configuration between trans and gauche+
# PMF force zero crossing: phi_numpy = 4.176 rad => PLUMED = (-4.176) % 2pi ≈ 2.107 rad ≈ 120.7°
PHI_BARRIER_RAD = 2.0071  # ~120.7 degrees


# ── Trajectory processing (just phi, phidot, ddot_fd) ──────────────────

def process_trajectory_simple(trr_path, tpr_path, dt):
    """Extract phi(t), phidot = J·v, ddot_fd = d(J·v)/dt from TRR.

    No force decomposition — only needs positions and velocities.
    """
    pos_all, vel_all, _ = read_trr(trr_path)
    nframes = pos_all.shape[0]

    phi_arr = np.zeros(nframes)
    Jv_arr = np.zeros(nframes)

    t0 = time.time()
    for k in range(nframes):
        pos_k = pos_all[k]
        vel_k = vel_all[k]
        phi_arr[k] = phi_numpy(pos_k)

        J, _ = get_jacobian_hessian(pos_k)
        Jv_arr[k] = sum(np.dot(J[i], vel_k[i]) for i in range(4))

        if (k + 1) % 2000 == 0 or k == nframes - 1:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            print(f"    frame {k+1}/{nframes} ({rate:.0f} fr/s)", flush=True)

    # ddot_fd = d(J·v)/dt via central FD
    ddot_fd = np.zeros(nframes)
    ddot_fd[1:-1] = (Jv_arr[2:] - Jv_arr[:-2]) / (2 * dt)

    return phi_arr, Jv_arr, ddot_fd


# ── GROMACS helpers ────────────────────────────────────────────────────

def write_mdp(path, nsteps, gen_vel=True, gen_seed=-1):
    """Write MDP. Records positions and velocities every step (no forces)."""
    with open(path, 'w') as f:
        f.write(f'integrator=md\ndt={DT}\nnsteps={nsteps}\n')
        f.write('nstxout=1\nnstvout=1\nnstfout=0\nnstenergy=500\n')
        f.write('nstxout-compressed=0\nnstlog=5000\n')
        f.write('cutoff-scheme=Verlet\nnstlist=10\ncoulombtype=PME\n')
        f.write('rcoulomb=0.9\nrvdw=0.9\npbc=xyz\n')
        f.write('tcoupl=v-rescale\ntc-grps=System\ntau-t=0.1\nref-t=300\n')
        f.write('pcoupl=Parrinello-Rahman\npcoupltype=isotropic\n')
        f.write('tau-p=2.0\nref-p=1.0\ncompressibility=4.5e-5\n')
        f.write('constraints=all-angles\nlincs-order=6\n')
        if gen_vel:
            f.write(f'gen-vel=yes\ngen-temp=300\ngen-seed={gen_seed}\n')
        else:
            f.write('gen-vel=no\ncontinuation=yes\n')


def write_plumed_restraint(path, at_rad, kappa=10000.0):
    with open(path, 'w') as f:
        f.write('phi: TORSION ATOMS=1,2,3,4\n')
        f.write(f'res: RESTRAINT ARG=phi AT={at_rad:.6f} KAPPA={kappa:.2f}\n')
        f.write('PRINT ARG=phi,res.bias STRIDE=100 FILE=COLVAR\n')


def write_plumed_free(path):
    with open(path, 'w') as f:
        f.write('phi: TORSION ATOMS=1,2,3,4\n')
        f.write('PRINT ARG=phi STRIDE=1 FILE=COLVAR\n')


def run_gromacs(gro, mdp, plumed_dat, workdir, tpr_name='run.tpr',
                trr_name='run.trr'):
    ff_link = os.path.join(workdir, 'gromos53a6.ff')
    if not os.path.exists(ff_link):
        os.symlink(FF_DIR, ff_link)

    tpr = os.path.join(workdir, tpr_name)
    trr = os.path.join(workdir, trr_name)

    gro_abs = os.path.abspath(gro)
    mdp_abs = os.path.abspath(mdp)
    top_abs = os.path.abspath(FULL_TOP)
    plm_abs = os.path.abspath(plumed_dat)
    tpr_abs = os.path.abspath(tpr)
    trr_abs = os.path.abspath(trr)

    cmd = [GMX, 'grompp', '-f', mdp_abs, '-c', gro_abs, '-p', top_abs,
           '-o', tpr_abs, '-maxwarn', '10']
    subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(tpr):
        return None, None

    cmd = [GMX, 'mdrun', '-s', tpr_abs, '-o', trr_abs, '-ntomp', '1',
           '-plumed', plm_abs]
    subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(trr):
        return None, None

    return trr, tpr


# ── Phase A: Bath equilibration at barrier ─────────────────────────────

def equilibrate_bath(n_configs=50, interval_ps=20.0, warmup_ps=500.0):
    """Equilibrate bath with dihedral restrained at the barrier.

    Runs a warmup period first (to let phi relax to the restraint target),
    then extracts n_configs bath configurations spaced by interval_ps.

    Parameters
    ----------
    n_configs : int
        Number of bath configurations to extract.
    interval_ps : float
        Time interval between extracted configurations (ps).
    warmup_ps : float
        Warm-up period to discard before extracting configs (ps).
        The dihedral needs time to relax from its starting value to
        the restraint target; configs during this transient are not
        equilibrated.
    """
    equil_dir = os.path.join(BASE_DIR, 'equil')
    os.makedirs(equil_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(equil_dir)
                       if f.startswith('config_') and f.endswith('.gro')])
    if len(existing) >= n_configs:
        print(f"Already have {len(existing)} bath configurations in {equil_dir}")
        return [os.path.join(equil_dir, f) for f in existing[:n_configs]]

    total_ps = warmup_ps + n_configs * interval_ps
    nsteps = int(total_ps / DT)
    interval_steps = int(interval_ps / DT)

    print(f"Phase A: Generating {n_configs} bath configurations at barrier...")
    print(f"  Barrier position: {np.degrees(PHI_BARRIER_RAD):.1f} deg")
    print(f"  Warm-up: {warmup_ps:.0f} ps (discarded)")
    print(f"  Production: {n_configs * interval_ps:.0f} ps")
    print(f"  Total simulation: {total_ps:.0f} ps ({nsteps} steps)")
    print(f"  Snapshot interval: {interval_ps:.0f} ps")

    mdp = os.path.join(equil_dir, 'equil.mdp')
    write_mdp(mdp, nsteps, gen_vel=True, gen_seed=42)
    # Override: don't record every step for equilibration
    with open(mdp, 'r') as f:
        content = f.read()
    content = content.replace('nstxout=1', f'nstxout={interval_steps}')
    content = content.replace('nstvout=1', 'nstvout=0')
    with open(mdp, 'w') as f:
        f.write(content)

    plumed = os.path.join(equil_dir, 'plumed.dat')
    write_plumed_restraint(plumed, at_rad=PHI_BARRIER_RAD)

    trr, tpr = run_gromacs(GRO_START, mdp, plumed, equil_dir)
    if trr is None:
        print("ERROR: equilibration failed!")
        sys.exit(1)

    # Extract configurations using gmx trjconv, skipping warmup period
    print(f"Extracting bath configurations (skipping first {warmup_ps:.0f} ps)...")
    cmd = (f'echo "0" | {GMX} trjconv -s {tpr} -f {trr} '
           f'-b {warmup_ps:.1f} '
           f'-o {equil_dir}/config_.gro -sep -pbc whole')
    subprocess.run(cmd, shell=True, capture_output=True, cwd=equil_dir)

    configs = sorted([os.path.join(equil_dir, f)
                      for f in os.listdir(equil_dir)
                      if f.startswith('config_') and f.endswith('.gro')])
    print(f"  Extracted {len(configs)} configurations")

    # Clean up large TRR
    if os.path.exists(trr):
        os.remove(trr)

    return configs[:n_configs]


# ── Phase B: Shooting ──────────────────────────────────────────────────

def run_single_shoot(config_gro, seed, nsteps=5000):
    """Run one shooting trajectory from a bath configuration.

    Returns phi, phidot, ddot_fd arrays, or None on failure.
    """
    shoot_dir = os.path.join(BASE_DIR, f'shoot_seed{seed}')
    npz_path = os.path.join(shoot_dir, 'phi_data.npz')

    if os.path.exists(npz_path):
        return npz_path

    os.makedirs(shoot_dir, exist_ok=True)

    mdp = os.path.join(shoot_dir, 'shoot.mdp')
    write_mdp(mdp, nsteps, gen_vel=True, gen_seed=seed)

    plumed = os.path.join(shoot_dir, 'plumed.dat')
    write_plumed_free(plumed)

    trr, tpr = run_gromacs(config_gro, mdp, plumed, shoot_dir)
    if trr is None:
        print(f"  seed {seed}: GROMACS failed!")
        return None

    # Process trajectory
    phi, phidot, ddot_fd = process_trajectory_simple(trr, tpr, DT)

    # Save
    np.savez(npz_path, phi=phi, phidot=phidot, ddot=ddot_fd,
             dt=DT, seed=seed, nframes=len(phi),
             phi_barrier=PHI_BARRIER_RAD)

    # Clean up TRR (large)
    os.remove(trr)
    # Keep TPR for reference
    for f in ['run.edr', 'run.log', 'COLVAR', 'mdout.mdp']:
        p = os.path.join(shoot_dir, f)
        if os.path.exists(p):
            os.remove(p)

    print(f"  seed {seed}: phi(0)={np.degrees(phi[0]):.1f}°, "
          f"phidot(0)={phidot[0]:.1f} rad/ps, {len(phi)} frames",
          flush=True)
    return npz_path


def run_shooting(nshoot=200, seed_start=1, n_configs=50, warmup_ps=500.0):
    """Run N shooting trajectories distributed across bath configurations."""
    configs = equilibrate_bath(n_configs=n_configs, warmup_ps=warmup_ps)

    print(f"\nPhase B: Launching {nshoot} shooting trajectories...")
    print(f"  Using {len(configs)} bath configurations")
    print(f"  10 ps each ({int(10/DT)} steps), dt = {DT} ps")

    nsteps = int(10 / DT)  # 10 ps
    t_start = time.time()
    n_done = 0

    for i in range(nshoot):
        seed = seed_start + i
        config = configs[i % len(configs)]
        result = run_single_shoot(config, seed, nsteps)
        if result:
            n_done += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (nshoot - i - 1) / rate
            print(f"  {i+1}/{nshoot} done ({rate:.1f}/s, "
                  f"ETA {eta/60:.0f} min)", flush=True)

    print(f"\nDone: {n_done}/{nshoot} trajectories completed")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--equilibrate', action='store_true')
    parser.add_argument('--shoot', action='store_true')
    parser.add_argument('--nshoot', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=1)
    parser.add_argument('--n-configs', type=int, default=50)
    parser.add_argument('--warmup-ps', type=float, default=500.0,
                        help='Warm-up period to discard (ps, default 500)')
    args = parser.parse_args()

    if not args.equilibrate and not args.shoot:
        args.equilibrate = True
        args.shoot = True

    if args.equilibrate:
        equilibrate_bath(n_configs=args.n_configs, warmup_ps=args.warmup_ps)

    if args.shoot:
        run_shooting(nshoot=args.nshoot, seed_start=args.seed_start,
                     n_configs=args.n_configs, warmup_ps=args.warmup_ps)
