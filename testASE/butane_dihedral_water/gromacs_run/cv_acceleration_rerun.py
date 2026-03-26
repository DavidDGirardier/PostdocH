#!/usr/bin/env python3
"""
Compute dihedral CV acceleration decomposition along trajectories.

  ddot(phi) = v^T.H.v  +  J.F_intra/m  +  J.F_water/m  +  J.F_constr/m

Runs N independent constrained MD trajectories (different seeds),
each of length t_total, with full output every step.
For each trajectory: extract butane, one GROMACS rerun for F_intra,
constraint forces from FD of velocities.

Usage: python3 cv_acceleration_rerun.py [gro_file] [dt] [t_total_ps] [nruns]
  defaults: npt.gro, 0.002, 100.0, 5
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd, jacrev
import numpy as np
import struct
import subprocess
import os
import sys
import tempfile
import shutil
import time

MASSES = np.array([15.035, 14.027, 14.027, 15.035])


# ── JAX dihedral ──────────────────────────────────────────────────────────

def compute_phi_rad(p1, p2, p3, p4):
    """Dihedral angle in radians, JAX-differentiable."""
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = jnp.cross(b1, b2); n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    m1 = jnp.cross(n1, b2 / jnp.linalg.norm(b2))
    angle = jnp.arctan2(jnp.dot(m1, n2), jnp.dot(n1, n2))
    return jnp.mod(angle, 2 * jnp.pi)


def phi_numpy(pos):
    """Dihedral from (4,3) numpy array."""
    b1 = pos[1]-pos[0]; b2 = pos[2]-pos[1]; b3 = pos[3]-pos[2]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    angle = np.arctan2(np.dot(m1, n2), np.dot(n1, n2))
    return angle % (2 * np.pi)


def get_jacobian_hessian(pos):
    """J: (4,3), H: (4,3,4,3) of dihedral w.r.t. positions."""
    p = [jnp.array(pos[i]) for i in range(4)]
    J_t = jacfwd(compute_phi_rad, argnums=(0, 1, 2, 3))(*p)
    J = np.array([np.array(J_t[i]) for i in range(4)])
    H_t = jacfwd(jacrev(compute_phi_rad, argnums=(0, 1, 2, 3)),
                 argnums=(0, 1, 2, 3))(*p)
    H = np.zeros((4, 3, 4, 3))
    for i in range(4):
        for j in range(4):
            H[i, :, j, :] = np.array(H_t[i][j])
    return J, H


# ── TRR reader ────────────────────────────────────────────────────────────

def _parse_trr_header(data, offset):
    """Parse one TRR frame header. Returns (header_dict, new_offset) or None."""
    if offset + 8 > len(data):
        return None, offset
    magic = struct.unpack_from('>i', data, offset)[0]
    if magic != 1993:
        return None, offset
    offset += 4
    slen = struct.unpack_from('>i', data, offset)[0]
    offset += 4
    offset += ((slen + 3) // 4) * 4
    hdr = struct.unpack_from('>13i', data, offset)
    offset += 13 * 4
    ir_size, e_size, box_size, vir_size, pres_size = hdr[0:5]
    top_size, sym_size, x_size, v_size, f_size = hdr[5:10]
    natoms, step, nre = hdr[10:13]
    use_double = False
    if x_size > 0:
        use_double = (x_size == natoms * 3 * 8)
    elif v_size > 0:
        use_double = (v_size == natoms * 3 * 8)
    elif f_size > 0:
        use_double = (f_size == natoms * 3 * 8)
    float_size = 8 if use_double else 4
    offset += 2 * float_size  # t and lambda
    return {
        'ir_size': ir_size, 'e_size': e_size, 'box_size': box_size,
        'vir_size': vir_size, 'pres_size': pres_size,
        'top_size': top_size, 'sym_size': sym_size,
        'x_size': x_size, 'v_size': v_size, 'f_size': f_size,
        'natoms': natoms, 'use_double': use_double,
    }, offset


def read_trr(filename):
    """Read .trr (XDR) → (xyz, vel, frc) as float64 arrays. For small files."""
    frames_xyz, frames_vel, frames_frc = [], [], []
    with open(filename, 'rb') as f:
        data = f.read()
    offset = 0
    while offset < len(data):
        hdr, offset = _parse_trr_header(data, offset)
        if hdr is None:
            break
        for sz in [hdr['ir_size'], hdr['e_size'], hdr['box_size'],
                   hdr['vir_size'], hdr['pres_size'], hdr['top_size'], hdr['sym_size']]:
            offset += sz
        natoms = hdr['natoms']
        use_double = hdr['use_double']
        for store, sz in [(frames_xyz, hdr['x_size']),
                          (frames_vel, hdr['v_size']),
                          (frames_frc, hdr['f_size'])]:
            if sz > 0:
                n = natoms * 3
                fmt = f'>{n}{"d" if use_double else "f"}'
                vals = struct.unpack_from(fmt, data, offset)
                store.append(np.array(vals, dtype=np.float64).reshape(natoms, 3))
                offset += sz
            else:
                store.append(None)
    xyz = np.array(frames_xyz) if frames_xyz and frames_xyz[0] is not None else None
    vel = np.array(frames_vel) if frames_vel and frames_vel[0] is not None else None
    frc = np.array(frames_frc) if frames_frc and frames_frc[0] is not None else None
    return xyz, vel, frc


def read_trr_forces_subset(filename, atom_indices):
    """
    Stream-read a large .trr from disk, extracting only forces for selected atoms.
    Returns forces array (nframes, len(atom_indices), 3) as float64.
    Never loads the full file into memory — seeks through it frame by frame.
    """
    idx = np.array(atom_indices)
    frames_frc = []

    with open(filename, 'rb') as fh:
        while True:
            # Read magic
            buf = fh.read(4)
            if len(buf) < 4:
                break
            magic = struct.unpack('>i', buf)[0]
            if magic != 1993:
                break
            # String length + padded string
            slen = struct.unpack('>i', fh.read(4))[0]
            fh.seek(((slen + 3) // 4) * 4, 1)
            # 13 ints
            hdr = struct.unpack('>13i', fh.read(13 * 4))
            ir_size, e_size, box_size, vir_size, pres_size = hdr[0:5]
            top_size, sym_size, x_size, v_size, f_size = hdr[5:10]
            natoms = hdr[10]
            use_double = False
            if x_size > 0:
                use_double = (x_size == natoms * 3 * 8)
            elif v_size > 0:
                use_double = (v_size == natoms * 3 * 8)
            elif f_size > 0:
                use_double = (f_size == natoms * 3 * 8)
            fsize = 8 if use_double else 4
            fchar = 'd' if use_double else 'f'
            # Skip t + lambda
            fh.seek(2 * fsize, 1)
            # Skip ir, e, box, vir, pres, top, sym
            fh.seek(ir_size + e_size + box_size + vir_size + pres_size +
                    top_size + sym_size, 1)
            # Skip x and v
            fh.seek(x_size + v_size, 1)
            # Read selected atoms from forces
            if f_size > 0:
                f_start = fh.tell()
                subset = np.zeros((len(idx), 3), dtype=np.float64)
                for ii, ai in enumerate(idx):
                    fh.seek(f_start + ai * 3 * fsize)
                    vals = struct.unpack(f'>3{fchar}', fh.read(3 * fsize))
                    subset[ii] = vals
                frames_frc.append(subset)
                # Seek to end of force block
                fh.seek(f_start + f_size)

    return np.array(frames_frc)


# ── GROMACS helpers ───────────────────────────────────────────────────────

def run_md(gro_path, full_top, ff_dir, gmx, workdir, dt=0.002, nsteps=100, seed=42):
    """Run constrained MD with full output every step. Returns (trr, tpr) paths."""
    ff_link = os.path.join(workdir, 'gromos53a6.ff')
    if not os.path.exists(ff_link):
        os.symlink(ff_dir, ff_link)

    mdp = os.path.join(workdir, 'run.mdp')
    with open(mdp, 'w') as f:
        f.write(f'integrator=md\ndt={dt}\nnsteps={nsteps}\n')
        f.write('nstxout=1\nnstvout=1\nnstfout=1\nnstenergy=1\n')
        f.write('cutoff-scheme=Verlet\nnstlist=10\ncoulombtype=PME\n')
        f.write('rcoulomb=0.9\nrvdw=0.9\npbc=xyz\n')
        f.write('tcoupl=v-rescale\ntc-grps=System\ntau-t=0.1\nref-t=300\n')
        f.write('constraints=all-angles\nlincs-order=6\n')
        f.write(f'gen-vel=yes\ngen-temp=300\ngen-seed={seed}\n')

    tpr = os.path.join(workdir, 'run.tpr')
    trr = os.path.join(workdir, 'run.trr')

    cmd = f'{gmx} grompp -f {mdp} -c {gro_path} -p {full_top} -o {tpr} -maxwarn 10'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(tpr):
        print(f"  grompp failed: {r.stderr[-300:]}")
        return None, None

    cmd = f'{gmx} mdrun -s {tpr} -o {trr} -ntomp 1'
    subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)

    return trr, tpr


def extract_butane_trr(full_trr, full_tpr, gmx, workdir):
    """Extract butane-only trajectory from full-system trr."""
    ndx = os.path.join(workdir, 'butane.ndx')
    with open(ndx, 'w') as f:
        f.write('[ butane ]\n1 2 3 4\n')

    out_trr = os.path.join(workdir, 'butane.trr')
    cmd = f'echo "0" | {gmx} trjconv -f {full_trr} -s {full_tpr} -o {out_trr} -n {ndx}'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(out_trr):
        print(f"  trjconv failed: {r.stderr[-300:]}")
        return None
    return out_trr


def prepare_vacuum_tpr(butane_trr, ff_dir, gmx, workdir):
    """Create vacuum TPR for rerun. Returns tpr path."""
    top = os.path.join(workdir, 'vacuum.top')
    with open(top, 'w') as f:
        f.write(f'#include "{ff_dir}/forcefield.itp"\n\n')
        f.write('[ moleculetype ]\nBUT  3\n\n')
        f.write('[ atoms ]\n')
        f.write('  1  CH3  1  BUT  C1  1  0.000  15.035\n')
        f.write('  2  CH2  1  BUT  C2  2  0.000  14.027\n')
        f.write('  3  CH2  1  BUT  C3  3  0.000  14.027\n')
        f.write('  4  CH3  1  BUT  C4  4  0.000  15.035\n\n')
        f.write('[ bonds ]\n  1  2  2  gb_27\n  2  3  2  gb_27\n  3  4  2  gb_27\n\n')
        f.write('[ angles ]\n  1  2  3  2  ga_13\n  2  3  4  2  ga_13\n\n')
        f.write('[ dihedrals ]\n  1  2  3  4  1  gd_34\n\n')
        f.write('[ pairs ]\n  1  4  1\n\n')
        f.write('[ system ]\nbutane\n\n[ molecules ]\nBUT  1\n')

    # Write .gro from first frame of butane trr
    gro = os.path.join(workdir, 'butane_frame0.gro')
    xyz_b, _, _ = read_trr(butane_trr)
    pos0 = xyz_b[0]
    box = 5.0
    com = pos0.mean(axis=0)
    pos_c = pos0 - com + box / 2
    names = ['C1', 'C2', 'C3', 'C4']
    with open(gro, 'w') as f:
        f.write('butane\n4\n')
        for i in range(4):
            f.write(f'{1:5d}{"BUT":<5s}{names[i]:>5s}{i+1:5d}'
                    f'{pos_c[i][0]:8.3f}{pos_c[i][1]:8.3f}{pos_c[i][2]:8.3f}\n')
        f.write(f'{box:10.5f}{box:10.5f}{box:10.5f}\n')

    mdp = os.path.join(workdir, 'vacuum.mdp')
    with open(mdp, 'w') as f:
        f.write('integrator=md\nnsteps=0\nnstxout=1\nnstfout=1\nnstenergy=1\n')
        f.write('cutoff-scheme=Verlet\nnstlist=1\ncoulombtype=Cut-off\n')
        f.write('rcoulomb=1.0\nrvdw=1.0\npbc=xyz\n')
        f.write('constraints=none\ncontinuation=yes\ngen-vel=no\n')

    ff_link = os.path.join(workdir, 'gromos53a6.ff')
    if not os.path.exists(ff_link):
        os.symlink(ff_dir, ff_link)

    tpr = os.path.join(workdir, 'vacuum.tpr')
    cmd = f'{gmx} grompp -f {mdp} -c {gro} -p {top} -o {tpr} -maxwarn 10'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(tpr):
        print(f"  vacuum grompp failed: {r.stderr[-300:]}")
        return None
    return tpr


def rerun_vacuum(butane_trr, vacuum_tpr, gmx, workdir):
    """Rerun vacuum topology on butane trajectory → F_intra at all frames."""
    rerun_trr = os.path.join(workdir, 'rerun.trr')
    cmd = (f'{gmx} mdrun -s {vacuum_tpr} -rerun {butane_trr} '
           f'-o {rerun_trr} -ntomp 1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(rerun_trr):
        print(f"  rerun failed: {r.stderr[-300:]}")
        return None
    return rerun_trr


def process_trajectory(pos_all, vel_all, F_total, F_intra, dt):
    """
    Compute decomposition arrays for one trajectory.

    Returns dict with arrays (nframes,) for:
      phi, hess, intra, water, constr, ddot_phys, ddot_full, ddot_fd
    """
    nframes = pos_all.shape[0]
    F_water = F_total - F_intra

    phi_arr = np.zeros(nframes)
    Jv_arr = np.zeros(nframes)
    hess_arr = np.zeros(nframes)
    intra_arr = np.zeros(nframes)
    water_arr = np.zeros(nframes)
    constr_arr = np.zeros(nframes)
    J_all = np.zeros((nframes, 4, 3))

    for k in range(nframes):
        pos_k = pos_all[k]
        vel_k = vel_all[k]
        phi_arr[k] = phi_numpy(pos_k)

        J, H = get_jacobian_hessian(pos_k)
        J_all[k] = J
        Jv_arr[k] = sum(np.dot(J[i], vel_k[i]) for i in range(4))

        hess_arr[k] = sum(vel_k[i] @ H[i, :, j, :] @ vel_k[j]
                          for i in range(4) for j in range(4))
        intra_arr[k] = sum(np.dot(F_intra[k, i] / MASSES[i], J[i]) for i in range(4))
        water_arr[k] = sum(np.dot(F_water[k, i] / MASSES[i], J[i]) for i in range(4))

    ddot_phys_arr = hess_arr + intra_arr + water_arr

    # FD of J.v → true ddot(phi)
    ddot_fd_arr = np.zeros(nframes)
    for k in range(1, nframes - 1):
        ddot_fd_arr[k] = (Jv_arr[k+1] - Jv_arr[k-1]) / (2 * dt)

    # Constraint forces from FD of velocities
    for k in range(1, nframes - 1):
        a_total = (vel_all[k+1] - vel_all[k-1]) / (2 * dt)
        F_constr_k = MASSES[:, None] * a_total - F_total[k]
        constr_arr[k] = sum(np.dot(F_constr_k[i] / MASSES[i], J_all[k, i])
                            for i in range(4))

    ddot_full_arr = ddot_phys_arr + constr_arr

    return {
        'phi': phi_arr, 'hess': hess_arr, 'intra': intra_arr,
        'water': water_arr, 'constr': constr_arr,
        'ddot_phys': ddot_phys_arr, 'ddot_full': ddot_full_arr,
        'ddot_fd': ddot_fd_arr,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_single_seed(base_dir, gro_path, full_top, ff_dir, gmx, seed, dt, nsteps):
    """Run a single seed: MD + extract + rerun + decomposition. Saves to run_seed{seed}/."""
    tmpdir = os.path.join(base_dir, f'run_seed{seed}')
    os.makedirs(tmpdir, exist_ok=True)

    # Skip if already completed
    run_outfile = os.path.join(tmpdir, 'cv_accel_data.npz')
    if os.path.exists(run_outfile):
        print(f"  [seed={seed}] Already done, loading {run_outfile}")
        return run_outfile

    t_run = time.time()
    print(f"  [seed={seed}] MD ({nsteps} steps)...", flush=True)
    t0 = time.time()
    full_trr, full_tpr = run_md(gro_path, full_top, ff_dir, gmx,
                                 tmpdir, dt=dt, nsteps=nsteps, seed=seed)
    if full_trr is None:
        print(f"  [seed={seed}] MD FAILED"); return None
    print(f"  [seed={seed}] MD done in {time.time()-t0:.0f}s", flush=True)

    print(f"  [seed={seed}] Extract butane...", flush=True)
    t0 = time.time()
    butane_trr = extract_butane_trr(full_trr, full_tpr, gmx, tmpdir)
    if butane_trr is None:
        print(f"  [seed={seed}] Extract FAILED"); return None
    print(f"  [seed={seed}] Extract done in {time.time()-t0:.0f}s", flush=True)

    vacuum_tpr = prepare_vacuum_tpr(butane_trr, ff_dir, gmx, tmpdir)
    if vacuum_tpr is None:
        print(f"  [seed={seed}] Vacuum TPR FAILED"); return None

    print(f"  [seed={seed}] Vacuum rerun...", flush=True)
    t0 = time.time()
    rerun_trr_path = rerun_vacuum(butane_trr, vacuum_tpr, gmx, tmpdir)
    if rerun_trr_path is None:
        print(f"  [seed={seed}] Rerun FAILED"); return None
    print(f"  [seed={seed}] Rerun done in {time.time()-t0:.0f}s", flush=True)

    print(f"  [seed={seed}] Reading TRRs...", flush=True)
    t0 = time.time()
    xyz_but, vel_but, _ = read_trr(butane_trr)
    _, _, frc_vac = read_trr(rerun_trr_path)
    F_total = read_trr_forces_subset(full_trr, [0, 1, 2, 3])
    print(f"  [seed={seed}] TRRs read in {time.time()-t0:.0f}s", flush=True)

    nframes = xyz_but.shape[0]
    pos_all = xyz_but.astype(np.float64)
    vel_all = vel_but.astype(np.float64)
    F_intra = frc_vac[:, :, :].astype(np.float64)

    print(f"  [seed={seed}] JAX decomposition ({nframes} frames)...", flush=True)
    t0 = time.time()
    res = process_trajectory(pos_all, vel_all, F_total, F_intra, dt)
    print(f"  [seed={seed}] JAX done in {time.time()-t0:.0f}s", flush=True)

    np.savez(run_outfile, phi=res['phi'], hess=res['hess'],
             intra=res['intra'], water=res['water'],
             constr=res['constr'], ddot_fd=res['ddot_fd'],
             dt=dt, seed=seed)
    print(f"  [seed={seed}] Saved to {run_outfile}")
    print(f"  [seed={seed}] Total: {time.time()-t_run:.0f}s", flush=True)
    return run_outfile


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gro_file = sys.argv[1] if len(sys.argv) > 1 else 'npt.gro'
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    t_total = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0  # ps
    nruns = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    # Optional: --seed N to run a single seed only
    single_seed = None
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        single_seed = int(sys.argv[idx + 1])

    nsteps = int(t_total / dt)

    gro_path = os.path.join(base_dir, gro_file)
    if not os.path.exists(gro_path):
        gro_path = os.path.join(base_dir, 'em.gro')

    gmx = os.environ.get('GMX_CMD', 'gmx_mpi')
    ff_dir = os.path.join(base_dir, 'gromos53a6.ff')
    full_top = os.path.join(base_dir, 'topol.top')

    if '--seeds' in sys.argv:
        idx = sys.argv.index('--seeds')
        seeds = [int(x) for x in sys.argv[idx + 1].split(',')]
    else:
        seeds = [42, 137, 271, 314, 577][:nruns]

    if single_seed is not None:
        print(f"=== Single-seed mode: seed={single_seed}, dt={dt}, t={t_total} ps ===")
        run_single_seed(base_dir, gro_path, full_top, ff_dir, gmx,
                        single_seed, dt, nsteps)
        return

    print(f"=== CV acceleration decomposition ===")
    print(f"  dt = {dt} ps, t_total = {t_total} ps ({nsteps} steps)")
    print(f"  {nruns} independent runs, seeds = {seeds}")
    print(f"  constraints = all-angles")
    nframes_expected = nsteps + 1
    trr_size_gb = nframes_expected * 2170 * 3 * 4 * 3 / 1e9
    print(f"  ~{trr_size_gb:.1f} GB per TRR (processed and deleted)\n")

    # Collect interior-frame data from all runs
    all_hess, all_intra, all_water, all_constr = [], [], [], []
    all_ddot_fd, all_ddot_phys, all_ddot_full = [], [], []
    all_phi = []

    # Load per-run data (produced by parallel --seed runs)
    for seed in seeds:
        run_file = os.path.join(base_dir, f'run_seed{seed}', 'cv_accel_data.npz')
        if not os.path.exists(run_file):
            print(f"  seed={seed}: NOT FOUND — run with --seed {seed} first")
            continue
        print(f"  seed={seed}: loading {run_file}")
        rd = np.load(run_file)
        s = slice(1, len(rd['phi']) - 1)
        all_phi.append(rd['phi'][s])
        all_hess.append(rd['hess'][s])
        all_intra.append(rd['intra'][s])
        all_water.append(rd['water'][s])
        all_constr.append(rd['constr'][s])
        all_ddot_fd.append(rd['ddot_fd'][s])
        all_ddot_phys.append((rd['hess'] + rd['intra'] + rd['water'])[s])
        all_ddot_full.append((rd['hess'] + rd['intra'] + rd['water'] + rd['constr'])[s])

    if not all_hess:
        print("No successful runs.")
        return

    # ── Aggregate results ─────────────────────────────────────────────────
    hess = np.concatenate(all_hess)
    intra = np.concatenate(all_intra)
    water = np.concatenate(all_water)
    constr = np.concatenate(all_constr)
    ddot_fd = np.concatenate(all_ddot_fd)
    ddot_phys = np.concatenate(all_ddot_phys)
    ddot_full = np.concatenate(all_ddot_full)
    phi = np.concatenate(all_phi)
    ntot = len(hess)

    print("=" * 80)
    print(f"AGGREGATE RESULTS ({nruns} runs, {ntot} interior frames)")
    print("=" * 80)

    # Closure check
    err_phys = np.abs(ddot_phys - ddot_fd)
    err_full = np.abs(ddot_full - ddot_fd)
    mean_mag = np.mean(np.abs(ddot_fd))

    print(f"\nClosure check:")
    print(f"  mean|ddot_FD|                      = {mean_mag:.2f} rad/ps^2")
    print(f"  mean|hess+intra+water - FD|         = {np.mean(err_phys):.4f}  "
          f"({np.mean(err_phys)/mean_mag*100:.2f}%)")
    print(f"  mean|hess+intra+water+constr - FD|  = {np.mean(err_full):.4f}  "
          f"({np.mean(err_full)/mean_mag*100:.2f}%)")

    # Mean contributions
    print(f"\nMean contributions:")
    print(f"  Hessian (kinematic) : {np.mean(hess):10.2f} rad/ps^2")
    print(f"  Intramolecular     : {np.mean(intra):10.2f} rad/ps^2")
    print(f"  Water              : {np.mean(water):10.2f} rad/ps^2")
    print(f"  Constraint         : {np.mean(constr):10.2f} rad/ps^2")
    print(f"  Total (FD)         : {np.mean(ddot_fd):10.2f} rad/ps^2")

    # RMS contributions
    print(f"\nRMS contributions:")
    print(f"  Hessian (kinematic) : {np.sqrt(np.mean(hess**2)):10.2f} rad/ps^2")
    print(f"  Intramolecular     : {np.sqrt(np.mean(intra**2)):10.2f} rad/ps^2")
    print(f"  Water              : {np.sqrt(np.mean(water**2)):10.2f} rad/ps^2")
    print(f"  Constraint         : {np.sqrt(np.mean(constr**2)):10.2f} rad/ps^2")
    print(f"  Total (FD)         : {np.sqrt(np.mean(ddot_fd**2)):10.2f} rad/ps^2")

    # Per-run summary
    print(f"\nPer-run RMS of total (FD):")
    for i, arr in enumerate(all_ddot_fd):
        print(f"  Run {i+1} (seed={seeds[i]}): {np.sqrt(np.mean(arr**2)):8.2f} rad/ps^2  "
              f"({len(arr)} frames)")

    # Save raw data for further analysis
    outfile = os.path.join(base_dir, 'cv_accel_data.npz')
    np.savez(outfile, phi=phi, hess=hess, intra=intra, water=water,
             constr=constr, ddot_fd=ddot_fd, dt=dt, seeds=seeds,
             nruns=nruns, t_total=t_total)
    print(f"\nRaw data saved to {outfile}")


if __name__ == '__main__':
    main()
