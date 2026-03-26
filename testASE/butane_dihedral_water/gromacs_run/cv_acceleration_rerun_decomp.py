#!/usr/bin/env python3
"""
Compute dihedral CV acceleration decomposition with close/far water split.

  ddot(phi) = v^T.H.v + J.F_intra/m + J.F_close/m + J.F_far/m
                       + J.F_PME_corr/m + J.F_constr/m

Close water:  molecules whose oxygen is within Rcut of at least one solute atom.
Far water:    all other water molecules.
PME correction: long-range electrostatic contribution not captured by cutoff.

Force decomposition of the water contribution:
  1. Full MD (PME)              → F_total_PME  on solute
  2. Vacuum rerun (cutoff)      → F_intra      on solute
  3. Full-system rerun (cutoff) → F_total_cut  on solute   [no PME]
  4. Python pair forces (cutoff) → F_close, F_far           [LJ + Coulomb]

  F_close + F_far  = F_total_cut − F_intra   (cutoff forces are pairwise additive)
  F_PME_corr       = F_total_PME − F_total_cut
  F_water_total    = F_close + F_far + F_PME_corr = F_total_PME − F_intra

Works for charged solutes: the non-decomposable PME reciprocal-space
contribution is tracked as a separate correction term.

Usage:
  python3 cv_acceleration_rerun_decomp.py [gro_file] [dt] [t_total_ps] [nruns] \
        [--seed N] [--Rcut R]
  defaults: npt.gro, 0.002, 100.0, 5, Rcut=0.5 nm
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
import time

MASSES = np.array([15.035, 14.027, 14.027, 15.035])

# ── System-specific parameters ────────────────────────────────────────────
# Modify this section for a different solute / solvent / force field.
#
# Solute: butane united-atom [CH3, CH2, CH2, CH3] in GROMOS53a6
N_SOLUTE = 4
SOLUTE_CHARGES = np.array([0.0, 0.0, 0.0, 0.0])  # e

# Solvent: SPC/E water  [OW, HW1, HW2]
ATOMS_PER_SOLVENT = 3
SOLVENT_CHARGES = np.array([-0.8476, 0.4238, 0.4238])  # e

# LJ parameters C6, C12 for each (solute atom, solvent atom) pair.
# From ffnonbonded.itp [ nonbond_params ], func type 1: V = C12/r^12 − C6/r^6
# Units: C6 [kJ/mol·nm^6],  C12 [kJ/mol·nm^12]
# H atoms have zero LJ in GROMOS53a6.
C6_PAIR = np.array([
    # OW            HW             HW        ← solvent atom
    [0.005016238,   0.0,           0.0],     # CH3
    [0.0044212472,  0.0,           0.0],     # CH2
    [0.0044212472,  0.0,           0.0],     # CH2
    [0.005016238,   0.0,           0.0],     # CH3
])  # shape (N_SOLUTE, ATOMS_PER_SOLVENT)

C12_PAIR = np.array([
    [8.377926e-06,  0.0,           0.0],
    [9.458844e-06,  0.0,           0.0],
    [9.458844e-06,  0.0,           0.0],
    [8.377926e-06,  0.0,           0.0],
])

# Cutoffs — must match the MD .mdp
RVDW     = 0.9   # nm
RCOULOMB = 0.9   # nm

# Coulomb constant in GROMACS units: kJ·nm/(mol·e²)
COULOMB_CONST = 138.935458

# Pre-compute flags to skip unnecessary work
_HAS_COULOMB = np.any(SOLUTE_CHARGES != 0)
_HAS_LJ = np.array([[C6_PAIR[i, a] != 0 or C12_PAIR[i, a] != 0
                      for a in range(ATOMS_PER_SOLVENT)]
                     for i in range(N_SOLUTE)])  # (N_SOLUTE, ATOMS_PER_SOLVENT)


# ── JAX dihedral ──────────────────────────────────────────────────────────

def compute_phi_rad(p1, p2, p3, p4):
    """Dihedral angle in radians, JAX-differentiable."""
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = jnp.cross(b1, b2); n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    m1 = jnp.cross(n1, b2 / jnp.linalg.norm(b2))
    return jnp.arctan2(jnp.dot(m1, n2), jnp.dot(n1, n2))


def phi_numpy(pos):
    """Dihedral from (4,3) numpy array."""
    b1 = pos[1]-pos[0]; b2 = pos[2]-pos[1]; b3 = pos[3]-pos[2]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1); n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))


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


# ── TRR readers ───────────────────────────────────────────────────────────

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
    offset += 2 * float_size
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


def stream_trr_for_decomposition(filename):
    """
    Stream a full-system TRR, yielding per frame:
        (box, positions, solute_forces)

    box:            (3, 3) array or None
    positions:      (natoms, 3) array or None — all atoms
    solute_forces:  (N_SOLUTE, 3) array or None — forces on solute only

    Velocities are skipped for efficiency.
    """
    with open(filename, 'rb') as fh:
        while True:
            buf = fh.read(4)
            if len(buf) < 4:
                break
            magic = struct.unpack('>i', buf)[0]
            if magic != 1993:
                break

            slen = struct.unpack('>i', fh.read(4))[0]
            fh.seek(((slen + 3) // 4) * 4, 1)

            hdr = struct.unpack('>13i', fh.read(13 * 4))
            ir_size, e_size, box_size = hdr[0], hdr[1], hdr[2]
            vir_size, pres_size      = hdr[3], hdr[4]
            top_size, sym_size       = hdr[5], hdr[6]
            x_size, v_size, f_size   = hdr[7], hdr[8], hdr[9]
            natoms = hdr[10]

            use_double = False
            if x_size > 0:
                use_double = (x_size == natoms * 3 * 8)
            elif v_size > 0:
                use_double = (v_size == natoms * 3 * 8)
            elif f_size > 0:
                use_double = (f_size == natoms * 3 * 8)
            fsize = 8 if use_double else 4
            dt_np = '>f8' if use_double else '>f4'

            fh.seek(2 * fsize, 1)            # t + lambda
            fh.seek(ir_size + e_size, 1)     # ir, e

            # Box
            box = None
            if box_size > 0:
                raw = fh.read(box_size)
                box = np.frombuffer(raw, dtype=dt_np).astype(np.float64).reshape(3, 3)

            fh.seek(vir_size + pres_size + top_size + sym_size, 1)

            # Positions (all atoms)
            positions = None
            if x_size > 0:
                raw = fh.read(x_size)
                positions = np.frombuffer(
                    raw, dtype=dt_np).astype(np.float64).reshape(natoms, 3)

            # Skip velocities
            if v_size > 0:
                fh.seek(v_size, 1)

            # Forces — only first N_SOLUTE atoms
            solute_forces = None
            if f_size > 0:
                f_start = fh.tell()
                nbytes = N_SOLUTE * 3 * fsize
                raw = fh.read(nbytes)
                solute_forces = np.frombuffer(
                    raw, dtype=dt_np).astype(np.float64).reshape(N_SOLUTE, 3)
                fh.seek(f_start + f_size)

            yield box, positions, solute_forces


def read_solute_forces_stream(filename):
    """Stream TRR and return only solute forces (nframes, N_SOLUTE, 3)."""
    F_list = []
    for _, _, forces in stream_trr_for_decomposition(filename):
        if forces is not None:
            F_list.append(forces.copy())
    return np.array(F_list)


# ── Close / far water force decomposition ─────────────────────────────────

def compute_pair_forces_decomposed(full_trr, Rcut):
    """
    Single pass through the full-system TRR.

    For each frame:
      1. Extract F_total_PME on solute (from TRR forces).
      2. Classify each solvent molecule as *close* (O within Rcut of any
         solute atom) or *far*.
      3. Compute LJ + Coulomb pair forces on solute from close and far
         solvent atoms using cutoff (no PME).

    Returns
    -------
    F_total_PME : (nframes, N_SOLUTE, 3)
    F_close     : (nframes, N_SOLUTE, 3)
    F_far       : (nframes, N_SOLUTE, 3)
    n_close     : (nframes,)
    """
    F_total_list = []
    F_close_list = []
    F_far_list = []
    n_close_list = []

    rvdw2 = RVDW ** 2
    rcoul2 = RCOULOMB ** 2
    Rcut2 = Rcut ** 2

    n_solvent_mol = None
    frame_count = 0

    for box, positions, solute_forces in stream_trr_for_decomposition(full_trr):
        if positions is None or solute_forces is None:
            continue

        # Detect system size from first frame
        if n_solvent_mol is None:
            natoms = positions.shape[0]
            n_solvent_atoms = natoms - N_SOLUTE
            assert n_solvent_atoms % ATOMS_PER_SOLVENT == 0, (
                f"Non-integer solvent count: {n_solvent_atoms} / {ATOMS_PER_SOLVENT}")
            n_solvent_mol = n_solvent_atoms // ATOMS_PER_SOLVENT
            print(f"    System: {natoms} atoms, {n_solvent_mol} solvent molecules")

        F_total_list.append(solute_forces.copy())

        pos_solute = positions[:N_SOLUTE]                                # (Ns, 3)
        pos_solvent = positions[N_SOLUTE:].reshape(n_solvent_mol,
                                                    ATOMS_PER_SOLVENT, 3)  # (Nw, Ka, 3)
        box_diag = np.diag(box)

        # ── Classify: solvent O (atom 0) within Rcut of ANY solute atom ──
        pos_ow = pos_solvent[:, 0, :]                                    # (Nw, 3)
        dr_class = pos_solute[:, None, :] - pos_ow[None, :, :]          # (Ns, Nw, 3)
        dr_class -= box_diag * np.round(dr_class / box_diag)
        r2_class = np.sum(dr_class ** 2, axis=2)                         # (Ns, Nw)
        close_mask = np.any(r2_class < Rcut2, axis=0)                    # (Nw,)
        n_close_list.append(int(np.sum(close_mask)))

        # ── Pair forces: LJ + Coulomb on solute from each solvent atom ──
        F_close_frame = np.zeros((N_SOLUTE, 3))
        F_far_frame = np.zeros((N_SOLUTE, 3))

        for a in range(ATOMS_PER_SOLVENT):
            # Check if this solvent atom type contributes anything
            has_lj_a = np.any(_HAS_LJ[:, a])
            has_coul_a = _HAS_COULOMB and SOLVENT_CHARGES[a] != 0
            if not has_lj_a and not has_coul_a:
                continue

            pos_a = pos_solvent[:, a, :]                                 # (Nw, 3)
            dr = pos_solute[:, None, :] - pos_a[None, :, :]             # (Ns, Nw, 3)
            dr -= box_diag * np.round(dr / box_diag)
            r2 = np.sum(dr ** 2, axis=2)                                 # (Ns, Nw)

            fmag = np.zeros_like(r2)

            # LJ: (12 C12/r^14 − 6 C6/r^8)
            if has_lj_a:
                within_lj = r2 < rvdw2
                r2_safe = np.where(within_lj, r2, 1.0)
                r2inv = 1.0 / r2_safe
                r6inv = r2inv ** 3
                fmag_lj = (12.0 * C12_PAIR[:, a, None] * r6inv * r6inv * r2inv
                           - 6.0 * C6_PAIR[:, a, None] * r6inv * r2inv)
                fmag += np.where(within_lj, fmag_lj, 0.0)

            # Coulomb: f * qi * qj / r^3
            if has_coul_a:
                within_coul = r2 < rcoul2
                r2_safe_c = np.where(within_coul, r2, 1.0)
                r_safe = np.sqrt(r2_safe_c)
                fmag_coul = (COULOMB_CONST * SOLUTE_CHARGES[:, None]
                             * SOLVENT_CHARGES[a] / (r2_safe_c * r_safe))
                fmag += np.where(within_coul, fmag_coul, 0.0)

            F_per = fmag[:, :, None] * dr                                # (Ns, Nw, 3)
            F_close_frame += np.sum(F_per[:, close_mask, :], axis=1)
            F_far_frame += np.sum(F_per[:, ~close_mask, :], axis=1)

        F_close_list.append(F_close_frame)
        F_far_list.append(F_far_frame)

        frame_count += 1
        if frame_count % 10000 == 0:
            print(f"    ... {frame_count} frames processed", flush=True)

    print(f"    Total: {frame_count} frames", flush=True)
    return (np.array(F_total_list), np.array(F_close_list),
            np.array(F_far_list), np.array(n_close_list))


# ── GROMACS helpers ───────────────────────────────────────────────────────

def run_md(gro_path, full_top, ff_dir, gmx, workdir, dt=0.002, nsteps=100, seed=42):
    """Run constrained MD with PME and full output every step."""
    ff_link = os.path.join(workdir, 'gromos53a6.ff')
    if not os.path.exists(ff_link):
        os.symlink(ff_dir, ff_link)

    mdp = os.path.join(workdir, 'run.mdp')
    with open(mdp, 'w') as f:
        f.write(f'integrator=md\ndt={dt}\nnsteps={nsteps}\n')
        f.write('nstxout=1\nnstvout=1\nnstfout=1\nnstenergy=1\n')
        f.write('cutoff-scheme=Verlet\nnstlist=10\ncoulombtype=PME\n')
        f.write(f'rcoulomb={RCOULOMB}\nrvdw={RVDW}\npbc=xyz\n')
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
    """Extract solute-only trajectory from full-system trr."""
    ndx = os.path.join(workdir, 'butane.ndx')
    with open(ndx, 'w') as f:
        f.write('[ butane ]\n')
        f.write(' '.join(str(i+1) for i in range(N_SOLUTE)) + '\n')

    out_trr = os.path.join(workdir, 'butane.trr')
    cmd = f'echo "0" | {gmx} trjconv -f {full_trr} -s {full_tpr} -o {out_trr} -n {ndx}'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(out_trr):
        print(f"  trjconv failed: {r.stderr[-300:]}")
        return None
    return out_trr


def prepare_vacuum_tpr(butane_trr, ff_dir, gmx, workdir):
    """Create vacuum TPR for rerun (cutoff, no PME). Returns tpr path."""
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
    """Rerun vacuum topology on solute trajectory → F_intra at all frames."""
    rerun_trr = os.path.join(workdir, 'rerun.trr')
    cmd = (f'{gmx} mdrun -s {vacuum_tpr} -rerun {butane_trr} '
           f'-o {rerun_trr} -ntomp 1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(rerun_trr):
        print(f"  rerun failed: {r.stderr[-300:]}")
        return None
    return rerun_trr


def prepare_cutoff_full_tpr(gro_path, full_top, ff_dir, gmx, workdir):
    """Create TPR for full-system rerun with cutoff electrostatics (no PME)."""
    ff_link = os.path.join(workdir, 'gromos53a6.ff')
    if not os.path.exists(ff_link):
        os.symlink(ff_dir, ff_link)

    mdp = os.path.join(workdir, 'cutoff_full.mdp')
    with open(mdp, 'w') as f:
        f.write('integrator=md\nnsteps=0\n')
        f.write('nstxout=0\nnstvout=0\nnstfout=1\nnstenergy=1\n')
        f.write('cutoff-scheme=Verlet\nnstlist=1\n')
        f.write(f'coulombtype=Cut-off\nrcoulomb={RCOULOMB}\n')
        f.write(f'rvdw={RVDW}\npbc=xyz\n')
        f.write('constraints=none\ncontinuation=yes\ngen-vel=no\n')

    tpr = os.path.join(workdir, 'cutoff_full.tpr')
    cmd = f'{gmx} grompp -f {mdp} -c {gro_path} -p {full_top} -o {tpr} -maxwarn 10'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(tpr):
        print(f"  cutoff grompp failed: {r.stderr[-300:]}")
        return None
    return tpr


def rerun_cutoff_full(full_trr, cutoff_tpr, gmx, workdir):
    """Rerun full system with cutoff → F_total_cutoff at all frames."""
    rerun_trr = os.path.join(workdir, 'cutoff_rerun.trr')
    cmd = (f'{gmx} mdrun -s {cutoff_tpr} -rerun {full_trr} '
           f'-o {rerun_trr} -ntomp 1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workdir)
    if not os.path.exists(rerun_trr):
        print(f"  cutoff rerun failed: {r.stderr[-300:]}")
        return None
    return rerun_trr


# ── Trajectory processing ─────────────────────────────────────────────────

def process_trajectory(pos_all, vel_all, F_total_PME, F_intra,
                       F_close, F_far, F_PME_corr, dt):
    """
    Compute decomposition arrays for one trajectory.

    Returns dict with arrays (nframes,) for:
      phi, hess, intra, water, close, far, pme_corr,
      constr, ddot_phys, ddot_full, ddot_fd
    """
    nframes = pos_all.shape[0]
    F_water = F_total_PME - F_intra

    phi_arr      = np.zeros(nframes)
    Jv_arr       = np.zeros(nframes)
    hess_arr     = np.zeros(nframes)
    intra_arr    = np.zeros(nframes)
    water_arr    = np.zeros(nframes)
    close_arr    = np.zeros(nframes)
    far_arr      = np.zeros(nframes)
    pme_corr_arr = np.zeros(nframes)
    constr_arr   = np.zeros(nframes)
    J_all        = np.zeros((nframes, 4, 3))

    for k in range(nframes):
        pos_k = pos_all[k]
        vel_k = vel_all[k]
        phi_arr[k] = phi_numpy(pos_k)

        J, H = get_jacobian_hessian(pos_k)
        J_all[k] = J
        Jv_arr[k] = sum(np.dot(J[i], vel_k[i]) for i in range(4))

        hess_arr[k]     = sum(vel_k[i] @ H[i, :, j, :] @ vel_k[j]
                              for i in range(4) for j in range(4))
        intra_arr[k]    = sum(np.dot(F_intra[k, i]    / MASSES[i], J[i]) for i in range(4))
        water_arr[k]    = sum(np.dot(F_water[k, i]    / MASSES[i], J[i]) for i in range(4))
        close_arr[k]    = sum(np.dot(F_close[k, i]    / MASSES[i], J[i]) for i in range(4))
        far_arr[k]      = sum(np.dot(F_far[k, i]      / MASSES[i], J[i]) for i in range(4))
        pme_corr_arr[k] = sum(np.dot(F_PME_corr[k, i] / MASSES[i], J[i]) for i in range(4))

    ddot_phys_arr = hess_arr + intra_arr + water_arr

    # FD of J·v → true d²φ/dt²
    ddot_fd_arr = np.zeros(nframes)
    for k in range(1, nframes - 1):
        ddot_fd_arr[k] = (Jv_arr[k+1] - Jv_arr[k-1]) / (2 * dt)

    # Constraint forces from FD of velocities
    for k in range(1, nframes - 1):
        a_total = (vel_all[k+1] - vel_all[k-1]) / (2 * dt)
        F_constr_k = MASSES[:, None] * a_total - F_total_PME[k]
        constr_arr[k] = sum(np.dot(F_constr_k[i] / MASSES[i], J_all[k, i])
                            for i in range(4))

    ddot_full_arr = ddot_phys_arr + constr_arr

    return {
        'phi': phi_arr, 'hess': hess_arr, 'intra': intra_arr,
        'water': water_arr, 'close': close_arr, 'far': far_arr,
        'pme_corr': pme_corr_arr, 'constr': constr_arr,
        'ddot_phys': ddot_phys_arr, 'ddot_full': ddot_full_arr,
        'ddot_fd': ddot_fd_arr,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_single_seed(base_dir, gro_path, full_top, ff_dir, gmx,
                    seed, dt, nsteps, Rcut):
    """Run one seed: MD + vacuum rerun + cutoff rerun + water decomposition."""
    tmpdir = os.path.join(base_dir, f'run_seed{seed}')
    os.makedirs(tmpdir, exist_ok=True)

    run_outfile = os.path.join(tmpdir, 'cv_accel_decomp_data.npz')
    if os.path.exists(run_outfile):
        print(f"  [seed={seed}] Already done, loading {run_outfile}")
        return run_outfile

    t_run = time.time()

    # ── 1. MD (PME) ──
    print(f"  [seed={seed}] MD ({nsteps} steps)...", flush=True)
    t0 = time.time()
    full_trr, full_tpr = run_md(gro_path, full_top, ff_dir, gmx,
                                 tmpdir, dt=dt, nsteps=nsteps, seed=seed)
    if full_trr is None:
        print(f"  [seed={seed}] MD FAILED"); return None
    print(f"  [seed={seed}] MD done in {time.time()-t0:.0f}s", flush=True)

    # ── 2. Extract solute ──
    print(f"  [seed={seed}] Extract solute...", flush=True)
    t0 = time.time()
    butane_trr = extract_butane_trr(full_trr, full_tpr, gmx, tmpdir)
    if butane_trr is None:
        print(f"  [seed={seed}] Extract FAILED"); return None
    print(f"  [seed={seed}] Extract done in {time.time()-t0:.0f}s", flush=True)

    # ── 3. Vacuum rerun (cutoff) → F_intra ──
    vacuum_tpr = prepare_vacuum_tpr(butane_trr, ff_dir, gmx, tmpdir)
    if vacuum_tpr is None:
        print(f"  [seed={seed}] Vacuum TPR FAILED"); return None
    print(f"  [seed={seed}] Vacuum rerun...", flush=True)
    t0 = time.time()
    rerun_trr_path = rerun_vacuum(butane_trr, vacuum_tpr, gmx, tmpdir)
    if rerun_trr_path is None:
        print(f"  [seed={seed}] Vacuum rerun FAILED"); return None
    print(f"  [seed={seed}] Vacuum rerun done in {time.time()-t0:.0f}s", flush=True)

    # ── 4. Full-system rerun with cutoff (no PME) → F_total_cutoff ──
    print(f"  [seed={seed}] Full-system cutoff rerun...", flush=True)
    t0 = time.time()
    cutoff_tpr = prepare_cutoff_full_tpr(gro_path, full_top, ff_dir, gmx, tmpdir)
    if cutoff_tpr is None:
        print(f"  [seed={seed}] Cutoff TPR FAILED"); return None
    cutoff_rerun_trr = rerun_cutoff_full(full_trr, cutoff_tpr, gmx, tmpdir)
    if cutoff_rerun_trr is None:
        print(f"  [seed={seed}] Cutoff rerun FAILED"); return None
    print(f"  [seed={seed}] Cutoff rerun done in {time.time()-t0:.0f}s", flush=True)

    # ── 5. Read data ──
    print(f"  [seed={seed}] Reading TRRs...", flush=True)
    t0 = time.time()
    xyz_but, vel_but, _ = read_trr(butane_trr)
    _, _, frc_vac = read_trr(rerun_trr_path)
    F_total_cutoff = read_solute_forces_stream(cutoff_rerun_trr)
    print(f"  [seed={seed}] TRRs read in {time.time()-t0:.0f}s", flush=True)

    # ── 6. Stream full TRR: F_total_PME + Python pair forces ──
    print(f"  [seed={seed}] Computing pair force decomposition "
          f"(Rcut={Rcut} nm)...", flush=True)
    t0 = time.time()
    F_total_PME, F_close, F_far, n_close = compute_pair_forces_decomposed(
        full_trr, Rcut)
    print(f"  [seed={seed}] Pair forces done in {time.time()-t0:.0f}s", flush=True)
    print(f"  [seed={seed}] Mean close solvent molecules: "
          f"{np.mean(n_close):.1f}", flush=True)

    nframes = xyz_but.shape[0]
    pos_all = xyz_but.astype(np.float64)
    vel_all = vel_but.astype(np.float64)
    F_intra = frc_vac[:, :, :].astype(np.float64)

    # ── 7. PME correction ──
    F_PME_corr = F_total_PME - F_total_cutoff

    # ── 8. Validation ──
    #   close + far should equal F_total_cutoff − F_intra  (cutoff is pairwise additive)
    F_water_cutoff = F_total_cutoff - F_intra
    F_water_python = F_close + F_far
    diff = F_water_cutoff - F_water_python
    rms_diff = np.sqrt(np.mean(diff**2))
    rms_water = np.sqrt(np.mean(F_water_cutoff**2))
    print(f"  [seed={seed}] Validation: |F_close+F_far − (F_cut−F_intra)| "
          f"RMS = {rms_diff:.2e}  ({rms_diff/max(rms_water,1e-30)*100:.4f}%)")
    rms_pme = np.sqrt(np.mean(F_PME_corr**2))
    print(f"  [seed={seed}] PME correction RMS = {rms_pme:.4f} kJ/(mol·nm)")

    # ── 9. JAX decomposition ──
    print(f"  [seed={seed}] JAX decomposition ({nframes} frames)...", flush=True)
    t0 = time.time()
    res = process_trajectory(pos_all, vel_all, F_total_PME, F_intra,
                             F_close, F_far, F_PME_corr, dt)
    print(f"  [seed={seed}] JAX done in {time.time()-t0:.0f}s", flush=True)

    # Check: close + far + pme_corr = water
    err = np.abs(res['close'] + res['far'] + res['pme_corr'] - res['water'])
    print(f"  [seed={seed}] Decomp check: "
          f"max|close+far+pme−water| = {np.max(err):.2e}, "
          f"mean = {np.mean(err):.2e}")

    np.savez(run_outfile,
             phi=res['phi'], hess=res['hess'],
             intra=res['intra'], water=res['water'],
             close_water=res['close'], far_water=res['far'],
             pme_corr=res['pme_corr'],
             constr=res['constr'], ddot_fd=res['ddot_fd'],
             dt=dt, seed=seed, Rcut=Rcut, n_close=n_close)
    print(f"  [seed={seed}] Saved to {run_outfile}")
    print(f"  [seed={seed}] Total: {time.time()-t_run:.0f}s", flush=True)
    return run_outfile


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gro_file = sys.argv[1] if len(sys.argv) > 1 else 'npt.gro'
    dt       = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    t_total  = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    nruns    = int(sys.argv[4])   if len(sys.argv) > 4 else 5

    single_seed = None
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        single_seed = int(sys.argv[idx + 1])

    Rcut = 0.5
    if '--Rcut' in sys.argv:
        idx = sys.argv.index('--Rcut')
        Rcut = float(sys.argv[idx + 1])

    nsteps = int(t_total / dt)

    gro_path = os.path.join(base_dir, gro_file)
    if not os.path.exists(gro_path):
        gro_path = os.path.join(base_dir, 'em.gro')

    gmx      = os.environ.get('GMX_CMD', 'gmx_mpi')
    ff_dir   = os.path.join(base_dir, 'gromos53a6.ff')
    full_top = os.path.join(base_dir, 'topol.top')

    seeds = [42, 137, 271, 314, 577][:nruns]

    # ── Single-seed mode ──
    if single_seed is not None:
        print(f"=== Single-seed mode: seed={single_seed}, dt={dt}, "
              f"t={t_total} ps, Rcut={Rcut} nm ===")
        run_single_seed(base_dir, gro_path, full_top, ff_dir, gmx,
                        single_seed, dt, nsteps, Rcut)
        return

    # ── Aggregation mode ──
    print(f"=== CV acceleration decomposition (close/far water split) ===")
    print(f"  dt = {dt} ps, t_total = {t_total} ps ({nsteps} steps)")
    print(f"  Rcut = {Rcut} nm, rvdw = {RVDW} nm, rcoulomb = {RCOULOMB} nm")
    print(f"  {nruns} independent runs, seeds = {seeds}")
    print(f"  constraints = all-angles\n")

    all_hess, all_intra, all_water = [], [], []
    all_close, all_far, all_pme = [], [], []
    all_constr = []
    all_ddot_fd, all_ddot_phys, all_ddot_full = [], [], []
    all_phi = []

    for seed in seeds:
        run_file = os.path.join(base_dir, f'run_seed{seed}',
                                'cv_accel_decomp_data.npz')
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
        all_close.append(rd['close_water'][s])
        all_far.append(rd['far_water'][s])
        all_pme.append(rd['pme_corr'][s])
        all_constr.append(rd['constr'][s])
        all_ddot_fd.append(rd['ddot_fd'][s])
        all_ddot_phys.append((rd['hess'] + rd['intra'] + rd['water'])[s])
        all_ddot_full.append(
            (rd['hess'] + rd['intra'] + rd['water'] + rd['constr'])[s])

    if not all_hess:
        print("No successful runs.")
        return

    hess     = np.concatenate(all_hess)
    intra    = np.concatenate(all_intra)
    water    = np.concatenate(all_water)
    close    = np.concatenate(all_close)
    far      = np.concatenate(all_far)
    pme_corr = np.concatenate(all_pme)
    constr   = np.concatenate(all_constr)
    ddot_fd   = np.concatenate(all_ddot_fd)
    ddot_phys = np.concatenate(all_ddot_phys)
    ddot_full = np.concatenate(all_ddot_full)
    phi = np.concatenate(all_phi)
    ntot = len(hess)

    print("=" * 80)
    print(f"AGGREGATE RESULTS ({nruns} runs, {ntot} interior frames, Rcut={Rcut} nm)")
    print("=" * 80)

    err_phys = np.abs(ddot_phys - ddot_fd)
    err_full = np.abs(ddot_full - ddot_fd)
    mean_mag = np.mean(np.abs(ddot_fd))

    print(f"\nClosure check:")
    print(f"  mean|ddot_FD|                      = {mean_mag:.2f} rad/ps^2")
    print(f"  mean|hess+intra+water − FD|         = {np.mean(err_phys):.4f}  "
          f"({np.mean(err_phys)/mean_mag*100:.2f}%)")
    print(f"  mean|hess+intra+water+constr − FD|  = {np.mean(err_full):.4f}  "
          f"({np.mean(err_full)/mean_mag*100:.2f}%)")

    err_decomp = np.abs(close + far + pme_corr - water)
    print(f"\nDecomposition check (close + far + pme_corr = water):")
    print(f"  max error  = {np.max(err_decomp):.2e}")
    print(f"  mean error = {np.mean(err_decomp):.2e}")

    print(f"\nMean contributions:")
    print(f"  Hessian (kinematic) : {np.mean(hess):10.2f} rad/ps^2")
    print(f"  Intramolecular     : {np.mean(intra):10.2f} rad/ps^2")
    print(f"  Water (total)      : {np.mean(water):10.2f} rad/ps^2")
    print(f"    Close water      : {np.mean(close):10.2f} rad/ps^2")
    print(f"    Far water        : {np.mean(far):10.2f} rad/ps^2")
    print(f"    PME correction   : {np.mean(pme_corr):10.2f} rad/ps^2")
    print(f"  Constraint         : {np.mean(constr):10.2f} rad/ps^2")
    print(f"  Total (FD)         : {np.mean(ddot_fd):10.2f} rad/ps^2")

    print(f"\nRMS contributions:")
    print(f"  Hessian (kinematic) : {np.sqrt(np.mean(hess**2)):10.2f} rad/ps^2")
    print(f"  Intramolecular     : {np.sqrt(np.mean(intra**2)):10.2f} rad/ps^2")
    print(f"  Water (total)      : {np.sqrt(np.mean(water**2)):10.2f} rad/ps^2")
    print(f"    Close water      : {np.sqrt(np.mean(close**2)):10.2f} rad/ps^2")
    print(f"    Far water        : {np.sqrt(np.mean(far**2)):10.2f} rad/ps^2")
    print(f"    PME correction   : {np.sqrt(np.mean(pme_corr**2)):10.2f} rad/ps^2")
    print(f"  Constraint         : {np.sqrt(np.mean(constr**2)):10.2f} rad/ps^2")
    print(f"  Total (FD)         : {np.sqrt(np.mean(ddot_fd**2)):10.2f} rad/ps^2")

    print(f"\nPer-run RMS of total (FD):")
    for i, arr in enumerate(all_ddot_fd):
        print(f"  Run {i+1} (seed={seeds[i]}): "
              f"{np.sqrt(np.mean(arr**2)):8.2f} rad/ps^2  ({len(arr)} frames)")

    outfile = os.path.join(base_dir, 'cv_accel_decomp_data.npz')
    np.savez(outfile, phi=phi, hess=hess, intra=intra, water=water,
             close_water=close, far_water=far, pme_corr=pme_corr,
             constr=constr, ddot_fd=ddot_fd,
             dt=dt, seeds=seeds, nruns=nruns, t_total=t_total, Rcut=Rcut)
    print(f"\nRaw data saved to {outfile}")


if __name__ == '__main__':
    main()
