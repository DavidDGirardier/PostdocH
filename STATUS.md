# Project status — butane dihedral barrier crossing

Last updated: 2026-05-04. Working branch: `main` @ `cf00da0+`.

## Physics question

The Grote–Hynes (GH) transmission coefficient computed from the extracted
memory kernel of butane dihedral in water gives κ_GH ≈ 0.89, but the
direct MD reactive flux gives κ_RF ≈ 0.25. The hypothesis being tested
is that the gap is **anharmonicity of the PMF at the barrier**, which the
GH formula assumes parabolic. A 1D GLE simulation that uses the same
memory kernel but the *full* anharmonic mean force should reproduce κ_RF.

## Current numerical results (N = 663 barrier-top shoots, nbins=36, polyfit ω_b)

| Quantity | Value | Comment |
|---|---|---|
| ω_b (barrier frequency) | **24.16 rad/ps** | parabolic polyfit of PMF on \|Δφ\|<0.15 rad. Stable to binning (24.0–24.5 for nbins∈{24,36,48}; spline-derivative-at-φ_b is much noisier). |
| kBT_eff = ⟨φ̇²⟩ | 30.29 rad²/ps² | initial-velocity variance at barrier |
| φ_barrier (numpy convention) | 245.00° | shooting φ(0): mean=244.99°, std=3.14° (well-centered) |
| K(0) (Free-LSQ) | 127–147 rad²/ps² | stride=1 → 127, stride=5 → 147 (FD-noise at small dt) |
| γ_int = ∫K dt | 13.5–15.0 rad/ps | stride=1 → 13.5, stride=5 → 15.0; stable to ~10% |
| κ_RF (MD plateau, 3–8 ps) | **0.250** | reactive flux from shooting trajectories |
| κ_GH (analytic, polyfit ω_b) | 0.899 | Grote–Hynes self-consistency root (stride=5) |
| κ_GLE_full (Free-LSQ K, anharm. F) | **0.241 / 0.223** | stride=5 / stride=1; matches κ_RF within ≈4% / 11% |
| κ_GLE_harm (Free-LSQ K, parabolic F) | 0.889 | matches κ_GH within 1% (sanity) |

The convergence figure (`butane_kappa_convergence.{png,pdf}`) is generated
at stride=5; the single-shot value 0.223 in `butane_gle_kappa.{png,pdf}` is
at stride=1. Both bracket κ_RF = 0.250 within ~10%.

The parabolic-barrier check confirms the GLE integrator is correct;
swapping the parabolic potential for the full anharmonic spline drops
κ from 0.89 to 0.24, closing the gap with κ_RF.

## Pipeline

```
gromacs_run/shooting_barrier/
   run_shooting_barrier.py       # GROMACS shooting from φ_b ≈ 120° (PLUMED)
   equil/                         # 51 bath configs (PLUMED RESTRAINT)
   shoot_seed{1..1007}/
      phi_data.npz                # phi(t), phidot=J·v, ddot via FD
                                  # 1006 successful (seed 807 orphaned)
testASE/butane_dihedral_water/
   butane_rf_vs_kernel.py         # RF vs kernel-derived GH κ; t0_max sweep
   butane_gle_kappa.py            # GLE κ simulation (--method prony2|free-lsq)
   butane_kappa_convergence.py    # κ_RF and κ_GLE vs N subsample
kernel_extraction/
   nonstationary_kernel_lsq.py    # Free-LSQ kernel extractor
   benchmark_prony_nlsq.py        # Prony NLSQ extractor (tau_bounds added)
   benchmark_extended_cases.py    # generate_gle_multiexp,
                                  # generate_gle_arbitrary (force callable + FFT noise)
```

## Key code adjustments made along the way

- **`benchmark_prony_nlsq.py`**: added `tau_bounds=(τ_min, τ_max)` so the
  NLSQ Prony fit can't allocate spurious 10-ps tails.
- **`benchmark_extended_cases.py::generate_gle_arbitrary`**:
  - now accepts `force` callable (was hard-coded double-well);
  - vectorized colored-noise generation across trajectories;
  - fixed FDT scaling: `S_sqrt = sqrt(S)` (was `sqrt(S*dt)`, off by `dt`);
  - allow `nfft ≥ nsteps` for long trajectories.
- **`butane_gle_kappa.py`**: `--n-traj-fit` for self-consistent N sweep,
  `--method {prony2|free-lsq}` switch, kernel interpolation onto fine
  GLE step (1 fs).

## What's confirmed

1. **Grote–Hynes integrator is correct** — parabolic-barrier GLE matches
   the analytic κ_GH within 1%.
2. **Anharmonicity is the dominant cause** of the κ_GH ≫ κ_RF gap.
3. **Free-LSQ kernel is more robust than 2-exp Prony** at large N — Prony
   tends to invent unphysical 10-ps slow tails that bias κ_GLE upward.
4. **ω_b extraction**: spline-derivative-at-φ_b is unstable to binning
   (3.3 ↔ 25.9 across nbins∈{24..120}); narrow PMF polyfit is robust
   (24.0–24.5 for nbins∈{24,36,48}, only 60-bin case dips to 20.6
   from a local PMF artifact). Settled on nbins=36 polyfit → ω_b=24.2.
5. **Stride sensitivity is mild**: κ_GLE moves by ≤0.04 between
   stride ∈ {1, 5}; integrated friction γ_int stable to ~8%.

## Open / next

- Compare to **VolterraBasis** equilibrium kernel from the 40-ns
  unrestrained run. (Plan 4 step 3 from `.recovered_plans.md`.)
- Position-dependent K(φ₀, t) — partially explored in `PositionalKernel_fd.py`
  but not yet plugged into a κ comparison.
- Same protocol with the **trans well** as the reactant state, to check
  the gauche↔trans rate vs Eyring/Kramers prediction.
- Decide whether the residual ~4% gap (0.250 vs 0.241) is statistical or
  reflects multidimensional reaction-coordinate effects.

## Recovery / context

- **Plans from prior sessions**: `.recovered_plans.md` (4 plans extracted
  from `~/.claude/projects/-home-david-Work-PostdocHadrien-kernel-extraction/`).
- **Memory snapshots**: `~/.claude/projects/-home-david-Work-PostdocHadrien/memory/`
  (`project_butane_*.md`, `reference_gromacs_env.md`).
- Always launch `claude` from the repo root so all sessions land in one
  project history.
