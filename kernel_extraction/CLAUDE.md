# Kernel Extraction

Non-stationary memory kernel extraction from GLE trajectories.

## Core library

- `nonstationary_kernel.py` — Volterra solvers (rect, trapz, second-kind, deconv), non-stationary and reweighted correlation functions, 1D/2D trajectory generators
- `nonstationary_kernel_lsq.py` — Multi-origin least-squares kernel extraction with Tikhonov regularization

## Comparison scripts

- `compare_volterra_lsq.py` — Volterra vs LSQ at different t0 ranges and N_trajs (1D harmonic GLE, cold start, a_det)
- `compare_volterra_t0.py` — Volterra at different time origins with pre-memory correction, plus averaged multi-t0
- `compare_all_methods.py` — All three methods side by side: Volterra t0=0, averaged Volterra multi-t0, LSQ multi-t0

## Documentation

- `lsq_kernel_method.tex` — LaTeX derivation of the LSQ algorithm, pre-memory explanation, pseudocode

## Key concepts

- **Cold start** (s=0): required for a_det recording — no initial condition bias, Volterra exact at all t0
- **Pre-memory**: at t0 > 0, the GLE integral goes back to t=0; naive slicing ignores this
- **LSQ advantage**: naturally includes pre-memory via j_max = min(n_kernel, t0+tau+1); no error propagation

## Running

```bash
python compare_all_methods.py    # main comparison figure
python compare_volterra_lsq.py   # Volterra vs LSQ
python compare_volterra_t0.py    # time origin effects
```
