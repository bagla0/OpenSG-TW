# KL-Timo Paper — Results Workspace

Dedicated results/work folder for the paper **"OpenSG-2.0: Shell-Based Modeling of
Thin-Walled Composite Beams Using Mechanics of Structure Genome"** (Overleaf project
`opensg-kl-timo`, id `6a43df3b93e759592e665763`, Elsevier / *Composite Structures*).

This paper focuses **only on the Kirchhoff–Love (KL) shell → Timoshenko beam** model,
implemented two ways and cross-compared:

- **FEniCSx — Discontinuous-Galerkin (OpenSG-1.0).** C1 enforced via interior-penalty DG.
- **JAX — Gradient-Kirchhoff cubic-Hermite (OpenSG-2.0).** C1 enforced directly by
  Hermite value+slope DOFs; matrix-free `jacfwd`+`jvp`, no penalty parameter.

Both implement the *same* KL (transverse-shear-free) shell kinematics, so the headline
result is: **do two different C1-enforcement strategies converge to the same Timoshenko
6×6?** — and how each compares to an independent ground truth (analytical / 2D-solid /
VABS).

## Location & sync
This folder lives inside the `OpenSG-TW` git repo, which itself sits under OneDrive —
so it is **both Git-tracked and OneDrive-synced** with no extra step. Commit to push to
GitHub `bagla0/OpenSG-TW`; OneDrive handles device sync automatically.

## Structure
```
kl_timo_paper/
  README.md          this file
  inputs/            1D-shell YAML cross-sections used by both codes
  jax/               JAX gradient-Kirchhoff KL drivers (run natively on Windows)
  fenics/            FEniCSx KL-DG drivers (run in WSL Ubuntu-22.04, dolfinx 0.8.0)
  results/           per-case 6×6 .dat (FEniCSx + JAX), %err 6×6, timing
  figures/           orientation PNGs (solid+shell), convergence/sweep plots
  compare/           comparison + table/plot builders -> figures/ & results/
```

## Planned result cases (mirrors Overleaf §8)
1. **Circular tube** — isotropic + anisotropic [45/-45]; h/R sweep; analytical (Yu 2005)
   as third reference. (figs: `sweep_tube_KL`, `xsec_tube_single`, `aniso`)
2. **BAR-URC wind blade section** — webbed composite section vs 2D-solid/VABS.
3. **Realistic wind turbine blade (IEA-22)** — spanwise stations.

## Per-run convention (standing)
Every case emits: (1) orientation PNG (solid + shell, e1/e2/e3 arrows),
(2) KL/Timo 6×6 `.dat` per code, (3) %err 6×6 with a 1e6 magnitude cutoff.
