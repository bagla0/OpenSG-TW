# RM-OpenSG Pagano validation collection

The curated Pagano cylindrical-bending validation set of the OpenSG-RM plate
chain (`opensg_jax/fe_jax/msg_rm_plate.py`): four laminate archetypes whose
through-thickness stress recovery is benchmarked against the exact 3-D
elasticity solution (state-space Pagano solver, `examples/garg/pagano_exact.py`)
and, for the Yu cases, an Abaqus 3-D solid model.

This folder is SELF-CONTAINED — the MAIN home of the Pagano validation work:

- `garg/` and `yu2003/` are complete copies of the original pipeline folders
  (exact solver `garg/pagano_exact.py`, statics/FSDT chain, benchmark
  orchestrators, Abaqus deck generators, all case subfolders and drivers) —
  everything RUNS from here, e.g.
  `python examples/RM_OpenSG_pagano/garg/caseA/7_helper_RM_Pagano_benchmark1.py`
- `garg_caseA/`, `garg_caseC/`, `yu2003_case1/`, `yu2003_case2/` are the
  CURATED four-case highlight set (data snapshots picked for the paper story;
  the driver scripts inside them are provenance copies that run from the
  pipeline folders above).

## The four cases and why these

| folder | layup (bottom→top) | problem | unique demonstration |
|---|---|---|---|
| `garg_caseA` | [0/90/0], Pagano 25:1 graphite/epoxy, equal thirds | top-loaded strip, S = a/h ∈ {4, 10, 50} | THE canonical Pagano-1969 benchmark; thickness convergence of the recovery |
| `garg_caseC` | [0/core/0] sandwich (faces 0.1h, soft core 0.8h) | same, S ∈ {4, 10, 50} | the FSDT-collapse case: constitutive-staircase errors 4200–5470 % while MSG-RM stays clean |
| `yu2003_case1` | [15/−15] antisymmetric angle ply | Yu-2003 sec. 6.1: L/h = 4, split face load ±p₀/2, psi units | B₁₆ extension–twist coupling; nonzero σ₂₃ recovered (impossible for cross-ply sets) |
| `yu2003_case2` | [30/−30/−30/30] symmetric angle ply | same | D₁₆ bending–twist coupling: the strongest σ₂₃ (~0.52 p₀) with M₁₂/Q₂ supplied by the 8×8 |

Not included by choice: garg caseB [0/90/90/0] repeats caseA's cross-ply
physics with milder contrast; yu2003 case3 [0.5/90.5/90.5/0.5] is a
near-duplicate of caseA (Yu perturbed the angles only for Sutyrin's code) with
a σ₂₃ too small to plot meaningfully.

## Headline numbers (rel. L2 vs exact 3-D)

garg cases (σ₁₃ at x = 0 / σ₃₃ at x = a/2; FSDT = Whitney-1973 k₁² staircase):

| case | S = 4 | S = 10 | S = 50 | FSDT σ₁₃ |
|---|---|---|---|---|
| caseA σ₁₃ | 20.93 % | 4.43 % | 0.18 % | 100–109 % |
| caseA σ₃₃ | 5.46 % | 1.16 % | 0.05 % | — |
| caseC σ₁₃ | 41.55 % | 9.17 % | 0.39 % | 4199–5468 % |
| caseC σ₃₃ | 4.76 % | 0.82 % | 0.03 % | — |

yu2003 cases at L/h = 4 (deliberately thick; first-order Eq.-63 recovery, σ₃₃
by thickness equilibrium from the loaded bottom face):

| case | σ₁₃ | σ₂₃ | σ₃₃ | Abaqus-FF chain vs 3-D solid |
|---|---|---|---|---|
| case1 | 3.55 % | 5.54 % | 1.64 % | 3.57 / 5.38 / 1.88 % |
| case2 | 9.64 % | 15.99 % | 4.59 % | 9.67 / 15.92 / 4.65 % |

The `yu_case*_6p2.*` files are the sec.-6.2 analog: Abaqus in the DYMORE role
(RM-shell section forces → FF → OpenSG-RM recovery), benchmarked against the
width-tied C3D8I solid strip (solid-vs-exact credentials σ₁₃ 3.7 / 1.1 %,
σ₃₃ ≤ 0.7 %).

## What each case folder holds

- `*_sg.yaml` + `*_sg.png` — the through-thickness 1-D SG mesh and its figure
- `pagano_S*.dat` / `yu_case*.dat` — full benchmark tables: the RM 8×8 ABDG,
  the FF input and statics anchors, both shear-stiffness constructions,
  Whitney k₁², error summary, and the pointwise profile columns
- `pagano_S*.png` / `yu_case*.png` — the comparison figures
- `rm_8x8.out` — the labeled 8×8 ABDG per aspect ratio
- `*.inp` — the submit-ready Abaqus decks (MSG general-section, community-FSDT,
  and for the Yu cases the 3-D solid benchmark strip)
- `Abaqus_Plate/` — real Abaqus run artifacts: job `.dat` (Yu cases: RM, FSDT,
  SOLID) or `.rpt` + example-7 dehomogenization outputs (garg caseA)
- `garg_*.SM/.EM/.U/.out` — example-7 dehom outputs at the documented FF
- `yu_case*_6p2.dat/.png` — the Abaqus-driven (sec.-6.2) recovery comparison

## Regenerating

```bash
python examples/garg/caseA/7_helper_RM_Pagano_benchmark1.py
python examples/garg/caseC/7_helper_RM_Pagano_benchmark3.py
python examples/yu2003/case1/7_helper_RM_Pagano_yu2003_case1.py
python examples/yu2003/case2/7_helper_RM_Pagano_yu2003_case2.py
python examples/yu2003/recover_6p2.py            # needs the Abaqus job .dat files
```

Method chain (strict separation, no Pagano content in the model inputs): exact
3-D reference = `pagano_exact.py`; MSG-RM = statics/harmonic-solve FF → 8×8
inversion → Yu-2003 Eq.-63 recovery → σ₃₃ by thickness equilibrium; FSDT =
Whitney-1973 Eq.-(7) k₁² staircase (`statics_fsdt.py`).
