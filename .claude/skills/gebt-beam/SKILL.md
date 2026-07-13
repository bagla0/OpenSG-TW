---
name: gebt-beam
description: "Build and run a GEBT (Geometrically Exact Beam Theory) 1-D beam model from cross-sectional Timoshenko 6x6 stiffness. Use when the user wants to assemble/run a GEBT input .dat for a blade or beam from per-station 6x6 (VABS .K or OpenSG RM/JAX/FEniCS), including converting a surface traction (e.g. 1200 Pa flapwise) into the beam-equivalent distributed load."
---

# GEBT beam model from cross-section Timoshenko 6x6

Build the GEBT input `.dat` and run `gebt.exe`. Companion of the `gebt-beam-runner` agent; full detail in
memory `ref_gebt_beam` (read it). A transferable copy of the agent lives in `.claude/agents/`.

## Key facts
- **GEBT sections take the 6x6 FLEXIBILITY (compliance = inv(stiffness)), NOT the stiffness.** Confirmed
  from the bundled examples (section 6x6 has ~1e-8 diagonals).
- Program (Windows/cygwin): `OneDrive\2024_195\Codes\gebt\gebt\gebt.exe` (+ `cyg*.dll`, `GEBTManual.pdf`,
  `examples.zip` → `cantilever1.dat` tip point-load, `cantilever8.dat` distributed load). Run `& gebt.exe
  input.dat` → `input.dat.out`.
- Timo 6x6 sources (VABS/OpenSG order 1=ext,2-3=shear,4=torsion,5-6=bend): VABS `.K`; OpenSG
  `homo_rm/C6_rm_*` (RM shell), `homo_jax/C6_jax_*` (JAX solid), `homo_fenics/C6_fenics_*` (FEniCSx solid) —
  see `ref_iea_all_stations`. JAX≡FEniCS≡VABS-solid; RM≈solid ~1-2%.

## Input `.dat` layout (check column order against an example before writing)
`dynamic_flag niter nstep` / counts (`npoint nmemb ncond_pt nmate nframe ndistr ntimefun ncurv nvel`) /
keypoints (x1 = span axis) / members (`kp1 kp2 csA csB frame# n_elem distr# vel#`) / point conditions
(dofs 1-6 = disp/rot, 7-12 = F1..M3; clamp ROOT = dofs 1-6, values 0) / ndistr load blocks / **nmate
FLEXIBILITY 6x6 blocks** / nframe 3x3 / sim range / time functions.

## FBD: surface traction → per-span beam load (Camarena's method)
A uniform FIXED-DIRECTION traction `p` (e.g. **1200 Pa flapwise x3**) on the whole OML (a normal pressure
nets zero on a closed section). Per unit span:
- **f3 = p·P**  (P = section perimeter)  — flapwise distributed force [N/m]
- **m1 = p·∮(x2−x2ref) ds**  — torsion (≈0 at the perimeter centroid)
- f1=f2=m2=m3=0
Get P(r) from `opensg_io.build_cross_section(blade, r)` OML contour arc length. Ernesto's paper:
`OneDrive\Latex Conference publications\windblade\shell_critical_Erensto.pdf` (extract via `pdftotext` on
the Linux server). Apply as a GEBT distributed (`ndistr`) load, piecewise-constant per member at the
member-midspan f3.

## Steps
1. Pick the stiffness `--source K|rm|jax|fenics`; read each station's 6x6; form `S = inv(K)`.
2. Keypoints at x1 = r·blade_length; members between; clamp root; sections = the flexibility blocks.
3. Compute the FBD load; write it as the distributed load; state the derivation in the file header.
4. Write the `.dat`; validate field-by-field vs an example; run `gebt.exe`; parse `.out` (tip deflection,
   resultants); report. Errors are almost always a counts/member column mismatch or stiffness-instead-of-
   flexibility — check those first.
