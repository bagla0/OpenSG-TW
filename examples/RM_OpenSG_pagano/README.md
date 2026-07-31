# RM-OpenSG Pagano validation — results archive

The curated Pagano cylindrical-bending validation RESULTS of the OpenSG-RM
plate chain (`opensg_jax/fe_jax/msg_rm_plate.py`): four laminate archetypes
benchmarked against the exact 3-D elasticity solution and (Yu cases) an Abaqus
3-D solid model.

**This folder holds DATA ONLY.** All runnable code lives in the original
pipeline folders and regenerates everything here:

```bash
python examples/garg/caseA/7_helper_RM_Pagano_benchmark1.py      # garg_caseA
python examples/garg/caseC/7_helper_RM_Pagano_benchmark3.py      # garg_caseC
python examples/yu2003/case1/7_helper_RM_Pagano_yu2003_case1.py  # yu2003_case1
python examples/yu2003/case2/7_helper_RM_Pagano_yu2003_case2.py  # yu2003_case2
python examples/yu2003/recover_6p2.py     # the Abaqus-driven 6.2 comparison
```

Method chain (strict separation — no Pagano content in the model inputs):
exact 3-D reference = `examples/garg/pagano_exact.py` (state-space solver);
MSG-RM = statics / harmonic-solve FF → 8×8 inversion → Yu-2003 Eq.-63 recovery
→ σ₃₃ by thickness equilibrium; FSDT baseline = Whitney-1973 Eq.-(7) k₁²
staircase (`examples/garg/statics_fsdt.py`).

## The four cases and why these

| folder | layup (bottom→top) | problem | unique demonstration |
|---|---|---|---|
| `garg_caseA` | [0/90/0], Pagano 25:1 graphite/epoxy, equal thirds | top-loaded strip, S = a/h ∈ {4, 10, 50} | THE canonical Pagano-1969 benchmark; thickness convergence of the recovery |
| `garg_caseC` | [0/core/0] sandwich (faces 0.1h, soft core 0.8h) | same, S ∈ {4, 10, 50} | the FSDT-collapse case: staircase errors 4200–5470 % while MSG-RM stays clean |
| `yu2003_case1` | [15/−15] antisymmetric angle ply | Yu-2003 sec. 6.1: L/h = 4, split face load ±p₀/2, psi units | B₁₆ extension–twist coupling; nonzero σ₂₃ recovered |
| `yu2003_case2` | [30/−30/−30/30] symmetric angle ply | same | D₁₆ bending–twist coupling: the strongest σ₂₃ (~0.52 p₀), M₁₂/Q₂ from the 8×8 |

Left out by choice: garg caseB (repeats caseA's cross-ply physics, milder
contrast) and yu2003 case3 (near-duplicate of caseA; σ₂₃ too small to plot).

`202603_PlateRM/` is the RAW Abaqus workspace snapshot copied verbatim from
the remote machine's OneDrive transfer folder: the original garg caseA MSG-deck
job (with its .odb and the .rpt extracts), the community-FSDT runs
(`FSDT/caseA_S10`, `caseA_S4`, `caseB_S10`, `caseC_S10`, `caseC_S4`), and the
yu2003 working copies used during the sec.-6.2 campaign — kept as provenance
for every Abaqus number quoted in the benchmarks.

## Headline numbers (rel. L2 vs exact 3-D)

garg cases (σ₁₃ at x = 0 / σ₃₃ at x = a/2):

| case | S = 4 | S = 10 | S = 50 | FSDT σ₁₃ |
|---|---|---|---|---|
| caseA σ₁₃ | 20.93 % | 4.43 % | 0.18 % | 100–109 % |
| caseA σ₃₃ | 5.46 % | 1.16 % | 0.05 % | — |
| caseC σ₁₃ | 41.55 % | 9.17 % | 0.39 % | 4199–5468 % |
| caseC σ₃₃ | 4.76 % | 0.82 % | 0.03 % | — |

yu2003 cases at L/h = 4 (first-order recovery, deliberately thick):

| case | σ₁₃ | σ₂₃ | σ₃₃ | Abaqus-FF chain vs 3-D solid |
|---|---|---|---|---|
| case1 | 3.55 % | 5.54 % | 1.64 % | 3.57 / 5.38 / 1.88 % |
| case2 | 9.64 % | 15.99 % | 4.59 % | 9.67 / 15.92 / 4.65 % |

FULL-FIELD recovery — ALL six stresses and three displacements per case
(`full_field.dat/.png`; u^2d from the Abaqus RM-shell jobs, fit residuals
≤ 2e-6; the face LOAD ladders V1L/V2L reinstated per Yu Eqs. 29/45 make the
recovered σ₃₃ faces machine-exact and supply the load-driven e₃₃ that σ₂₂
needs — caseA σ₂₂ 30 % → 5.6 % with them):

| case | σ₁₁ | σ₂₂ | σ₁₂ | σ₁₃ | σ₂₃ | σ₃₃ | U₁ | U₂ | U₃ |
|---|---|---|---|---|---|---|---|---|---|
| garg caseA (S = 10) | 2.00 | 5.60 | 0 (mach.) | 4.52 | 0 (mach.) | 1.29 | 1.48 | 0 (mach.) | 1.29 |
| garg caseC (S = 10) | 32.3 † | 56.4 † | 0 (mach.) | 9.13 | 0 (mach.) | 0.88 | 1.29 | 0 (mach.) | 2.17 |
| yu case1 (L/h = 4) | 8.69 | 8.25 | 14.01 | 4.42 | 5.52 | 3.99 | 6.56 | 9.75 | 4.46 |
| yu case2 (L/h = 4) | 15.06 | 15.69 | 18.49 | 10.28 | 16.00 | 6.27 | 4.99 | 9.27 | 3.50 |

(rel. L2 vs exact, %.  σ₃₃ plotted from the equilibrium integration; the
DIRECT Eq.-66 σ₃₃ with the load columns has machine-exact face values and is
recorded in each `full_field.dat`.  † the sandwich in-plane errors are REGIME,
not defect: they collapse 32.3 → 5.4 → 1.35 % (σ₁₁) and 56.4 → 9.9 → 2.48 %
(σ₂₂) over S = 10 → 25 → 50 — the soft-core expansion parameter is simply not
small at S = 10, and FSDT/CLT is no better there.)

σ₁₂ is COMPLETELY validated against Pagano (`sig12_sweep.dat/.png`, archive
root): the cross-ply recovered σ₁₂ sits at the round-off floor
(max|σ₁₂|/max|σ₁₁| ≈ 2e-16, the exact field is identically zero), and the
angle-ply σ₁₂ errors converge at the second-order rate — [30/−30/−30/30]
order 2.00–2.09 (18.5 → 0.067 % over S = 4 → 64), [15/−15] order → 1.95
(14.1 → 0.18 %) — so the thick-plate values above are thickness effects, not
defects.  † the sandwich in-plane recovery at S = 10 is the honest limit of
the gradient expansion over a soft core (σ₂₂ is also a small field there).

Per case folder the new files are `inplane.dat/.png` (σ₁₁, σ₂₂, σ₁₂ at
x = a/2: exact vs MSG-RM Eq. 66 vs FSDT/CLT), `disp_abaqus.dat/.png` (Eq.-65
U₁/U₂/U₃ with u^2d from the Abaqus RM job) and `disp_theory.dat/.png` (same
with the internal harmonic solve).  Generated by
`examples/pagano_recovery/recovery_bench.py`.

## File inventory

### garg_caseA/  (and garg_caseC/ with C in place of A)

| file | content |
|---|---|
| `garg_A_sg.yaml` | the through-thickness 1-D SG mesh (5-noded elements, ply sets, material db) the homogenization reads |
| `garg_A_sg.png` | figure of that SG mesh (line elements colored per ply set) |
| `pagano_S4.dat`, `pagano_S10.dat`, `pagano_S50.dat` | full benchmark table per aspect ratio: RM 8×8 ABDG, statics FF input, Whitney vs MSG shear stiffness, k₁², error summary, then columns z, σ₁₃ (MSG / exact / FSDT), σ₃₃ (MSG / exact) |
| `pagano_S4.png`, `pagano_S10.png`, `pagano_S50.png` | the two-panel comparison figures (σ₁₃ at x = 0; σ₃₃ at x = a/2) |
| `rm_8x8.out` | the labeled RM 8×8 ABDG at every aspect ratio (rows e11, e22, g12, k11, k22, k12, 2g13, 2g23) |
| `garg_A.SM` / `.EM` / `.U` / `.out` | example-7 dehomogenization outputs at the documented FF: material-frame stress, strain, warping displacement, and the run report |
| `garg_caseA_S10.inp` | Abaqus strip deck carrying the MSG-RM 8×8 as *SHELL GENERAL SECTION + MSG *TRANSVERSE SHEAR STIFFNESS |
| `garg_caseA_S10_FSDT.inp`, `garg_caseA_S4_FSDT.inp` | the community-FSDT variants (*SHELL SECTION, COMPOSITE + lamina materials; Abaqus builds ABD + its own shear) |
| `Abaqus_Plate/garg_caseA_S10.dat` | the REAL Abaqus 2024 job output of the MSG deck (validates deck + section: U3 0.013 % vs closed form, M11/Q1 on the statics anchors) |
| `Abaqus_Plate/garg_caseA_S10.inp` | the exact deck that produced it |
| `Abaqus_Plate/garg_caseA_S10_SF_SM.rpt`, `..._U.rpt` | section-force/moment and displacement reports extracted from the .odb (integration-point FF source) |
| `Abaqus_Plate/dehom_mid.*`, `dehom_end.*` | example-7 dehomogenization driven by the Abaqus-extracted FF at the mid-span and end stations (.SM/.EM/.U/.out each) |

(`garg_caseC/` has no `Abaqus_Plate/` — the caseA round trip is the deck
validation for all cases.)

### yu2003_case1/  (and yu2003_case2/ with 2 in place of 1)

| file | content |
|---|---|
| `yu_1_sg.yaml`, `yu_1_sg.png` | the through-thickness 1-D SG mesh of the Yu laminate and its figure |
| `yu_case1.dat` | the sec.-6.1 benchmark table: 8×8 ABDG, the harmonic RM cylindrical-bending solve (DOF amplitudes, resultants, statics checks, coupling M₁₂/Q₂), Whitney k₁², errors, then columns z, σ₁₃ (MSG / exact / FSDT), σ₂₃ (MSG / exact), σ₃₃ (MSG / exact) |
| `yu_case1.png` | the three-panel sec.-6.1 figure (σ₁₃, σ₂₃ at x = 0; σ₃₃ at x = L/2), all σ/p₀ |
| `rm_8x8.out` | the labeled RM 8×8 ABDG (L/h = 4) |
| `yu_case1_RM.inp` | Abaqus S4 strip with the MSG 8×8 general section, width-tied (all 6 dofs — the infinite-plate condition), NO drilling BC (angle-ply lesson) |
| `yu_case1_FSDT.inp` | the community-FSDT strip (TSHR13/23 requested — comes back zero for S4: no inbuilt through-thickness estimate exists) |
| `yu_case1_SOLID.inp` | the 3-D C3D8I benchmark strip: width *EQUATION ties, per-ply *ORIENTATION, split ±p₀/2 face load, harmonic-consistent supports |
| `Abaqus_Plate/yu_case1_RM.dat` | Abaqus job output of the RM deck — the FF source of the 6.2 chain (SF/SM at the x = 0 and x = L/2 stations) |
| `Abaqus_Plate/yu_case1_FSDT.dat` | job output of the FSDT deck (documents the all-zero TSHR finding) |
| `Abaqus_Plate/yu_case1_SOLID.dat` | job output of the solid benchmark (centroidal S along the two station columns; NOTE: printed in the PLY LOCAL frame — recover_6p2.py rotates back) |
| `yu_case1_6p2.dat` | the sec.-6.2 comparison table: Abaqus FF amplitudes, exact-section anchors Q₁/M₁₁, recovery-vs-solid errors, solid-vs-exact credentials, recovery profile columns |
| `yu_case1_6p2.png` | the three-panel 6.2 figure: exact Pagano vs OpenSG-RM recovery driven by the Abaqus FF |
