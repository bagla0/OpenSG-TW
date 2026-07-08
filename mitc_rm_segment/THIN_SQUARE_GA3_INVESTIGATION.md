# Thin tapered-SQUARE GA3 (C33) deficit — root-cause investigation

**Scope:** tapered square-tube shell homogenization (`taper_square.py` + `taper_study.shell_solve`,
general-RM operator `segment_element_general.py`) vs FEniCS 3-D solid
(`examples/data/benchmark/taper_square_solid_{iso,m45}.npz`). Env `C:\conda_envs\opensg_2_0_env`.
Harness: `dbg_ga3.py` (commands: `both`, `refine`, `arsweep`, `eps`, `ablate`, `floor`, `circle`).

## TL;DR

The thin tapered-square transverse-shear stiffness **GA3 (C33)** is under-predicted −24% (iso) / −40%
([-45]) vs solid; **C36 tracks it**. Root cause is **NOT** what the code comment (lines 423–427)
asserts (the transverse-shear strain rows). It is the **ω₃-drilling elimination degenerating on the
flat walls where `C33 = n·b₃ ≡ 0`** — the walls that carry the segment GA3 shear flow under taper.
It is **square/flat-wall specific**: the circular tube (no whole-wall `C33=0`) is clean and symmetric.
It is a **converged formulation (model) error, not discretization**. Real curved blade cross-sections
(airfoils, like the circle) are unaffected.

## What was reproduced

| case | SEG C33 | SEG C22 | SEG C36 |
|---|---|---|---|
| thin_iso_aR070 | **−24.4%** | −1.1% | (C36≈0, iso) |
| thin_m45_aR070 | **−39.9%** | +5.5% | **−39.7%** |
| thick_iso_aR070 | −5.7% | −5.5% | — |
| thick_m45_aR070 | −1.3% | +8.6% | +8.5% |

## Evidence chain (each step is a `dbg_ga3.py` experiment)

1. **Localized to the tapered segment.** The prismatic boundary rings are clean and symmetric
   (thin_m45 ring C33 = **+0.9%** while segment C33 = −39.9%). The entire deficit is in the taper.
2. **Not discretization / not quad distortion (rules out ideas 1 & 2).** Axial refinement NL=10→80
   leaves C33 **flat** (iso −24.4→−24.9%, m45 −39.9→−40.2%) even as the trapezoids → parallelograms.
   Converged operator error.
3. **Second-order in taper, thin-wall only.** Extra deficit (beyond the aR=1 prismatic floor) ∝
   (dR/dz)² at small taper (iso 0.9%/.025² ≈ 3.6%/.05² ≈ 1440) and **essentially zero for thick walls**
   (thick_iso is flat ≈−6% for all aR). So the taper-induced loss is a distinct thin-wall, 2nd-order effect.
4. **NOT the transverse-shear strain operator.** Ablating **all** taper-activated shear sub-blocks
   (BGe macro coupling, BGl w′ block, y1=n·b₁ couplings) moves C33 by <1% (−24.4→−25.2%). This
   **overturns the code's own comment** ("Fix belongs in the strain rows BGe/BGh/BGl").
5. **It IS the ω₃-drilling-in-curvature (Lambda) mechanism.** `LAMBDA_ON=0` collapses **both** GA2
   and GA3 to **zero** in the segment (while the prismatic ring stays fine) — i.e. under taper the
   segment's transverse-shear stiffness is generated *through* the drilling-curvature coupling.
6. **Not the C33 regularizations.** GA3 is insensitive to the shear-block Tikhonov `C33_EPS`
   (0.005→0.3: no change — because `invc33 = y3/(y3²+ε²) = 0` identically on the `y3=0` walls) and to
   the curvature-block floor `FLOOR33` (0.005→0.5: −24.2→−24.8%). The floor *value* is not the lever;
   the elimination being singular there is.
7. **Geodesic curvature `kg` irrelevant** on flat walls (kg=0 by construction; ablation null).
8. **Square-specific (the decisive control).** The **circular** tube thin taper is **clean and
   symmetric**: C33 +3.7% (iso) / +4.9% (m45), with **C22 ≈ C33** (no asymmetry). On the circle
   `C33 = n·b₃` is zero only at isolated points; on the square whole walls have `C33 ≡ 0`.

## Root cause

The RM drilling rotation ω₃ is eliminated algebraically via the in-plane symmetry constraint,
`ω₃ = S/(2 C33) − (C₃ᵦ/C33) ωᵦ`, with denominator `C33 = n·b₃`. On the two square walls whose
membrane shear carries the section shear force in direction 3, the normal is ∥ b₂, so `C33 = n·b₃ ≡ 0`
over the **entire wall** (not isolated points, as on a smooth section). There the elimination is
singular: ω₃ is no longer determined by the constraint, yet its contribution to the transverse shear
is dropped/floored. Under taper the segment routes GA3 through exactly this (now-degenerate) channel,
so GA3 is under-represented. GA2's carrier walls have `C33 = ±1` (healthy elimination) → GA2 stays
correct. This is the classic drilling-at-folds pathology of drilling-eliminated shell formulations
(cf. the existing `GDRILL_ON=0` note: keeping the eliminated drilling active on folds blows GJ up +1280%).

## Recommended fix (theoretically endorsed, deferred as invasive)

Row-selective, **not** a better regularization scalar: on walls with `|C33| = |n·b₃| < tol`, **do not
eliminate ω₃** — keep it as an independent rotation DOF carrying transverse-shear stiffness through the
wall's own G (as the un-reduced RM kinematics intend, and as GA2's healthy walls already do). Keep the
Tikhonov/floor drop of ω₃ only in the twist/curvature rows (to preserve the GJ fold cure). This is a
5→6-DOF-on-singular-walls structural change and must be validated against the full battery
(prismatic `verify_strains_paper`, circle taper, thick square, GA2, GJ-no-blowup, aR sweep) before adoption.

## Practical impact / bound

- Confined to **flat-walled / folded tapered** sections (square-tube stress test). **Curved sections
  are unaffected**: circle thin-taper GA3 < 5%; real blade airfoil skins are curved.
- For flat-walled tapered sections, bound the thin GA3 (and C36) shortfall as ≈ (dR/dz)²-growing,
  reaching −24% (iso) / −40% ([-45]) at dR/dz = −0.15, t/R = 0.02; negligible for t/R ≳ 0.2.

## Diagnostic infrastructure left in place (all default to no-op / production)

`segment_element_general.py`: `SH_ABL_BGE/_BGL/_Y1` (shear sub-block ablations), `KG_ABL`,
`G_SHEAR_SCALE`, `FLOOR33` (live drilling-floor). `LAMBDA_ON` already existed. All verified no-op at
defaults (production C33 = −24.4%/−39.9% unchanged). Harness `dbg_ga3.py`.
