# Reproduce the ASC-2026 tube tables (Timoshenko 6×6)

Standalone reproduction of the **single-cell** and **two-cell tube** results in the paper
*"OpenSG: A Native JAX-MSG Framework for Thin-Walled Composite Beams"* (ASC 2026).

It re-runs the two JAX shell homogenizations — **Kirchhoff–Love** (gradient / Hermite-`C1`)
and **Reissner–Mindlin** (MITC, `shear="mitc_both"`) — for every case, stores each
Timoshenko `6×6`, plots the `e1/e2/e3` orientation, and reports the percent error against
the **FEniCS 2-D-solid (VABS)** reference next to the numbers printed in the paper.

Timoshenko order everywhere: **`[EA, GA2, GA3, GJ, EI2, EI3]`**
(`C11=EA C22=GA2 C33=GA3 C44=GJ C55=EI2 C66=EI3`; `C14`=extension–twist, `C25`=`GA2–EI2`, `C36`=`GA3–EI3`).

## Cases

| case | geometry | material | reference |
|------|----------|----------|-----------|
| `single_rh01 … single_rh10` | single-cell circle, R = 0.0715 m, `R/h = 1…10` | single `[-45°]` ply (ud_frp) | `reference/C6_solid_rhNN.txt` |
| `2cell_iso_thin/thick` | two-cell tube (diametral web), R = 0.05 m, `R/h = 12.5 / 3.1` | isotropic (E = 68.9 GPa, ν = 0.33) | `reference/C6_solid_tube2cell_*.txt` |
| `2cell_aniso_thin/thick` | same two-cell geometry | `[-45°]` ud_frp | `reference/C6_solid_tube2cell_aniso_*.txt` |

`single_rh02` / `single_rh10` are the paper's *thick* / *thin* single-cell tables;
the two-cell iso/aniso thin/thick cases are the paper's multi-cell tables.

## Two curvature paths (intentional)

- **single-cell smooth circle → exact hoop curvature `k22 = −1/R`.** A plain circle is a
  known smooth surface, so the curvature is imposed analytically (`lib/tube_lib.homog`,
  `k22_mode="exact"`). This is what the paper used; the generic geometric-curvature driver
  gives a noticeably different (worse) KL shear on the faceted single circle.
- **two-cell webbed tube → geometric per-element curvature.** The internal web is a *flat*
  wall (`k22 ≈ 0`) while the outer wall is curved, so curvature is computed element-by-element
  from the mesh by the public drivers `gradient_junction_kirchhoff` /
  `rm_timoshenko_6x6(curved=True)`.

## Requirements

- An **OpenSG-TW** checkout (this folder lives inside it; the repo root — the folder that
  contains `opensg_jax/fe_jax` — is found automatically, no paths to edit).
- Its conda environment: JAX (x64) + `pypardiso` + `numpy`/`scipy`/`pyyaml` + `matplotlib`.

## Run

From this folder, with the OpenSG-TW python on PATH:

```bash
python run_all.py                 # runs steps 0 → 4
```

or step by step:

```bash
python 0_generate_meshes.py       # single-cell circles  -> meshes/shell_rhNN.yaml
python 1_run_kirchhoff.py         # KL  6x6 all cases     -> results/C6_KL_<case>.dat
python 2_run_rm.py                # RM  6x6 all cases     -> results/C6_RM_<case>.dat
python 3_orientation.py           # e1/e2/e3 PNGs         -> figures/orient_<case>.png
python 4_compare_to_solid.py      # %err vs solid + paper -> results/RESULTS_verification.txt
```

## Layout

```
reproduce_6x6/
  common.py              case list + KL/RM drivers + helpers (portable repo bootstrap)
  lib/gen_meshes.py      single-cell circle mesh generator (pure numpy/yaml)
  lib/tube_lib.py        smooth-tube homogenizer: exact-k22 KL + RM (self-contained)
  0..4_*.py, run_all.py  the pipeline (above)
  meshes/                INPUT 1D-shell YAMLs (single-cell generated; two-cell webbed shipped)
  reference/             FEniCS-2D-solid (VABS) 6x6 references + solid meshes (orientation)
  results/               OUTPUT  C6_KL_*.dat, C6_RM_*.dat, RESULTS_verification.txt
  figures/               OUTPUT  orient_*.png
```

## Reference axis

**The 2-D-solid reference is the geometric center (centroid) for BOTH the single-cell and the
two-cell tube**, matching the shell (which is meshed on the centered mid-wall and referenced at
the axis). For the two-cell tube use the *web-centered* solid 6×6 (`..._wc`, axial–bending
coupling `C16 ≈ 0`); the un-centered solid (`C16 ≈ 7e5`) is offset ~1.6 mm from the axis and must
not be used as the reference — it inflates the two-cell coupling errors by up to ~3 pt.

## Verification result

`results/RESULTS_verification.txt` (written by step 4) tabulates, per case and per term, the
recomputed KL/RM percent error, the paper's tabulated value, and the difference. With the
center-referenced solid for both examples, the rerun reproduces the published tables to
**≤ 0.62 percentage points overall**:

- **Kirchhoff–Love reproduces the paper essentially exactly** — every tabulated term within
  **≤ 0.05 pt** (single-cell `rh02`/`rh10` to 0.00–0.02 pt; two-cell iso/aniso thin & thick).
- **Reissner–Mindlin matches to ≤ 0.62 pt**, and the residual is entirely on the
  transverse-shear terms (`GA2`,`GA3` = `C22`,`C33`) — e.g. aniso-thick `GA3` −2.68 % vs −3.30 %.
  This is a small drift in the RM/MITC default (`shear="mitc_both"`) since the table was made;
  the *model* is unchanged.

See `results/RESULTS_verification.txt` for the exact per-term deltas.
