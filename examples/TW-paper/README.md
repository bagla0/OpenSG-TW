# TW-paper — result-generation files for the OpenSG thin-walled beam paper

This folder archives the scripts, meshes, reference data, and output figures/tables that
generate the results in the OpenSG thin-walled (TW) composite-beam paper — Reissner–Mindlin
(RM) and Kirchhoff–Love (KL) shell models benchmarked against the 2-D solid (OpenSG / VABS).

All RM results use the **`mitc_both`** assumed-strain transverse-shear scheme (both
`2ε13` and `2ε23` tied at the Barlow points), which is the production default.

## Contents

### `single_cell_tube/` — Single-celled tube (paper §"Single-celled Tube")
Single `[-45°]` ply circular tube, mean radius `R = 0.0715 m`, wall swept `R/h = 1..10`,
`N = 3200` shell elements, centric (mid-wall) reference, exact hoop curvature `k22 = -1/R`.
- `scripts/sweep_jax.py`   — JAX-RM (`mitc_both`) + JAX-KL 6×6 for each `R/h`
- `scripts/sweep_meshes.py`, `scripts/sweep_solid.py` — PreVABS/FEniCS 2-D-solid reference (run in WSL)
- `scripts/sweep_plot.py`  — convergence figures → `sweep/figs/sweep_RM.png`, `sweep_KL.png`
- `sweep/make_aniso_tube_table.py` — the `R/h = 2` (thick) and `R/h = 10` (thin) tables
- `sweep/data/` — `shell_rh*.yaml`, `solid_rh*.yaml`, `C6_{solid,jax_rm,jax_kirch}_rh*.txt`, `sweep_errors.txt`
- `sweep/figs/` — `sweep_RM.png`, `sweep_KL.png`, `xsec_tube_single.png`

Generates: Figures `sweep_tube_RM`/`sweep_tube_KL`; Tables `tab:thick_wall_r2`, `tab:thin_wall_r10`.

### `two_cell_tube/` — Two-cell webbed tube (paper §"Two-cell tube")
Tube `R = 0.05 m` with a diametral web; isotropic and `[-45°]` walls; thin (`R/h = 12.5`)
and thick (`R/h = 3.1`).
- `make_paper_tables.py` — KL + RM(`mitc_both`) 6×6 and %-error tables
- `build_tube2cell.py`   — 2-cell shell/solid mesh generation
- `tube2cell_solid.py`, `tube2cell_solid_aniso.py` — FEniCS 2-D-solid reference (WSL)
- `data/` — `tube2cell_*.yaml`, `solid_tube2cell_*.yaml`, `C6_solid_tube2cell_*.txt`
- `tex/`  — `tab_multicell2_iso.tex`, `tab_multicell2_aniso.tex`
- `figures/` — `xsec_2cell*.png`, `conv_2cell_aniso_*.png`

Generates: Tables `tab:multicell2_iso_{thin,thick}`, `tab:multicell2_aniso_{thin,thick}`; Fig `xsec_2cell`.

### `iea22_blade/` — IEA-22-280 reference blade (paper §IEA results)
- `build_iea22_full_blade.py` — 8-station (r = 0.2…0.9) full-blade homogenization driver
- `iea22_full_blade.ipynb`    — executed tutorial (produces the span figures + mid-span table)
- `data/` — `shell_r0*.yaml`, `C6_solid_r0*.txt`, per-station orientation PNGs

Generates: Tables `tab:iea_r050`, `tab:iea_time`; Figures `iea22_blade_span` (shear + classical).

### `lib/` — shared tube homogenization library
- `tube_lib.py`   — `homog()`: JAX-RM (`timoshenko_rm`) + JAX-KL (C1 Hermite); `shear="mitc_both"` default
- `gen_meshes.py` — tube YAML mesh generator + material/geometry constants

### `mitc_both_verification/` — scripts used to update RM to `mitc_both`
- `twocell_rm_only.py`      — two-cell RM(`mitc_both`) %-error vs the published solid
- `single_tube_check.py`    — single-tube `R/h = 2,10`: `mitc` vs `mitc_both` vs published solid
- `sweep_rm_both.py`        — recompute single-tube `C6_jax_rm` with `mitc_both` (all `R/h`)
- `compare_both_twocell.py` — two-cell KL vs RM(`mitc`) vs RM(`mitc_both`)

## Running

These depend on the OpenSG-TW package (`opensg_jax/fe_jax`) and the conda env
`C:\conda_envs\opensg_2_0_env` (JAX CPU + pypardiso). The JAX / shell steps run on Windows;
the 2-D-solid reference steps (`sweep_solid.py`, `tube2cell_solid*.py`) run in WSL under the
FEniCS env `opensg_env_v8`.

```powershell
$env:PATH = "C:\conda_envs\opensg_2_0_env;...;" + $env:PATH   # see repo CLAUDE.md for the full PATH
& "C:\conda_envs\opensg_2_0_env\python.exe" single_cell_tube\scripts\sweep_jax.py
```

> **Note:** the scripts carry absolute `C:\Users\bagla0\...` paths pointing at their original
> `tests/research/...` locations — this folder is an archival copy. Adjust the `DATA` / `TUBE`
> path constants near the top of each script to run them from here.

## Excluded from this archive
PreVABS `.sg` / `.sg.mat` intermediates, `*.log` debug logs, `_t1only` duplicate meshes,
4-cell and web-centered (`_wc`) exploration variants, and `__pycache__` — none are part of
the published paper results.
