# Tutorials

Six **executed** notebooks — each loads (or generates) an OpenSG YAML, draws the $e_1/e_2/e_3$ material
orientation, prints the **full Timoshenko $6\times6$**, and reports the per-term **%-diff on every non-zero
$C_{ij}$** (not just the diagonal) against a benchmark. They are committed pre-run, so the numbers and figures
you see are the real outputs. Every input is bundled in the repo under
[`examples/data/`](https://github.com/bagla0/OpenSG-TW/tree/main/examples/data) — clone and run, no external paths.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1 · RM shell (+ KL)
:link: rm_timo_from_yaml
:link-type: doc
Reissner–Mindlin Timoshenko on the two-cell $[-45]$ tube, with a Kirchhoff–Love subsection and the
RM / KL / 2-D-solid full-6×6 comparison.
:::

:::{grid-item-card} 2 · IEA-22 full blade
:link: iea22_full_blade
:link-type: doc
Eight span stations regenerated from windIO — RM & KL vs the 2-D solid on the full 6×6, spanwise plots.
:::

:::{grid-item-card} 3 · BAR-URC blade (thick web)
:link: st15_solid_vs_shell
:link-type: doc
A thick-web blade station from both solid and shell — solid exact (quad mesh), shells drift on the web.
:::

:::{grid-item-card} 4 · 3D-SG tapered segment
:link: taper_3dsg_segment
:link-type: doc
Three BAR-URC tapered shell segments (5/12/15) — boundary rings + MITC-RM tapered 6×6 vs the 3-D solid
at the same origin, plus the JAX-vs-OpenSG boundary-YAML equivalence check.
:::

:::{grid-item-card} 5 · Taper convergence (isotropic)
:link: taper_convergence_iso
:link-type: doc
Incremental-taper sweep of the general RM taper operators on an isotropic tube, thin + thick wall —
L/taper/R full 6×6 vs the 3-D solid, convergence plot, no shear locking.
:::

:::{grid-item-card} 6 · Taper convergence ([-45] aniso)
:link: taper_convergence_m45
:link-type: doc
The same sweep with a single-ply −45° wall: surface-following fiber, anisotropic couplings,
thin + thick tables and convergence plot vs the 3-D solid.
:::
::::

```{list-table} Cross-sections and benchmarks used
:header-rows: 1
:widths: 22 26 26 26

* - Tutorial
  - Input YAML
  - Driver
  - Benchmark
* - RM (+ KL)
  - `data/1d_yaml/tube2cell_m45_shell.yaml`
  - `rm_timoshenko_6x6` + `gradient_junction_kirchhoff`
  - 2-D solid `data/benchmark/tube2cell_m45_solid_ref.txt`
* - IEA-22 full blade
  - `data/iea_blade/shell_*.yaml` + `data/2d_yaml/iea22_r050_solid.yaml`
  - RM + KL + `compute_timo_from_yaml`
  - 2-D solid `data/iea_blade/C6_solid_*.txt`
* - BAR-URC blade
  - `data/1d_yaml/st15_shell.yaml` + `data/2d_yaml/st15_solid.yaml` (quad)
  - KL + RM + `compute_timo_from_yaml`
  - VABS `data/benchmark/st15_vabs.K`
* - 3D-SG tapered segment
  - `data/3d_yaml/BAR_URC_numEl_52_segment_{5,12,15}.yaml`
  - `boundary_from_yaml.extract` + `solve_boundary_bundle` + `compute_timo_taper`
  - 3-D solid `data/benchmark/bar_urc_taper_solid_refs.npz`
* - Taper convergence (iso / −45)
  - `data/taper_study/meshes/*.yaml` (generated in-notebook)
  - `taper_study.gen_case` + `ring_general` + `assemble_segment_general`
  - 3-D solid `data/benchmark/taper_study_solid_{iso,m45}.npz`
```

All paths are relative to [`examples/`](https://github.com/bagla0/OpenSG-TW/tree/main/examples).

Each notebook can be reproduced from the command line by the matching numbered script in `examples/`
(see {doc}`../examples`).
