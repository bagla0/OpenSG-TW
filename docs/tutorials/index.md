# Tutorials

Seven **executed** notebooks — each loads an OpenSG YAML, draws the $e_1/e_2/e_3$ material orientation,
prints the **full Timoshenko $6\times6$**, and reports the per-term **%-diff on every non-zero $C_{ij}$**
(not just the diagonal) against a benchmark. They are committed pre-run, so the numbers and figures you see
are the real outputs. Every input is bundled in the repo under
[`examples/data/`](https://github.com/bagla0/OpenSG-TW/tree/main/examples/data) — clone and run, no external paths.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 1 · RM shell
:link: rm_timo_from_yaml
:link-type: doc
Reissner–Mindlin Timoshenko from a 1-D shell YAML — recovers $GA_2,GA_3$. ($[-45]$ tube vs 2-D solid)
:::

:::{grid-item-card} 2 · KL shell
:link: kl_timo_from_yaml
:link-type: doc
Kirchhoff–Love Timoshenko from a 1-D shell YAML — exact classical, no transverse shear. ($[-45]$ tube)
:::

:::{grid-item-card} 3 · 2-D solid
:link: solid_timo_from_yaml
:link-type: doc
JAX 2-D solid Timoshenko from a 2-D solid YAML — full 6×6 vs VABS. (MH-104 airfoil)
:::
::::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 4 · IEA-22 windIO → full 6×6
:link: iea22_windio_to_timo
:link-type: doc
Real wind-turbine blade: windIO → OpenSG YAML (via OpenSG_io) → full Timoshenko 6×6, matched to VABS.
:::

:::{grid-item-card} 5 · Two-cell [-45] + R/h convergence
:link: twocell_m45_asc
:link-type: doc
Multi-cell tube (ASC): RM vs KL vs 2-D solid at thin **and** thick walls, full 6×6, plus the $R/h$
convergence plot — RM stays <5% on shear where KL collapses.
:::

:::{grid-item-card} 6 · Station-15 (thick web)
:link: st15_solid_vs_shell
:link-type: doc
A thick-web blade station from both solid and shell — solid exact (quad mesh), shells drift on the web.
:::

:::{grid-item-card} 7 · IEA-22 full blade
:link: iea22_full_blade
:link-type: doc
Eight span stations regenerated from windIO — RM & KL vs the 2-D solid on the full 6×6, spanwise %-diff
table and plot.
:::
::::

```{list-table} Cross-sections and benchmarks used
:header-rows: 1
:widths: 22 26 26 26

* - Tutorial
  - Input YAML
  - Driver
  - Benchmark
* - RM
  - `data/1d_yaml/tube_m45_shell.yaml`
  - `rm_timoshenko_6x6`
  - 2-D solid `data/benchmark/tube_m45_solid_ref.txt`
* - KL
  - `data/1d_yaml/tube_m45_shell.yaml`
  - `gradient_junction_kirchhoff`
  - 2-D solid `data/benchmark/tube_m45_solid_ref.txt`
* - Solid
  - `data/2d_yaml/mh104_solid.yaml`
  - `compute_timo_from_yaml`
  - VABS `data/benchmark/mh104.sg.K`
* - IEA-22
  - `data/2d_yaml/iea22_r050_solid.yaml`
  - `compute_timo_from_yaml` (+ RM)
  - VABS `data/benchmark/iea22_r050.sg.K`
* - Two-cell [-45]
  - `data/1d_yaml/tube2cell_m45_shell.yaml`
  - KL + RM + `compute_timo_from_yaml`
  - 2-D solid `data/benchmark/tube2cell_m45_solid_ref.txt`
* - Station-15
  - `data/1d_yaml/st15_shell.yaml` + `data/2d_yaml/st15_solid.yaml` (quad)
  - KL + RM + `compute_timo_from_yaml`
  - VABS `data/benchmark/st15_vabs.K`
```

All paths are relative to [`examples/`](https://github.com/bagla0/OpenSG-TW/tree/main/examples).

Each notebook can be reproduced from the command line by the matching numbered script in `examples/`
(see {doc}`../examples`).
