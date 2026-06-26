# Tutorials

Three **executed** notebooks — each loads an OpenSG YAML, draws the $e_1/e_2/e_3$ material orientation,
computes the Timoshenko $6\times6$, and reports the per-term **%-error against a benchmark**. They are
committed pre-run, so the numbers and figures you see are the real outputs.

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

```{list-table} Cross-sections and benchmarks used
:header-rows: 1
:widths: 22 26 26 26

* - Tutorial
  - Input YAML
  - Driver
  - Benchmark
* - RM
  - `tube_thesis_314/.../shell_center.yaml`
  - `rm_timoshenko_6x6`
  - 2-D solid `C6_solid_314.txt`
* - KL
  - `tube_thesis_314/.../shell_center.yaml`
  - `gradient_junction_kirchhoff`
  - 2-D solid `C6_solid_314.txt`
* - Solid
  - `prevabs_mh104/2Dsolid_VABS_mh_104.yaml`
  - `compute_timo_from_yaml`
  - VABS `mh104.sg.K`
```

Each notebook can be reproduced from the command line by the matching numbered script in `examples/`
(see {doc}`../examples`).
