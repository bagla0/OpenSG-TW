# Examples

The `examples/` folder holds **numbered, runnable scripts** (the command-line counterparts of the
{doc}`tutorials/index`), following the upstream OpenSG `1_get_beam_props_*` convention.

```{list-table}
:header-rows: 1
:widths: 38 30 32

* - Script
  - Model
  - Reproduces
* - `1_get_beam_props_rm_shell.py`
  - RM shell
  - {doc}`tutorials/rm_timo_from_yaml`
* - `2_get_beam_props_kl_shell.py`
  - KL shell
  - {doc}`tutorials/kl_timo_from_yaml`
* - `3_get_beam_props_jax_solid.py`
  - 2-D solid
  - {doc}`tutorials/solid_timo_from_yaml`
* - `4_run_airfoil_cross_section.py`
  - KL shell (full driver)
  - end-to-end airfoil cross-section → Timoshenko 6×6 (timing + geometry)
* - `5_run_dehomogenization.py`
  - dehomogenization
  - recover the pointwise 3-D stress/strain across the wall from a beam load
```

Each numbered example is a **distinct feature** — the Timoshenko 6×6 from the RM shell, the KL shell or the
2-D solid (1–3), the end-to-end airfoil driver (4), and dehomogenization / local stress recovery (5). The
homogenization examples emit the `e1/e2/e3` orientation PNG, compute the 6×6, and print the per-term
%-error against the benchmark. The many validation, OML-stress and report scripts live under
`examples/benchmarks/`.

```powershell
# Windows: prepend the env to PATH first (see Installation)
python examples\1_get_beam_props_rm_shell.py
python examples\2_get_beam_props_kl_shell.py
python examples\3_get_beam_props_jax_solid.py
```

Two further cross-sections are demonstrations rather than core concepts, so they live elsewhere:

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Case
  - Where
  - What it shows
* - IEA-22 windIO blade
  - tutorial {doc}`tutorials/iea22_windio_to_timo`
  - windIO → OpenSG YAML → full 6×6 vs VABS (uses the solid concept on a real blade)
* - Two-cell [-45] (multi-cell)
  - **test** `tests/test_twocell_m45_benchmark.py` + tutorial {doc}`tutorials/twocell_m45_asc`
  - KL vs RM vs solid across an internal-web junction (the ASC multi-cell benchmark)
```

```powershell
python tests\test_twocell_m45_benchmark.py     # benchmark table
pytest  tests\test_twocell_m45_benchmark.py    # regression assertions
```

## Benchmark utilities

```{list-table}
:header-rows: 1
:widths: 46 54

* - Tool
  - Use
* - `opensg_jax/fe_jax/benchmark_vabs.py`
  - full-6×6 (all couplings) JAX-solid vs a VABS `.K`
* - `scripts/rm_research/tw_regression_guardrail.py`
  - must-pass RM/KL vs 2-D-solid regression on the thin-wall benchmarks
```

```powershell
python -m opensg_jax.fe_jax.benchmark_vabs `
  tests\research\iea22_windio\solid_iea22_r050.yaml `
  tests\research\iea22_windio\prevabs_r050\iea22_r050.sg.K
```

## API

The three public drivers:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Driver
  - Signature
* - JAX 2-D solid
  - `opensg_jax.fe_jax.solid_timo.compute_timo_from_yaml(yaml) -> (6,6)`
* - KL shell
  - `gradient_kirchhoff.gradient_junction_kirchhoff(yaml, frac, dshift) -> (C6, ...)`
* - RM shell
  - `strip_RM.rm_timoshenko_6x6(yaml, frac, dshift, curved, shear="mitc") -> (6,6)`
```
