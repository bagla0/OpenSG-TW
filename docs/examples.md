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
```

Each script: emits the `e1/e2/e3` orientation PNG, computes the Timoshenko 6×6, and prints the per-term
%-error against the benchmark.

```powershell
# Windows: prepend the env to PATH first (see Installation)
python examples\1_get_beam_props_rm_shell.py
python examples\2_get_beam_props_kl_shell.py
python examples\3_get_beam_props_jax_solid.py
```

## Benchmark utilities

```{list-table}
:header-rows: 1
:widths: 46 54

* - Tool
  - Use
* - `opensg_jax/fe_jax/benchmark_vabs.py`
  - full-6×6 (all couplings) JAX-solid vs a VABS `.K`
* - `rm/tw_regression_guardrail.py`
  - must-pass RM/KL vs 2-D-solid regression on the thin-wall benchmarks
```

```powershell
python -m opensg_jax.fe_jax.benchmark_vabs `
  prevabs_mh104\2Dsolid_VABS_mh_104.yaml `
  "training data\opensg-FEniCS\data\mh104_training\mh104.sg.K"
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
