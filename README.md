# OpenSG-TW

**JAX Mechanics-of-Structure-Genome beam homogenization** — compute the **Timoshenko 6×6** (and
Euler–Bernoulli 4×4) stiffness of an arbitrary composite cross-section, from an OpenSG YAML, with **no
FEniCSx / MPI**.

### 📖 Documentation → **https://bagla0.github.io/OpenSG-TW/**

[![docs](https://github.com/bagla0/OpenSG-TW/actions/workflows/docs.yml/badge.svg?branch=main)](https://bagla0.github.io/OpenSG-TW/)

---

OpenSG-TW implements the **Mechanics of Structure Genome (MSG)** dimensional reduction in pure JAX
(`basix` only tabulates the element basis/quadrature; `pypardiso` does the sparse saddle-point solve). It
ships three cross-sectional models that share one solver back-end and return the same 6×6 layout
$[\,EA,\;GA_2,\;GA_3,\;GJ,\;EI_2,\;EI_3\,]$:

| Model | Input | Use it for |
|---|---|---|
| **Kirchhoff–Love (KL) shell** | 1-D shell SG YAML | thin walls, classical stiffness (Hermite-$C^1$ arc) |
| **Reissner–Mindlin (RM) shell** | 1-D shell SG YAML | thin walls **with transverse shear** ($GA_2,GA_3$); MITC |
| **2-D solid** | 2-D solid SG YAML | thick / arbitrary sections; full 3-D fidelity, **matched to VABS** |

## Validation

The pure-JAX 2-D solid reproduces the **VABS** Timoshenko 6×6 — diagonals *and* couplings — to a few parts
in $10^6$ across triangle and quad meshes:

| Case | max diagonal err | worst coupling |
|---|---|---|
| MH-104 airfoil | 0.0002 % | 0.023 % |
| 9-web airfoil | 0.0002 % | 0.0014 % |
| IEA-22 blade r=0.5 | 0.0011 % | 0.014 % |
| Station-15 (quad mesh) | 0.0010 % | — |

### RM replaces the 2-D solid for thin walls

The point of the RM shell is to **stand in for the much more expensive 2-D solid on thin-walled sections**.
Benchmarked on the **full 6×6** (every non-zero $C_{ij}$) against the solid as the two-cell $[-45]$ tube wall
thickens ($h/R$), RM holds the transverse shear to within ~5% where KL collapses:

| $h/R$ ($R/h$) | RM $GA_2$ | KL $GA_2$ | RM worst term | KL worst term |
|---|---|---|---|---|
| 0.08 (12.5) | −1.7 % | −14 % | 2.0 % | 14 % |
| 0.16 (6.25) | −3.5 % | −41 % | 4.3 % | 41 % |
| 0.25 (4.0) | −5.8 % | −62 % | 7.8 % | 62 % |

Across the **IEA-22 blade** (8 stations, $r/R=0.2\ldots0.9$, regenerated from windIO) RM and KL both stay
within ~5% of the 2-D solid on the full 6×6 — RM the better of the two on transverse shear. The
[tutorials](https://bagla0.github.io/OpenSG-TW/tutorials/) show the $R/h$ convergence plot and the spanwise
per-term tables.

## Quick start

```bash
pip install "jax[cpu]" pypardiso fenics-basix numpy scipy pyyaml numba matplotlib
```

```python
# 2-D solid Timoshenko 6x6 from a YAML (all inputs are bundled under examples/data/)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
C6 = compute_timo_from_yaml("examples/data/2d_yaml/mh104_solid.yaml")
print(C6)   # [EA, GA2, GA3, GJ, EI2, EI3] on the diagonal
```

Benchmark the full 6×6 against a VABS `.K`:

```bash
python -m opensg_jax.fe_jax.benchmark_vabs examples/data/2d_yaml/mh104_solid.yaml examples/data/benchmark/mh104.sg.K
```

The runnable [`examples/`](examples) (`1_/2_/3_get_beam_props_*`) reproduce the RM, KL and solid paths from
the command line; see the [tutorials](https://bagla0.github.io/OpenSG-TW/tutorials/) for executed,
step-by-step notebooks.

```{note}
On Windows, prepend the conda env to `PATH` (not just activate it) so the MKL/PARDISO DLLs resolve — see
[Installation](https://bagla0.github.io/OpenSG-TW/installation.html).
```

## Layout

```
opensg_jax/fe_jax/      JAX MSG engine (mirrors the OpenSG_2.0 / fea-in-jax architecture)
  solid_timo.py           2-D solid Timoshenko homogenizer  -> compute_timo_from_yaml()
  segment.py              2-D solid SG YAML reader (tri + quad)
  msg_*.py / msg_rm*.py   shell ABD, mesh, Hermite-C1 (KL) & Reissner-Mindlin operators
  transverse_shear.py     MSG plate transverse-shear (RM) block
  strip_RM.py / strip_Kirchhoff.py / strip_solid.py   RM / KL / solid drivers -> Timoshenko 6x6
  benchmark_vabs.py       full-6x6 JAX-vs-VABS .K comparison
examples/               1_..5_ unique-feature scripts (RM / KL / solid / airfoil driver / dehom)
  data/                   bundled inputs: 1d_yaml/ 2d_yaml/ benchmark/ xml/ windio/ (all tutorials & tests read here)
  benchmarks/             OML-stress validation, comparison & report scripts
scripts/                strip_*.yaml inputs, ABD verifiers, rm_research/ (RM derivation + studies)
tests/                  pytest regression: st12 / st15 / two-cell [-45] RM-vs-KL benchmarks
docs/                   Sphinx site (theory + executed tutorials) -> GitHub Pages
```

## Related

- **[OpenSG_io](https://github.com/bagla0/OpenSG_io)** — prepares OpenSG cross-section YAMLs from windIO,
  PreVABS XML, or OpenFAST inputs (the IEA-22 tutorial uses it).
- Built on the [OpenSG_2.0 / fea-in-jax](https://github.com/KeithBallard/fea-in-jax) JAX FEA architecture.

## Reference

W. Yu, D. H. Hodges, J. C. Ho, *Variational asymptotic beam sectional analysis — an updated version*,
Int. J. Eng. Sci. 59 (2012) 40–64. · W. Yu, *Mechanics of Structure Genome*, Purdue University.
