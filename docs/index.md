# OpenSG-TW

**JAX Mechanics-of-Structure-Genome beam homogenization** — Kirchhoff–Love & Reissner–Mindlin shell and
2-D solid cross-sections, from an OpenSG YAML to the Timoshenko $6\times6$.

OpenSG-TW is the **thin-walled / pure-JAX extension of [OpenSG](https://github.com/wenbinyugroup/OpenSG)**.
It computes the $4\times4$ Euler–Bernoulli and $6\times6$ Timoshenko beam stiffness of an arbitrary composite
cross-section using the **Mechanics of Structure Genome (MSG)**, entirely in **JAX + `pypardiso`** (no
FEniCSx / MPI). The Timoshenko stiffness is returned in the order $[\,EA,\;GA_2,\;GA_3,\;GJ,\;EI_2,\;EI_3\,]$.

## Key features and capabilities

- **Three cross-sectional models, one solver back-end** — Kirchhoff–Love shell, Reissner–Mindlin shell
  (MITC transverse shear), and a 2-D solid.
- **VABS-matched 2-D solid** — reproduces the VABS Timoshenko $6\times6$ (diagonal *and* couplings) to a few
  parts in $10^6$ on triangle and quad meshes.
- **RM replaces the 2-D solid for thin walls** — the Reissner–Mindlin shell holds the **full $6\times6$** to
  within ~5 % of the solid where the cheaper KL shell loses the transverse shear $GA_2,GA_3$.
- **Pure JAX** — `basix` only tabulates the element basis/quadrature; `pypardiso` does the sparse
  saddle-point solve.
- **windIO / PreVABS front-ends** through the bundled [`OpenSG_io`](https://github.com/bagla0/OpenSG_io)
  converter (`third_party/OpenSG_io`).

| Model | Input | When to use |
|---|---|---|
| **Kirchhoff–Love (KL) shell** | 1-D shell SG YAML | thin walls, classical stiffness ($EA,EI,GJ$); Hermite-$C^1$ arc elements |
| **Reissner–Mindlin (RM) shell** | 1-D shell SG YAML | thin walls **with transverse shear** ($GA_2,GA_3$); MITC assumed-strain |
| **2-D solid** | 2-D solid SG YAML | thick / arbitrary sections; full 3-D fidelity, matched to **VABS** |

## At a glance

On the single-ply $[-45]$ tube the three models bracket the 2-D-solid reference and show exactly where each
lives — KL loses the transverse shear, RM recovers it, the solid matches VABS:

| term | KL %err | RM %err | 2-D solid vs VABS | meaning |
|---|---|---|---|---|
| $EA$ | +0.03 | −0.02 | −0.0002 % | axial |
| $GA_2,GA_3$ | **−44 / −69** | **−13 / −13** | −0.0002 % | transverse shear |
| $EI_2,EI_3$ | −16 / −19 | −12 / −12 | −0.0002 % | bending |

```{toctree}
:hidden:
:caption: Introduction

installation
tutorials/index
tutorials/rm_timo_from_yaml
tutorials/iea22_full_blade
tutorials/st15_solid_vs_shell
```

```{toctree}
:hidden:
:caption: Tapered 3-D segments

tutorials/taper_3dsg_segment
tutorials/taper_convergence_iso
tutorials/taper_convergence_m45
```

```{toctree}
:hidden:
:caption: Usage

architecture
theory/index
examples
references
api
```

```{toctree}
:hidden:
:caption: Backmatter

citing
license
```
