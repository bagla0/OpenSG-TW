---
sd_hide_title: true
---

# OpenSG-TW

```{div} sd-text-center sd-fs-2 sd-font-weight-bold
OpenSG-TW
```

```{div} sd-text-center sd-fs-5 sd-text-muted
JAX Mechanics-of-Structure-Genome beam homogenization — Kirchhoff–Love & Reissner–Mindlin shell and 2-D solid cross-sections, from an OpenSG YAML to the Timoshenko 6×6.
```

---

**OpenSG-TW** computes the **4×4 Euler–Bernoulli** and **6×6 Timoshenko** beam stiffness of an arbitrary
composite cross-section using the **Mechanics of Structure Genome (MSG)**. It runs entirely in **JAX +
`pypardiso`** (no FEniCSx / MPI), and ships three cross-sectional models that share one solver back-end:

```{list-table}
:header-rows: 1
:widths: 22 30 48

* - Model
  - Input
  - When to use
* - **Kirchhoff–Love (KL) shell**
  - 1-D shell SG YAML
  - thin walls, classical stiffness (EA, EI, GJ); Hermite-$C^1$ arc elements
* - **Reissner–Mindlin (RM) shell**
  - 1-D shell SG YAML
  - thin walls **with transverse shear** ($GA_2,GA_3$); MITC assumed-strain
* - **2-D solid**
  - 2-D solid SG YAML
  - thick / arbitrary sections; full 3-D fidelity, matched to **VABS**
```

The Timoshenko stiffness is returned in the order $[\,EA,\;GA_2,\;GA_3,\;GJ,\;EI_2,\;EI_3\,]$.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Installation
:link: installation
:link-type: doc
Set up the environment and run your first cross-section.
:::

:::{grid-item-card} {octicon}`book` Theory
:link: theory/index
:link-type: doc
MSG structure genome, KL & RM shell models, and the 2-D solid Timoshenko derivation — in detail.
:::

:::{grid-item-card} {octicon}`beaker` Tutorials
:link: tutorials/index
:link-type: doc
Executed notebooks: RM, KL, and JAX-solid Timoshenko from a YAML, with orientation plots and %-error.
:::
::::

## At a glance

On the single-ply $[-45]$ tube, the three models bracket the 2-D solid reference and show exactly where each
model lives — KL loses the transverse shear, RM recovers it, the solid matches VABS:

```{list-table}
:header-rows: 1
:widths: 18 20 20 20 22

* - term
  - KL %err
  - RM %err
  - solid (mh104) vs VABS
  - meaning
* - $EA$
  - +0.03
  - −0.02
  - −0.000%
  - axial
* - $GA_2,GA_3$
  - **−44 / −69**
  - **−13 / −13**
  - −0.000%
  - transverse shear
* - $EI_2,EI_3$
  - −16 / −19
  - −12 / −12
  - −0.000%
  - bending
```

```{toctree}
:hidden:
:caption: Getting started

installation
```

```{toctree}
:hidden:
:caption: Background

theory/index
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/rm_timo_from_yaml
tutorials/kl_timo_from_yaml
tutorials/solid_timo_from_yaml
```

```{toctree}
:hidden:
:caption: Reference

examples
```
