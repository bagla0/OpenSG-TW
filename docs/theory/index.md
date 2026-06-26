# Theory

OpenSG-TW is built on the **Mechanics of Structure Genome (MSG)** — a unified variational
dimensional-reduction framework (Yu and co-workers) that replaces *ad hoc* cross-sectional
assumptions with a single principle: minimize the difference between the strain energy of the original
3-D (or 2-D) heterogeneous structure and that of an equivalent 1-D beam, over a periodic/free
**Structure Gene (SG)**.

The four pages below build up the theory used by the three solvers in this toolkit.

```{toctree}
:maxdepth: 1

msg_structure_genome
kirchhoff_love
reissner_mindlin
jax_solid
```

## How the pieces fit

```{list-table}
:header-rows: 1
:widths: 26 22 52

* - Page
  - SG dimension
  - Content
* - {doc}`msg_structure_genome`
  - —
  - the MSG variational statement, the warping field, the EB→Timoshenko condensation shared by all solvers
* - {doc}`kirchhoff_love`
  - 1-D arc (shell)
  - classical normal-stays-normal shell; Hermite-$C^1$ arc elements; ABD plate constitutive law
* - {doc}`reissner_mindlin`
  - 1-D arc (shell)
  - independent director ⇒ transverse shear; MITC selective assumed-strain; the $GA_2,GA_3$ recovery
* - {doc}`jax_solid`
  - 2-D section (solid)
  - full 3-D constitutive law on the filled mesh; rigid-body (Eq. 85) projection; VABS-matched 6×6
```

All three return the Timoshenko stiffness in the order
$\big[\,EA,\;GA_2,\;GA_3,\;GJ,\;EI_2,\;EI_3\,\big]$, conjugate to the generalized strain
$\big[\,\gamma_{11},\;2\gamma_{12},\;2\gamma_{13},\;\kappa_1,\;\kappa_2,\;\kappa_3\,\big]$
(axial stretch, two transverse shears, twist, two bending curvatures).
