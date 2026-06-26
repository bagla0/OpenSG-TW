# Software framework

OpenSG-TW computes the Timoshenko $6\times6$ through **three independent branches** — the Reissner–Mindlin
shell, the Kirchhoff–Love shell, and the 2-D solid — that share one MSG core. Each branch is a driver
`.py` exposing a single entry function that returns the stiffness in the order
$[\,EA,\,GA_2,\,GA_3,\,GJ,\,EI_2,\,EI_3\,]$, so the three are drop-in interchangeable.

```{image} _img/architecture_timo.svg
:alt: OpenSG-TW three-branch Timoshenko architecture (RM, KL, 2-D solid)
:width: 100%
:align: center
```

| Branch | Driver `.py` | Entry function | Input | Engine / key methods |
|---|---|---|---|---|
| **Reissner–Mindlin shell** | `strip_RM.py` | `rm_timoshenko_6x6` | 1-D shell SG YAML | `msg_rm_timo.assemble_all`, `build_C_Psi`, `transverse_shear.transverse_shear_stiffness` (MITC) |
| **Kirchhoff–Love shell** | `gradient_kirchhoff.py` | `gradient_junction_kirchhoff` | 1-D shell SG YAML | `msg_hermite.solve_tw_from_yaml`, the $\Gamma_e/\Gamma_h/\Gamma_l$ Hermite-$C^1$ operators |
| **2-D solid** | `solid_timo.py` | `compute_timo_from_yaml` | 2-D solid SG YAML | `segment.read_solid_yaml` (tri + quad), `get_heterogeneous_C_matrix` (R_sig), KKT solve |

## The three branches

- **Reissner–Mindlin shell** — 5 DOF/node $[w_1,w_2,w_3,\omega_1,\omega_2]$ on a $C^0$ Lagrange contour;
  the independent director carries transverse shear, fixed against locking by **MITC** assumed strain. This
  is the branch that recovers $GA_2,GA_3$ and so **replaces the 2-D solid for thin walls** (see
  {doc}`theory/reissner_mindlin`).
- **Kirchhoff–Love shell** — cubic Hermite $C^1$ contour (3 DOF/node, value + arc-slope), no transverse
  shear; the wall curvatures carry the second contour derivatives and the $V_1$ condensation alone yields
  the beam shear $GA$ (see {doc}`theory/kirchhoff_love`).
- **2-D solid** — a filled $P_1$ tri/quad mesh with the full 3-D $6\times6$ material law; no thin-wall
  reduction, matched to **VABS** (see {doc}`theory/jax_solid`).

## Shared MSG core

All three branches reuse the same back-end: `msg_materials` (1-D structure-genome → plate ABD),
`msg_mesh` (line/area mesh, curvature), `msg_solver` (the saddle-point KKT solve and the Eq.85 projection),
and `timo_report` (the full-$6\times6$, every-$C_{ij}$ benchmarking). The full call signatures are in the
{doc}`api`.
