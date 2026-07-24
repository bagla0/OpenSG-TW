# API documentation

The homogenization engine lives in the `opensg_jax.fe_jax` package. Below are the public drivers and
the shared MSG / reporting utilities, grouped the way OpenSG groups its `opensg.core` / `opensg.mesh` /
`opensg.utils` API. Every public function returns or operates on the Timoshenko $6\times6$ in the order
$[\,EA,\,GA_2,\,GA_3,\,GJ,\,EI_2,\,EI_3\,]$.

The RM cross-section drivers exercised by the paper and by the
{doc}`tutorials/iea_r020_homo_dehom` / {doc}`tutorials/iea_spanwise` tutorials are *not* inside the
installed package — they are top-level modules under `examples/TW-paper/xsec_paper/` and
`mitc_rm_segment/`, documented in {ref}`their own section <rm-xsec-drivers>` at the end of this page.

## 2-D solid

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.solid_timo
   :members:
   :undoc-members:

.. automodule:: opensg_jax.fe_jax.segment
   :members:
```

## Reissner–Mindlin shell

The theory behind these modules is in {doc}`theory/reissner_mindlin`.

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.strip_RM
   :members:

.. automodule:: opensg_jax.fe_jax.msg_rm
   :members:

.. automodule:: opensg_jax.fe_jax.msg_rm_timo
   :members:

.. automodule:: opensg_jax.fe_jax.msg_transverse_shear
   :members:

.. automodule:: opensg_jax.fe_jax.transverse_shear
   :members:
```

## Kirchhoff–Love shell

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.strip_Kirchhoff
   :members:

.. automodule:: opensg_jax.fe_jax.msg_hermite
   :members:

.. automodule:: opensg_jax.fe_jax.gradient_kirchhoff
   :members:
```

## Dehomogenization (3-D recovery)

The inverse of homogenization: given the converged warping and a beam force/moment resultant, recover
the pointwise 3-D field. Step 1 takes the beam strains to the generalized shell strains along the
contour; step 2 reuses the *same* through-thickness plate SG that produced the ABD, so the recovery is
energy-consistent with the homogenization. This is the Kirchhoff–Love (Hermite) branch; the
RM-consistent counterpart is `dehom_rm`, in the {ref}`RM cross-section drivers <rm-xsec-drivers>`
section below.

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.msg_dehom
   :members:
```

## Tapered 3-D segment (solid)

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.solid_taper
   :members:
```

## Shared MSG core

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.msg_materials
   :members:

.. automodule:: opensg_jax.fe_jax.msg_mesh
   :members:

.. automodule:: opensg_jax.fe_jax.msg_solver
   :members:
```

## Reporting & benchmarking

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.timo_report
   :members:

.. automodule:: opensg_jax.fe_jax.benchmark_vabs
   :members:

.. automodule:: opensg_jax.fe_jax.orient_plot
   :members:

.. automodule:: opensg_jax.fe_jax.blade_viz
   :members:
```

(rm-xsec-drivers)=
## RM cross-section drivers

These are the modules the Composites Part B RM cross-section paper and the two IEA tutorials
({doc}`tutorials/iea_r020_homo_dehom`, {doc}`tutorials/iea_spanwise`) actually call. They implement the
**6-DOF ring** element — nodal DOF $[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$ with the drilling
rotation $\omega_3$ kept as an *independent* DOF constrained by an element-wise Lagrange multiplier,
and the transverse shear $\gamma_{23}$ tied by an MITC assumed-strain scheme (`shear="mitc4_g23"`).
That is the formulation that passes cleanly through the web/skin T-junctions where a
drilling-eliminated element degenerates on wall-parallel webs; the older drilling-eliminated 5-DOF
strip driver (`strip_RM`, DOF $[w_1,w_2,w_3,\omega_1,\omega_2]$) is listed under
*Reissner–Mindlin shell* above and remains available. See {doc}`theory/reissner_mindlin` for the
derivation.

Two conventions run through every entry point below:

* the **reference surface** is a single argument (`ref` / `center_ref` / `fraction`) that propagates to
  the contour geometry, the wall constitutive law *and* the recovery depth — `0.5` = center
  (mid-surface, what the paper adopts), `0.0` = OML, `1.0` = IML;
* the **wall law** is the MSG-RM $8\times8$ plate stiffness

  $$
  \begin{bmatrix} \bA & \bB & 0 \\ \bB & \bD & 0 \\ 0 & 0 & \mathbf{G} \end{bmatrix},
  $$

  with the $2\times2$ transverse-shear block $\mathbf{G}$ obtained from the MSG/VAM Reissner–Mindlin plate route
  (zeroth-order plate SG $\rightarrow$ classical ABD, gradient energy, least-squares projection onto
  the Reissner form) rather than an assumed shear-flow closure — `msg_rm_plate.rm_plate_msg`.

These modules import each other by bare module name, so `docs/conf.py` puts
`examples/TW-paper/xsec_paper/` and `mitc_rm_segment/` on `sys.path`; several helpers are
signature-only (no docstring) and are listed here with `:undoc-members:` so the full entry-point set is
visible.

### User-facing wrappers — `examples/TW-paper/xsec_paper/`

```{eval-rst}
.. automodule:: xsec_5v6_master
   :members:
   :undoc-members:

.. automodule:: oml_ring
   :members:
   :undoc-members:

.. automodule:: msg_rm_plate
   :members:
   :undoc-members:

.. automodule:: dehom_rm
   :members:
   :undoc-members:
```

### Ring element and solver — `mitc_rm_segment/`

```{eval-rst}
.. automodule:: run_ring_indep
   :members:
   :undoc-members:
   :exclude-members: main

.. automodule:: segment_indep
   :members:
   :undoc-members:
```
