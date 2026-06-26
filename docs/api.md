# API documentation

The homogenization engine lives in the `opensg_jax.fe_jax` package. Below are the three public drivers and
the shared MSG / reporting utilities, grouped the way OpenSG groups its `opensg.core` / `opensg.mesh` /
`opensg.utils` API. Every public function returns or operates on the Timoshenko $6\times6$ in the order
$[\,EA,\,GA_2,\,GA_3,\,GJ,\,EI_2,\,EI_3\,]$.

## 2-D solid

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.solid_timo
   :members:
   :undoc-members:

.. automodule:: opensg_jax.fe_jax.segment
   :members:
```

## Reissner–Mindlin shell

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.strip_RM
   :members:

.. automodule:: opensg_jax.fe_jax.msg_rm_timo
   :members:

.. automodule:: opensg_jax.fe_jax.transverse_shear
   :members:
```

## Kirchhoff–Love shell

```{eval-rst}
.. automodule:: opensg_jax.fe_jax.strip_Kirchhoff
   :members:

.. automodule:: opensg_jax.fe_jax.gradient_kirchhoff
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
