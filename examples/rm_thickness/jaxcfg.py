"""jaxcfg.py -- import this FIRST, before any other jax import in this package.

Everything in ``examples/rm_thickness`` runs in double precision: the 1-D SG solves have
a 3-dimensional rigid-body null space that is projected out, and the exact-elasticity
transfer matrices are products of matrix exponentials -- both lose all meaning in float32.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (must follow the x64 switch)

assert jnp.zeros(1).dtype == jnp.float64, "x64 did not take effect"

__all__ = ["jax", "jnp"]
