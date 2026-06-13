import jax

jax.config.update("jax_enable_x64", True)

# Legacy FEniCSx-dependent modules (require flax, petsc4py, dolfinx)
try:
    from .np_types import *
    from .basis_quadrature import *
    from .fea import *
    from .linear_elasticity import *
    from .hyperelasticity import *
    from .profiling import *
    from .setup import *
    from .utils import *
    from .sc_to_msh import *
    from .sparse_matrix import *
    from .sparse_linear_solve import *
    from .constraints import *
    from .constraint_system import *
    from .boundary_conditions import *
    #from .periodic_dofmap import *
    #from .multiscale import *
    #from .fiber_mechanics import *
except ImportError:
    pass  # FEniCSx / flax not installed — MSG shell modules still available

# MSG Shell Timoshenko beam homogenization (quadratic Lagrange elements)
from .msg_materials import (
    build_stiffness_6x6,
    rotation_6x6,
    rotated_stiffness_6x6,
    compute_ABD_matrix,
    compute_ABD_CLT,
)
from .msg_mesh import (
    load_yaml,
    order_mesh,
    compute_curvature,
)
from .msg_shell import (
    gauss_legendre_01,
    quad_shape_functions,
    make_pipe_mesh,
    compute_element_geometry,
    build_periodic_dof_map,
    compress_dof_map,
    assemble_system_matrices,
    build_lagrange_constraints,
    build_psi_matrix,
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)

