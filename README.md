# OpenSG-TWJAX

**Thin-Walled Composite Beam Homogenization via MSG — JAX Implementation**

Computes the 4×4 Euler-Bernoulli and 6×6 Timoshenko beam stiffness matrices for composite
thin-walled cross-sections (airfoils, pipes, arbitrary profiles) using the
Mechanics of Structure Genome (MSG) theory. No FEniCSx or MPI required.

- **GitHub:** https://github.com/bagla0/OpenSG-TW
- **Branch:** `jax-msg-shell`
- **Formulation:** See [`docs/MSG_TW_Beam_Formulation.md`](docs/MSG_TW_Beam_Formulation.md)

---

## Architecture

```
OpenSG-TWJAX/
│
├── opensg_jax/fe_jax/              ← installable package  (pip install -e .)
│   │
│   ├── msg_materials.py            ── LAYER 1: Material & Plate Stiffness
│   │     build_stiffness_6x6()         6×6 orthotropic C from (E, G, ν)
│   │     rotation_6x6()                OpenSG R_σ rotation (Voigt, degrees)
│   │     rotated_stiffness_6x6()       R C Rᵀ for off-axis plies
│   │     compute_ABD_matrix()          1D through-thickness MSG SG → 6×6 ABD
│   │     compute_ABD_CLT()             Classical Laminate Theory (comparison)
│   │
│   ├── msg_mesh.py                 ── LAYER 2: YAML → FEM Mesh
│   │     load_yaml()                   Read OpenSG Shell_1DSG YAML format
│   │     order_mesh()                  CCW chain ordering + midside node insertion
│   │     compute_curvature()           κ₂₂ per element (circumscribed circle)
│   │
│   ├── msg_shell.py                ── LAYER 3: FEM Assembly + Solvers
│   │     gauss_legendre_01()           Gauss-Legendre quadrature on [0, 1]
│   │     quad_shape_functions()        3-node Lagrange N, N′, N″
│   │     compute_element_geometry()    Arc length L_e, tangent (ẋ₂, ẋ₃) per element
│   │     build_periodic_dof_map()      Merge first/last node (closed section)
│   │     compress_dof_map()            Full → unique DOF renumbering
│   │     assemble_system_matrices()    JAX vmap energy autodiff → D_hh, D_he, D_ee,
│   │                                   D_ll, D_hl, D_le
│   │     build_lagrange_constraints()  4 rigid-body integral constraints C (4 × N)
│   │     build_psi_matrix()            N × 4 null-space basis (3 transl. + twist)
│   │     solve_fluctuation_field()     KKT solve → V0, D1  (pypardiso PARDISO)
│   │     prepare_v1_rhs()              V1 RHS with Psi/Dc null-space projection
│   │     finalize_v1_and_compute_deff()  V1 projection → 6×6 Timoshenko S
│   │
│   └── __init__.py                 ── re-exports all three layers;
│                                      legacy FEniCSx imports in try/except
│
├── examples/
│   └── run_airfoil_cross_section.py   Full pipeline driver  (YAML → 6×6 S)
│
├── tests/
│   ├── conftest.py                    yaml_1dshell_0 … yaml_1dshell_29 fixtures
│   ├── test_pipe_validation.py        Analytical pipe benchmark (6 assertions)
│   ├── test_1dshell_stiffness.py      1Dshell_0 regression  (8 tests, 0.5 % tol.)
│   └── data/  1Dshell_0.yaml … 1Dshell_29.yaml   (30 OpenSG airfoil cases)
│
├── docs/
│   └── MSG_TW_Beam_Formulation.md     Full MSG shell-TW variational formulation
│
└── CLAUDE.md                          Auto-loaded project context for Claude agents
```

---

## Data Flow

```
YAML file
   │
   ▼  load_yaml()
nodes_3d, elements, material_db, layup_db, elem_to_layup
   │
   ├──► compute_ABD_matrix()  ──► 6×6 ABD per layup   [quadratic 1D through-thickness SG]
   │
   ▼  order_mesh() + compute_curvature()
nodes_2d (with midside nodes), cells (3-node), κ₂₂ per element
   │
   ▼  assemble_system_matrices()          [JAX vmap, energy-based autodiff]
D_hh (N×N),  D_he (N×4),  D_ee (4×4)
D_ll (N×N),  D_hl (N×N),  D_le (N×4)
   │
   ├──► build_lagrange_constraints()  →  C  (4 × N)
   ├──► build_psi_matrix()            →  Ψ  (N × 4)
   │
   ▼  solve_fluctuation_field()       [pypardiso KKT:  [D_hh  Cᵀ; C  0] V = [-D_he; 0]]
V0 (N×4),   C_eff = D_ee + V0ᵀ D_he   →  4×4 Euler-Bernoulli stiffness
   │
   ▼  prepare_v1_rhs() + pypardiso.spsolve()   [reuse same KKT matrix]
V1 (N×4)
   │
   ▼  finalize_v1_and_compute_deff()
6×6 Timoshenko S   [EA,  GA₁₂,  GA₁₃,  GJ,  EI₂,  EI₃]
```

---

## Quick Start

```powershell
# Windows — prepend conda env to PATH for MKL DLLs
$env:PATH = "C:\conda_envs\opensg_2_0_env;...;" + $env:PATH
$env:PYTHONPATH = "path\to\OpenSG-TWJAX\opensg_jax"

# Run on any Shell_1DSG YAML
python examples/run_airfoil_cross_section.py tests/data/1Dshell_0.yaml

# Run test suite
python -m pytest tests/ -v
```

---

## Dependencies

| Package | Role |
|---------|------|
| `jax[cpu] >= 0.4` | Energy autodiff, vmap element assembly |
| `pypardiso >= 0.4` | Intel MKL PARDISO sparse direct solver (KKT) |
| `numpy`, `scipy` | COO/CSR sparse matrix construction |
| `pyyaml` | OpenSG YAML input parsing |
| `pytest` | Test suite |

Full environment: [`environment_jax.yml`](environment_jax.yml)

---

## Key Design Choices

- **No FEniCSx / MPI** — pure JAX + scipy sparse + pypardiso (Intel MKL PARDISO)
- **Quadratic Lagrange C0** along the cross-section arc — 3 DOFs/node [w₁, w₂, w₃],
  machine-precision ABD (vs. 5.6 % error with linear elements)
- **Energy-based autodiff** via `jax.hessian` / `jax.jacfwd` — no hand-coded stiffness matrices
- **Two-level homogenization**: (1) through-thickness 1D SG → ABD; (2) cross-section 1D SG → 6×6 S
- **Custom mesh generation** (e.g., pipe cross-section) lives outside `fe_jax` in user scripts

---

## Reference

Yu, W., Hodges, D. H., & Ho, J. C. (2012). *Variational asymptotic beam sectional analysis —
an updated version*. International Journal of Engineering Science, 59, 40–64.

Wenbin Yu, *Mechanics of Structure Genome*, Purdue University.
