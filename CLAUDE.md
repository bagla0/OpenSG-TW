# OpenSG-TWJAX — Claude Agent Context

## Project Overview

**OpenSG-TWJAX** is a JAX-based thin-walled beam homogenization toolkit built on the Mechanics of Structure Genome (MSG) theory. It computes the 4×4 Euler-Bernoulli and 6×6 Timoshenko stiffness matrices for composite thin-walled cross-sections (airfoils, pipes, arbitrary profiles) without requiring FEniCSx or MPI.

- **GitHub:** https://github.com/bagla0/OpenSG-TW
- **Primary branch:** `jax-msg-shell`
- **Conda env:** `C:\conda_envs\opensg_2_0_env` (Python 3.12, JAX CPU, pypardiso)

## Key Files

| File | Role |
|------|------|
| `opensg_jax/fe_jax/msg_materials.py` | 6×6 orthotropic stiffness, R_sig rotation, through-thickness ABD via quadratic 1D SG |
| `opensg_jax/fe_jax/msg_mesh.py` | YAML loader, CCW mesh ordering, midside-node insertion, curvature computation |
| `opensg_jax/fe_jax/msg_shell.py` | Full Kirchhoff shell MSG solve — quadratic Lagrange C0 elements, periodic BCs, KKT solve |
| `opensg_jax/fe_jax/__init__.py` | Exports all msg_* symbols; legacy FEniCSx imports guarded in try/except |
| `examples/4_run_airfoil_cross_section.py` | Top-level driver: loads YAML → ABD → mesh → MSG solve → 6×6 stiffness |
| `Shell_Hermite.py` | Standalone script (same algorithm, not packaged) |
| `run_airfoil_cross_section.py` | Standalone script version of the driver |

## Theory Summary

**MSG 1D Shell SG (through-thickness ABD):**
- 3-node quadratic Lagrange elements over each ply layer
- Variational principle: `D_eff = D_ee + V0^T @ F_load` where `V0 = K^{-1}(-F_load)`
- **Sign rule:** `D1 = V0.T @ F_load` (NOT `-F_load`) — D1 < 0, reduces stiffness

**MSG Kirchhoff Shell Cross-Section:**
- Quadratic Lagrange C0 elements: 3 DOFs/node [w1, w2, w3], 3 nodes/element (corner–midside–corner)
- Total nodes: `2*n_elem + 1` (even indices = corners, odd = midside)
- DOF layout per element: `NDOFS_ELEM=9`, idx_w1=[0,3,6], idx_w2=[1,4,7], idx_w3=[2,5,8]
- 4-pt Gauss quadrature over [0,1]
- Periodic BCs enforced via `build_periodic_dof_map` + Lagrange multipliers (KKT system)
- Timoshenko enhancement: `G = (Q^T A^{-1} C A^{-1} Q)^{-1}`, solved via pypardiso PARDISO

**Gamma operators (6-component Voigt notation):**
```
gamma_h(v): [0, 0, dv3/dx, dv2/dx, dv1/dx, 0]
gamma_e:    rows mapping [eps11, eps22, 2eps12, 2eps13, 2eps23, eps33] to generalized strains
```

**OpenSG R_sig rotation:** `rotation_6x6(theta_deg)` takes degrees, converts internally.

## YAML Format (OpenSG Shell_1DSG)

- Nodes: 3D coordinates (z=0 for 2D cross-sections), 1-indexed in elements
- Elements: 1-indexed `[n1 n2]` pairs (space-separated, no commas)
- Sets: element sets mapping elements to layup names
- Sections: layup as `[mat_name, thickness, angle_deg]` per ply
- Materials: E[3], G[3], nu[3] lists (orthotropic engineering constants)
- `elementOrientations`: 9-component rotation matrices per element (used for curvature)

**Test data:** `tests/data/1Dshell_0.yaml` through `1Dshell_29.yaml` — all 30 variants of a composite wind-turbine blade airfoil cross-section.

## Running Code

```powershell
# Set up environment (Windows — must prepend conda env to PATH for DLLs)
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "C:\conda_envs\opensg_2_0_env;C:\conda_envs\opensg_2_0_env\Library\mingw-w64\bin;C:\conda_envs\opensg_2_0_env\Library\usr\bin;C:\conda_envs\opensg_2_0_env\Library\bin;C:\conda_envs\opensg_2_0_env\Scripts;" + $env:PATH

# Run standalone example
& "C:\conda_envs\opensg_2_0_env\python.exe" examples/4_run_airfoil_cross_section.py tests/data/1Dshell_0.yaml

# Run tests
& "C:\conda_envs\opensg_2_0_env\python.exe" -m pytest tests/ -v
```

## Coding Conventions

- **Notation:** Match MSG theory exactly — use `D_ee`, `D_hh`, `D_he`, `Psi`, `V0`, `Ceff`; never rename to generic `K`, `f`, `x`
- **Precision:** Always `jax.config.update("jax_enable_x64", True)` (set in `__init__.py`)
- **Elements:** Quadratic Lagrange (3-node) everywhere — do NOT revert to linear
- **Sign convention:** `D1 = V0.T @ F_load` (positive F_load, not negated)
- **No FEniCSx imports** in msg_* files — legacy imports are guarded in `__init__.py`
- **Validate** new features against analytical solutions (pipe: closed-form Euler-Bernoulli + Timoshenko)
- **No comments** explaining what the code does — only add comments for non-obvious WHY (hidden invariants, sign conventions, workarounds)

## Known Issues / Gotchas

- `sparse_linear_solve.py` imports `flax` which is not in `opensg_2_0_env` — this is why legacy imports are in `try/except`
- Windows DLL loading requires the conda env to be prepended to `$env:PATH` (not just activating the env)
- YAML parser requires `_parse_row()` helper because values are space-separated (not comma-separated) inside brackets
- `compute_ABD_matrix` returns `(ABD, CLT)` tuple — always unpack both even if CLT is unused
- Periodic DOF map: first and last node on closed cross-section are merged; `n_unique = n_nodes - 1`

## gh CLI Location

Portable gh CLI (GitHub authentication as bagla0):
```
C:\Users\bagla0\AppData\Local\Temp\gh_cli\bin\gh.exe
```
