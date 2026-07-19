# Shell-based linear buckling (RM-OpenSG)

All buckling code, data, tests, and figures live here. The buckling **solver** is also mirrored into the
core package at `opensg_jax/fe_jax/shell_buckling.py` for `import opensg_jax` users; this folder holds the
authoritative dev copy plus every driver/benchmark/test.

## Layout
```
buckling/
├── shell_buckling.py            core solver: MITC4 facet shell + geometric stiffness + eigsh
│                                (K + lambda*KG) phi = 0 ; also solve_static + element_membrane_N
├── cyl_buckling_bench.py        isotropic cylinder: analytical | JAX-FEA | RM-OpenSG (3-way)
├── blade_buckling.py            IEA-22 blade under traction: JAX-FEA vs RM-OpenSG  [WIP]
├── render_cyl_modes.py          cylinder mode-shape + membrane-N figures -> fig/
├── SHELL_BUCKLING_FORMULATION.md  Abaqus/Ansys formulation grounding + benchmark note
├── tests/
│   ├── test_cyl_bc.py           SS / clamp-clamp / clamp-free cylinder vs classical
│   └── dbg_cyl.py               BC + mesh + drilling diagnostic sweep
├── data/                        ring yaml, ABD yaml, cyl_bench.{json,npz}, caches
└── fig/                         rendered PNGs
```

## Run (server: opensg_2_0)
```
cd examples/buckling
python shell_buckling.py all          # self-tests: plate 0.9996, SS3 cyl 0.952, cantilever 0.379
python cyl_buckling_bench.py          # 3-way cylinder benchmark -> data/cyl_bench.json,.npz
python render_cyl_modes.py            # -> fig/cyl_modes.png, fig/cyl_N.png
python tests/test_cyl_bc.py           # BC study
```

## Validation status
| case | result |
|------|--------|
| SS flat plate            | 0.9996 x (4 pi^2 D/a^2) — exact |
| SS3 isotropic cylinder   | 0.9524 x classical (mode n=6) |
| clamp-free cantilever    | 0.379  (free-edge boundary layer, correct physics) |
| RM-OpenSG vs JAX-FEA cyl | N_cr identical to 5 sig figs, MAC = 1.0000 |

The pre-buckling membrane N enters KG two ways: **JAX-FEA** (direct shell static solve) and **RM-OpenSG**
(RM cross-section homogenization + two-step dehomogenization). Both feed the same eigensolver.
