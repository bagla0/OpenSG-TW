# `examples/data/` — bundled cross-section data

Every tutorial ({doc}`../../docs/tutorials/index`) and benchmark test in this repo reads its inputs
from here, so a fresh clone runs end-to-end with **no external/absolute paths**. Layout:

| Folder | Contents |
|---|---|
| `1d_yaml/` | **1-D shell SG** YAMLs (`sections`/`layup`, line mesh) — input to the RM & KL solvers |
| `2d_yaml/` | **2-D solid SG** YAMLs (filled mesh + `elementOrientations` + `materials`) — input to the JAX solid solver |
| `benchmark/` | reference stiffness: VABS `*.sg.K` / `*.K`, or a bare 6×6 `*_solid_ref.txt` (FEniCS 2-D solid) |
| `xml/` | **PreVABS** source XML (geometry + layup + materials) that meshes into the `2d_yaml/` solids |
| `windio/` | the source **windIO** blade (`IEA-22-280-RWT.yaml`) that `OpenSG_io` converts into the IEA-22 YAMLs |

## Cases

| Case | shell (1d) | solid (2d) | benchmark | used by |
|---|---|---|---|---|
| **tube_m45** — single-ply `[-45]` tube (thesis §3.1.4) | `tube_m45_shell.yaml` | `tube_m45_solid.yaml` | `tube_m45_solid_ref.txt` | RM, KL tutorials |
| **mh104** — MH-104 airfoil | `mh104_shell.yaml` | `mh104_solid.yaml` | `mh104.sg.K` | solid tutorial |
| **iea22_r050** — IEA-22-280-RWT blade, r/R=0.5 | `iea22_r050_shell.yaml` | `iea22_r050_solid.yaml` | `iea22_r050.sg.K` | IEA-22 tutorial |
| **tube2cell_m45** — two-cell `[-45]` tube (ASC) | `tube2cell_m45_shell.yaml` | `tube2cell_m45_solid.yaml` | `tube2cell_m45_solid_ref.txt` | two-cell tutorial + `test_twocell_m45_benchmark.py` |
| **st15** — station-15 (thick web) | `st15_shell.yaml` | `st15_solid.yaml` (quad) | `st15_vabs.K` | st15 tutorial + `test_st15_benchmark.py` |
| **st12** — station-12 | `st12_shell.yaml` | `st12_solid.yaml` | `st12_solid_ref.txt` | `test_st12_benchmark.py` |

## Provenance

```
PreVABS XML (xml/<case>/)  --PreVABS mesh-->  2-D solid YAML (2d_yaml/)  --VABS-->  benchmark .K
windIO blade (windio/)     --OpenSG_io     -->  1-D shell + 2-D solid YAML  (iea22_r050)
```

Order in every benchmark is `[EA, GA2, GA3, GJ, EI2, EI3]` = VABS `[ext, shear2, shear3, twist, bend2, bend3]`.
The larger `2d_yaml/` meshes are full 2-D solid cross-sections (≈1–4 MB each); the `1d_yaml/` shells are small.
