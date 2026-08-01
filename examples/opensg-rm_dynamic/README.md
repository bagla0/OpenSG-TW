# opensg-rm_dynamic — transient sandwich plate (OpenSG-RM vs Abaqus 3-D)

The realistic dynamic counterpart of the static Pagano validation: a simply
supported square **sandwich** plate under a **suddenly applied transverse
pulse**, the problem class whose transient response is the classical sandwich
benchmark of the literature. OpenSG-RM (Abaqus S4 shell carrying the 8×8 +
through-thickness recovery) is to be judged against the **Abaqus 3-D solid**
model of the same plate; the built-in composite-shell output (TSHR ≡ 0 for
S4, no σ33) is the industrial baseline it improves upon.

## Problem statement

| item | value |
|---|---|
| plate | square, a = b = 0.5 m, total thickness h = 0.05 m (a/h = 10) |
| layup | [0 / core / 0]: faces 0.1 h = 5 mm each, core 0.8 h = 40 mm |
| BCs | simply supported all edges (SS-1: w = 0 + tangential pin per edge) |
| load | uniformly distributed pressure q0 = 10 kPa applied suddenly at t = 0 and held (step load) on the TOP face |
| time | dt = 5×10⁻⁵ s, 500 increments (0.025 s ≈ several fundamental periods) |
| outputs | center deflection W(t); per-increment section forces at three 2×2 element patches (center, x-edge middle, y-edge middle) for the recovery; solid: through-thickness S at the matching element columns + W(t) |

Recovery quantities (what OpenSG-RM delivers and the shell baseline cannot):
transverse shear σ13/σ23 through the thickness at the edge-middle stations,
core/face interface σ33 and the in-plane set at the center, as profiles at
the response peak and as time histories.

## Configuration provenance (every attributed number is checkable)

- **Problem class + benchmark protocol**: transient response of simply
  supported composite **sandwich** plates under suddenly applied pulse loads,
  with center-deflection time histories as the comparison quantity —
  A.K. Nayak, R.A. Shenoi, S.S.J. Moy, *"Transient response of composite
  sandwich plates"*, **Composite Structures 64 (2004) 249–267**. (Their
  formulation is a refined higher-order FSDT; realistic marine-type sandwich:
  fibre-composite skins on structural foam core.)
- **Sandwich stiffness data** (identical to the statically validated
  `garg_caseC` archive case; printed in Garg et al., *Composite Structures
  305 (2023) 116551*, whose PDF is in the project library):
  faces E1 = 131 GPa, E2 = E3 = 10.34 GPa, G12 = G13 = 6.205 GPa,
  G23 = 3.0 GPa, ν = 0.22; core E = 0.5776 GPa, G = 0.1079 GPa, ν = 0.0025.
- **Dynamic-recovery protocol** (2-D solver supplies the plate solution per
  time step, the SG recovers the 3-D fields): the dynamic example of
  W. Yu, D.H. Hodges, V.V. Volovoi, *Computers & Structures 81 (2003)
  439–454*, sec. 6.2 (PDF in the project library).
- **Densities** (needed only for dynamics; OUR stated choices, not
  attributed): faces ρ = 1600 kg/m³ (the graphite/epoxy density used in the
  Yu 2003 dynamic example, Eq. 69) and core ρ = 100 kg/m³ (nominal
  structural-PVC-foam class, e.g. HEREX C70-series as used by
  Nayak–Shenoi–Moy). Section mass: 2·(0.005·1600) + 0.04·100 = 20 kg/m².

## Files

| file | content |
|---|---|
| `make_1dsg.py` | writes the through-thickness 1-D SG (`sandwich_sg.yaml`, 5-noded elements, one per layer incl. the core) + its mesh figure |
| `sandwich_sg.yaml`, `sandwich_sg.png` | the generated SG (committed) |
| `make_abaqus_dyn.py` | writes both decks from the SG |
| `sandwich_RM.inp` | Abaqus S4 plate: OpenSG-RM 8×8 general section + section density, SS-1 edges, step DLOAD, implicit dynamics; per-increment prints of U at the center node and SF/SM + COORD at the three recovery patches |
| `sandwich_SOLID.inp` | the 3-D benchmark: 20×20×12 C3D8I (2 elements per face sheet, 8 through the core), per-layer orientation/material, densities, same BCs/load/time; per-increment W(t), through-thickness S every 20th increment at the three element columns |

## Running (on the Abaqus machine)

```
abaqus job=sandwich_RM    interactive
abaqus job=sandwich_SOLID interactive
```

Copy both job `.dat` files back here; the recovery/post-processing script is
the next step of this folder (it patch-fits the SF/SM histories, recovers the
3-D fields per increment, and produces the three-method comparison against
the solid benchmark).
