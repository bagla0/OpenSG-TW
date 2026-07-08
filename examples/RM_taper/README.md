# RM_taper — reproducible tapered thin-walled beam examples

Supporting the paper *"Timoshenko Beam Modeling of Tapered Thin-Walled Composite Structures
Using the Reissner–Mindlin Model."* Three standalone scripts homogenize the equivalent-beam
Timoshenko `6×6` (`C^b`) of a tapered composite tube with the **6-DOF independent-ω₃
Reissner–Mindlin shell** and compare it, term by term, against a conforming **3-D FEniCS
solid** reference — for both the **boundary ring** and the **tapered segment**, at two wall
thicknesses (thin `t/R = 0.02`, thick `t/R = 0.20`), single `[-45°]` ply (`m45`).

```
python examples/RM_taper/circle.py    # circular tube
python examples/RM_taper/square.py    # square tube
python examples/RM_taper/ellipse.py   # webbed ellipse (multi-cell, 3 shear webs)
```

Input is a 1-D shell mesh YAML; output is the printed Timoshenko `6×6` and its `%`-error vs
the solid. Together the three scripts cover the **12 cases** of the paper (3 geometries ×
2 thicknesses × {boundary, taper}).

## Transverse-shear scheme (6-DOF everywhere)

| stage | thin `t/R ≤ 0.02` | thick `t/R > 0.02` |
|-------|-------------------|--------------------|
| tapered segment | MITC (`mitc4_both`) | full 2×2 Gauss |
| boundary ring   | γ₂₃-tie (`mitc4_g23`) | full |

MITC controls transverse-shear locking where the wall is thin; full integration is used on
the thick walls where locking is absent (and MITC would alias the drilling rotation on flat
walls / webs). Selected by `_rm_common.shear_for(stage, tR)`.

## Data

| geometry | shell mesh | solid ref (`L` = boundary, `seg` = taper) |
|----------|------------|--------------------------------------------|
| circle | `data/taper_study/meshes/shell_<regime>_m45_aR070.yaml` | `data/benchmark/taper_study_solid_m45.npz` |
| square | `data/taper_square/meshes/shell_<regime>_m45_aR070.yaml` | `data/benchmark/taper_square_solid_m45.npz` |
| ellipse | `data/rm_taper_ellipse/meshes/shell_<regime>_m45.yaml` | `data/benchmark/ellipse_solid_m45.npz` |

Solid references are pre-computed FEniCS 3-D-solid `6×6` stiffnesses (Pa). The shell solve is
run live by the scripts. `_rm_common.py` holds the shared solve/report logic
(`solve_boundary`, `solve_taper`, `report`) and the shear rule.

## Notes

- Circle and square reproduce the solid to a few percent on every term (diagonal and
  couplings).
- The webbed ellipse diagonal reproduces well (thin `GA₃` carries the transverse-shear
  locking onset; thick over-predicts `∝ t/R` from the T-junction material double-count). Its
  **off-diagonal couplings show sign flips** against the hex-solid reference — a known
  frame/ply-tilt convention mismatch between the webbed-hex solid and the shell (flagged in
  `rm_taper/instruction.md`); the diagonal, being sign-independent, is unaffected.
