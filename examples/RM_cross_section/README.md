# MSG-RM cross-section — the minimal example

`rm_core.py` is the **smallest complete MSG-RM workflow**: read a 1-D shell structure-gene
YAML, homogenize it to a Timoshenko $6\times6$, and dehomogenize a beam load back to
pointwise 3-D stress and displacement. No plotting, no VABS comparison, no masking — just
the calls that matter.

```bash
python examples/RM_cross_section/rm_core.py
```

Everything is read from data committed in this repository and every path is resolved
relative to the script, so it runs from a fresh clone at any location.

## The three steps

| Step | Call | Lives in |
|---|---|---|
| 1 · homogenize | `ring_6dof(load_ring(yaml))` → Timoshenko 6×6 `[EA,GA2,GA3,GJ,EI2,EI3]` | `examples/TW-paper/xsec_paper/xsec_5v6_master.py`, solved by `mitc_rm_segment/run_ring_indep.py::ring_indep` |
| 2 · wall law | `rm_plate_msg(thk, ang, mat, mdb, z_ref=…)` → `A6` (6×6 ABD) + `G_msg` (2×2), stacked into the 8×8 | `examples/TW-paper/xsec_paper/msg_rm_plate.py` |
| 3 · dehomogenize | `build_rm_bundle(yaml)` then `stress_at_points` / `disp_at_points` | `examples/TW-paper/xsec_paper/dehom_rm.py` |

## Expected output (IEA-22, `r/R = 0.2`, mid-surface reference)

```
1. HOMOGENIZATION  (~1.3 s)
   EA   =  2.76606e+10      GJ   =  2.43806e+09
   GA2  =  7.18650e+08      EI2  =  3.51439e+10
   GA3  =  4.21763e+08      EI3  =  6.81297e+10

3. DEHOMOGENIZATION  (~13 s, bundle build dominates)
   stress in Voigt [S11,S22,S33,S23,S13,S12], material frame
```

## Notes

- **`jax.config.update("jax_enable_x64", True)` is mandatory** — the KKT warping solve is float64.
- The reference surface (`center` / `oml` / `iml`) is read from the YAML's own `reference`
  field by `build_rm_bundle`; it propagates to the contour geometry, the wall law `z_ref`, and
  the recovery depth. `load_ring` does **not** read that field (it assumes centre) — keep the
  two consistent.
- `disp_at_points` returns the **warping only**. A total displacement needs the beam
  kinematics `u = u_g + C(w + r) − r`; see `docs/tutorials/iea_spanwise.py` for that step.
- The beam force must be in **VABS order** `[F1, F2, F3, M1, M2, M3]`; it is asserted, never
  checked, so a differently-ordered vector silently produces wrong stress.

## Going further

- theory — {doc}`docs/theory/reissner_mindlin.md`, {doc}`docs/theory/dehomogenization.md`
- the full validated study (3 recovery paths vs VABS) — `docs/tutorials/iea_r020_homo_dehom.ipynb`
- 51-station spanwise — `docs/tutorials/iea_spanwise.ipynb`
- the constitutive law on its own — run `python examples/TW-paper/xsec_paper/msg_rm_plate.py`,
  whose `__main__` validates isotropic → $G = \tfrac56 Gh$ exactly.
