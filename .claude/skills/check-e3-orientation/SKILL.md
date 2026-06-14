---
name: check-e3-orientation
description: Verify the MSG plate-dehom through-thickness direction (e3, OML->IML) agrees with the YAML material orientation e3, per element, for the airfoil skin AND the web, under both the OML and IML reference. Use whenever changing the dehom reference, the IML offset, the layup direction, or before trusting a dehom / IML homogenization result on a new station.
---

# Check e3 / layup-direction consistency (plate dehom vs material orientation)

The MSG plate model stacks plies from the **OML** (outer surface, x = 0) to the
**IML** (inner surface, x = h).  Its through-thickness axis **e3 therefore points
OML -> IML (inward)**.  The YAML `elementOrientations` store the material e3 in
its cross-section components `(o[6], o[7])`.  These two MUST agree, element by
element, for the skin and the web.  If they are flipped, the dehom transverse
stresses (S33/S13/S23), the laminate->material rotation, and the IML offset
direction are all silently wrong (sign-flipped), and the shell/solid laminates no
longer represent the same physical layup.

## Invariants this enforces
1. **e3 direction:** `dot(material_e3, plate_dehom_e3) > 0` for every element.
2. **Layup order:** plies are added OML-first (`thick[0]`/`angles[0]` is the
   outer ply).  The 1Dshell and 2Dsolid YAMLs come from the same parent blade
   mesh, so their layups + orientations already agree — this guards the SHELL
   code from re-flipping them.
3. **IML reference:** keep e3 = material e3 by referencing the IML with a
   PARALLEL-AXIS shift (`msg_materials.shift_abd_reference`, z0 = thickness) and
   offsetting along the material e3 (`offset_oml_to_iml(elem_e3=...)`).  Do NOT
   reverse the layup to reach the IML — reversal flips e3 (dots -> -1).

## When to run
- After editing `offset_oml_to_iml`, the IML branch of `solve_tw_from_yaml`,
  `plate_stress_at_depth`, or the dehom laminate->global/material rotation.
- Before trusting a dehom or IML-reference result on a new cross-section.

## How to run
```
python .claude/skills/check-e3-orientation/check_e3.py <path/to/1Dshell_NN.yaml>
```
It prints, grouped by layup (so the web layup is visible separately):
`dot(material_e3, geometric OML->IML)` min/mean and any FLIPPED elements.
PASS = every dot > 0 (skin and web).  A web group with dots ~ -1 means the
web's stored orientation runs the opposite way and must be handled explicitly.
