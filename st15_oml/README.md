# st15_oml — st15 OML Timoshenko: shell models vs VABS-solid (+ orientation check)

st15 airfoil cross-section at the **OML reference**: every **nonzero** Timoshenko stiffness term, the
**solid (VABS) value**, and the **% error** of the three shell models — FEniCS-shell (FE-Kirchhoff),
JAX-Kirchhoff (C1-Hermite), JAX-RM. Order `[EA, GA2, GA3, GJ, EI2, EI3]`.

## Table — `st15_oml_table.txt`
Diagonal: EA +2%, GA2 −3 to −10%, GA3 −9 to −16%, GJ −2 to −5%, EI2 +0.5 to +1%, EI3 +14% — the JAX
and FEniCS shells all track the solid well. Nonzero couplings (C23, C24, C34, C15, C16, C56) are
included with their solid values. Two stand out and are **not** orientation artifacts:
- `C15 (EA-EI2)` is +95% for **all three** models → a shell-model coupling limitation.
- `C24 (GA2-GJ)` is FE-shell-only (+149% vs JAX −13%) → a FEniCS-shell artifact.

## Orientation check — `figures/`
- `st15_shell_orient.png` — the shell e1/e2/e3: e1=(0,0,1) everywhere, e2⊥e3, right-handed,
  webs consistent (−x).  The shell frame is **self-consistent**.
- `st15_orient_full.png` / `_e3zoom.png` — shell vs solid: e2·e2 = e3·e3 mean +0.65, with 11/64
  elements reading flipped vs the solid.  Because the **diagonal terms still match the solid**
  (EA +2%, GA3 −16%, EI2 +1%), those flips fall in the [0/0/0] web / symmetric regions where the
  frame flip is ABD-invariant (cosmetic), not in the load-bearing off-axis skin.

## Files
- `st15_oml_table.txt`, `st15_table.py` (builder).
- `figures/` — orientation images.
- 6×6 data: `../st_oml_compare/data/C6_st15_*.txt`.
