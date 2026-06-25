# st_oml_compare — st15 & st12 OML Timoshenko, shell models vs VABS-solid

Cross-check of the **generalized C1 penalty** (now default in the FEniCS-shell) on the st15 and st12
airfoil cross-sections, at the **OML reference**. Compares the three shell models —
**FEniCS-shell (FEniCS-Kirchhoff)**, **JAX-Kirchhoff (C1-Hermite)**, **JAX-RM** — against the
**FEniCS 2D-solid (VABS reference)**. Order `[EA, GA2, GA3, GJ, EI2, EI3]`.

## Result
The generalized penalty (`β·max|D-block|`) is **sane on st15 and st12** — no thin-wall flapwise
collapse; the FEniCS-shell tracks the JAX models closely. st15 agrees well across all three
(EA/EI2 ~1-2%, GA3 ~−9 to −16%); st12 is a *thicker* section so all three carry larger errors
(GJ +28-43%, EI3 +33%) — the thick-wall shell regime, not a penalty artifact.

## Files
- `st_comparison_table.txt` — % error of all three shell models vs solid, every term, st15 + st12.
- `data/C6_st{15,12}_solid.txt` — FEniCS 2D-solid (VABS reference).
- `data/C6_st{15,12}_fenics_shell.txt` — FEniCS-shell, generalized penalty, OML.
- `data/C6_st{15,12}_jax_kirch.txt`, `…_jax_rm.txt` — JAX-Kirchhoff / JAX-RM, OML.
- `st_jax.py` (Windows JAX), `st_fenics.py` (WSL FEniCS), `st_table.py` (table builder).
