# oml_mh104_4way — mh104 OML Timoshenko, 4-way comparison (f = 0.1 … 0.75)

The full 4-model OML comparison for the **mh104** airfoil: **JAX-Kirchhoff (C1-Hermite)**,
**JAX-Reissner-Mindlin (C0)**, **FEniCS-shell**, and the VABS-validated **FEniCS 2D-solid**, over
wall-thickness factor `f ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.75}`. Order `[EA, GA2, GA3, GJ, EI2, EI3]`;
reference = quarter-chord.

**FEniCS-shell now uses the GENERALIZED C1 penalty** (`penalty = β·max|ABD[3:,3:]|`, scaled to the
laminate bending stiffness ≈ E·t³ instead of the old fixed `[1.2e13,2.5e13]` window). This removed
the thin-wall flapwise collapse: FEniCS-shell GA3 went from **−94% → −5%** at f=0.2 (EI2 −85% → +3%
at f=0.1). The tube is unchanged (its max_D → penalty ≈ the old validated 1.3e13).

## Plots (significant terms; couplings stuck in the ~1e6 range dropped)
Each panel: JAX-Kirchhoff (blue ●), JAX-RM (red ▲), FEniCS-shell (green ◆), FEniCS-solid (black ■).
Diagonal panels labelled `EA (C11)…`; coupling panels `C14: EA-GJ` etc. Dashed line = **f=0.3**
thin/thick boundary (spar h/H≈0.08). **Error = signed RELATIVE % error** `(model−solid)/solid`.
- `figures/oml_abs_diagonal.png` — all 6 stiffnesses (absolute).
- `figures/oml_abs_coupling.png` — couplings C14, C15, C16, C24, C25 (|solid|≥1e7).
- `figures/oml_pcterr_diagonal.png`, `oml_pcterr_coupling.png` — the same as % error.

## Headline
- **JAX-Kirchhoff and JAX-RM bracket the solid** on the shear terms (Kirchhoff over, RM under).
- **FEniCS-shell tracks the solid well** with the generalized penalty: GA3 within −13%→+40%, EI2
  +1%→+13% over the whole sweep (was a collapse at thin walls). EA/GJ/EI3 agree for all three.
- All errors grow past **f=0.3** (the thin/thick boundary), as expected for thin-shell models.

## Data
- `data/C6_shell_jax_OML_f0NN.txt` — JAX-Kirchhoff;  `C6_shell_rm_OML_f0NN.txt` — JAX-RM.
- `data/C6_fenics_shell_OML_f0NN.txt` — FEniCS-shell (generalized penalty).
- `data/C6_solid_f0NN.txt` — FEniCS solid reference.
- `data/timo_full_comparison.txt` — every term, all 4 models, % errors, per f.
