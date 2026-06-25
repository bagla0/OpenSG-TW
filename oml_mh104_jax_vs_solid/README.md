# oml_mh104_jax_vs_solid — mh104 OML, JAX shells vs FEniCS solid (no FEniCS-shell)

Clean OML-reference comparison of the two **JAX MSG-shell models** — **Kirchhoff (C1-Hermite)** and
**Reissner-Mindlin (C0)** — against the **VABS-validated FEniCS 2D-solid**, for the **mh104** airfoil
over wall-thickness factor `f = 0.1…0.6`. Order `[EA, GA2, GA3, GJ, EI2, EI3]`; reference =
quarter-chord.

**FEniCS-shell is deliberately excluded here** — its flapwise GA3/EI2 collapse for thin walls
(f=0.1–0.2) and distort the Timo plots. (The full 4-way version, including FEniCS-shell, lives in
`../oml_mh104/`.)

## Plots (all 21 nonzero 6×6 terms; diagonal 6 + coupling 8 + coupling 7)
Each panel: JAX-Kirchhoff (blue ●), JAX-RM (red ▲), FEniCS-solid (black ■ dashed). Off-diagonal panels
are labelled with the bracket index **C_ij** and the stiffness pair (e.g. `C12: EA-GA2`). The dashed
vertical line marks **f=0.2 = the nominal (validated) mh104 thickness** — thinner walls to the left,
thicker to the right.
- `figures/oml_abs_{diagonal,coupling_1,coupling_2}.png` — **absolute** stiffness, 3 curves.
- `figures/oml_pcterr_{diagonal,coupling_1,coupling_2}.png` — **% error** vs solid, 2 curves (Kirchhoff, RM).

## Headline
- **JAX-Kirchhoff and JAX-RM bracket the solid** on the shear terms (GA2, GA3) and their couplings:
  Kirchhoff over-predicts, RM under-predicts; the solid sits between.
- EA, GJ, EI2, EI3 agree well for both; GJ within ±5% across `f`. GA3's residual (~+17% even thin) is
  the Kirchhoff transverse-shear model limit; RM's transverse shear is softer.

## Data
- `data/C6_shell_jax_OML_f0NN.txt` — JAX-Kirchhoff OML 6×6.
- `data/C6_shell_rm_OML_f0NN.txt` — JAX-RM OML 6×6.
- `data/C6_solid_f0NN.txt` — FEniCS solid 6×6 reference.
- `data/oml_timo_table.txt` — diagonal: Kirchhoff, RM, solid + % errors, per f.
