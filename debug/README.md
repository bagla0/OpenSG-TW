# Debug scripts — C13 (ext–bend2 coupling) investigation

These are kept for later reference while we resolve the **C13 / tension-center**
discrepancy in the Kirchhoff MSG-TW homogenization.  They use absolute paths to
the local data (`OpenSG/examples/data/Shell_1DSG`, the `training data` `.K`
files) and the `opensg_jax` package.

## Findings so far (station 15, OML reference)

- `C13` (EB ext–bend2 = Timo `C15`) is **2.86e7 vs VABS 1.435e7 ≈ 2×**.
- `C13 = EA · (tension-center y3)`.  The "2×" is **not** a formulation factor-2:
  - `diag_c13.py`: st0 gives C13 ratio **−280** (sign-flipped, near-zero coupling),
    st15 **1.99**; C14 (ext–bend3) is 0.91 (st15) / 0.07 (st0).  A real factor-2
    would hit all of them ≈2.  It doesn't.  Dividing C13 by 2 fixes st15 but
    breaks st0 (wrong sign).
  - The real signal: JAX's **EA-weighted flatwise centroid (tension-center y3)**
    is **~1–1.6 mm too high** on both stations (st0: +1.56 mm, st15: +1.05 mm).
    C13 is just `EA·tc_y3`, so a ~1 mm bias looks like 2× when the true value is
    ~1 mm (st15) and explodes when it is ~0 (st0).

- Reference dependence: the tension center should be **reference-invariant** but
  JAX gives tc_y3 = 2.15 (OML) / 3.44 (centroid) / 4.77 mm (IML) — it drifts.
  - `diag_offsetpipe.py`: an offset symmetric pipe (tc must equal `y0`) shows the
    **OML representation is exact**, the **centroid is −1.6% at n=60** but
    **converges to exact at n=200** and shrinks with `h/R`.  => the centroid
    reference is a **curvature × discretization** effect that vanishes with mesh
    refinement, **not a sign/factor bug** in `shift_abd_reference` / the offset.

## Open question

The airfoil OML C13 = 2× is the thick-laminate B-coupling on a near-zero
coupling.  A **refined 1D cross-section mesh** (user to provide) should collapse
the OML/centroid/IML spread and tell us how much of the residual is mesh vs the
thick-cap shell idealization.  RM (transverse shear) is **not** expected to fix
C13 directly, but is being built for the thick-web shear/torsion terms.

## Scripts
- `diag_c13.py` — JAX EB vs VABS `.K` C13/C14 for st0 and st15 (+ tension centers).
- `diag_offsetpipe.py` — reference-consistency test on an offset symmetric pipe
  (OML vs centroid representation, convergence with n and h/R).
