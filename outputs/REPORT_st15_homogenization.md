# Station 15 — MSG-TW homogenization report

Reference-surface study (1Dshell_15) vs FEniCS solid (2Dsolid_VABS_15).  6x6 Timoshenko stiffness, order [F1,F2,F3,M1,M2,M3] <-> [eps11,g12,g13,k1,k2,k3].

## Diagonal (engineering) stiffnesses

| term | OML | centroid | IML | FEsolid | OML % | cen % | IML % |
|------|-----|----------|-----|---------|-------|-------|-------|
| C11=EA | 1.333e+10 | 1.296e+10 | 1.261e+10 | 1.308e+10 | +1.9 | -0.9 | -3.6 |
| C22=GA12 | 4.462e+08 | 4.362e+08 | 4.261e+08 | 4.580e+08 | -2.6 | -4.8 | -7.0 |
| C33=GA13 | 9.627e+07 | 1.005e+08 | 1.049e+08 | 1.055e+08 | -8.7 | -4.7 | -0.6 |
| C44=GJ | 1.504e+08 | 1.539e+08 | 1.567e+08 | 1.560e+08 | -3.6 | -1.4 | +0.4 |
| C55=EI2 | 1.672e+09 | 1.631e+09 | 1.590e+09 | 1.663e+09 | +0.5 | -1.9 | -4.4 |
| C66=EI3 | 5.809e+09 | 5.672e+09 | 5.558e+09 | 5.107e+09 | +13.7 | +11.1 | +8.8 |

## Full 6x6 — 21 unique terms  (% = 100*(JAX-FE)/FE)

| Cij | OML | centroid | IML | FEsolid | OML % | cen % | IML % |
|-----|-----|----------|-----|---------|-------|-------|-------|
| C11 | 1.333e+10 | 1.296e+10 | 1.261e+10 | 1.308e+10 | +1.9 | -0.9 | -3.6 |
| C12 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C13 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C14 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C15 | 2.860e+07 | 4.457e+07 | 6.015e+07 | 1.435e+07 | +99.4 | +210.7 | +319.3 |
| C16 | -3.242e+09 | -3.055e+09 | -2.878e+09 | -3.571e+09 | -9.2 | -14.5 | -19.4 |
| C22 | 4.462e+08 | 4.362e+08 | 4.261e+08 | 4.580e+08 | -2.6 | -4.8 | -7.0 |
| C23 | -2.174e+07 | -2.161e+07 | -2.164e+07 | -2.355e+07 | -7.7 | -8.3 | -8.1 |
| C24 | -2.471e+07 | -2.328e+07 | -2.188e+07 | -2.179e+07 | +13.4 | +6.8 | +0.4 |
| C25 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C26 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C33 | 9.627e+07 | 1.005e+08 | 1.049e+08 | 1.055e+08 | -8.7 | -4.7 | -0.6 |
| C34 | 4.822e+07 | 4.820e+07 | 4.769e+07 | 5.055e+07 | -4.6 | -4.6 | -5.7 |
| C35 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C36 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C44 | 1.504e+08 | 1.539e+08 | 1.567e+08 | 1.560e+08 | -3.6 | -1.4 | +0.4 |
| C45 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C46 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  n/a |  n/a |  n/a |
| C55 | 1.672e+09 | 1.631e+09 | 1.590e+09 | 1.663e+09 | +0.5 | -1.9 | -4.4 |
| C56 | 2.972e+08 | 2.978e+08 | 3.013e+08 | 2.586e+08 | +15.0 | +15.2 | +16.5 |
| C66 | 5.809e+09 | 5.672e+09 | 5.558e+09 | 5.107e+09 | +13.7 | +11.1 | +8.8 |

- full-6x6 ||OML-FE||/||FE|| = **5.85%**

- full-6x6 ||centroid-FE||/||FE|| = **6.22%**

- full-6x6 ||IML-FE||/||FE|| = **7.89%**

## Out-of-plane stress (dehom) — validated against native FEniCS-TW

The Kirchhoff-shell dehom recovers the 3D stress from the **in-plane shell
strains** (membrane + curvature) only.  The three transverse-traction
components come out **zero by construction**, and this was verified directly
against the FEniCS-TW (`opensg.core.shell`) plate model — not a JAX artifact:

| quantity | JAX | native FEniCS-TW |
|----------|-----|------------------|
| plate warping `v1` (→ γ13) | 0 (all 6 cols, incl. off-axis [45,-45,0]) | 0 (all 6 cols) |
| plate warping `v2` (→ γ23) | 0 | 0 |
| plate warping `v3` (→ γ33) | 1.77e-2 / 7.26e-4 (e/k cols) | 1.77e-2 / 7.26e-4 |
| dehom `max|S33| / max|in-plane|` | ~1e-14 | ~1e-14 |
| dehom `max|S13|, max|S23|` | **0 exactly** | **0 exactly** |

**Why** (correct MSG plate behaviour, not a bug):
- `σ33`, `σ13`, `σ23` are the transverse-*traction* components. In a 1D
  through-thickness SG with free top/bottom faces, equilibrium forces them to
  vanish — this is the plate (plane-stress-through-thickness) condition.
- `v1 = v2 = 0`: the in-plane Kirchhoff macro (`Ge` non-zero only in rows
  0,1,5) never excites the transverse-shear warping, so `γ13 = dv1/dx = 0` and
  `γ23 = dv2/dx = 0` → `σ13 = σ23 = 0`.
- The 3D *strain* is genuinely full: `γ11, γ22, γ12, γ33` are all non-zero
  (`γ33 = dv3/dx ≠ 0`). But `σ33 = (C·γ)_33 ≈ 0` precisely because `v3` is
  solved to nullify the through-thickness traction — that *is* the definition
  of the warping function.
- Non-zero `σ13/σ23` would require feeding the **transverse-shear** macro
  (Reissner–Mindlin `n_model=3`, `Ge=I` with the `g13/g23` columns). That was
  deliberately excluded here ("recover all 6 from Kirchhoff shell strain").
  VABS reports non-zero `σ13/σ23` in thick spar caps because it carries that
  transverse-shear flow; the Kirchhoff-shell dehom does not.
