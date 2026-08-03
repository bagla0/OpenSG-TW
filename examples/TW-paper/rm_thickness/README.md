# `rm_thickness` — physics-based replacement for the Garg-2023 GPR through-thickness patch

Reference: A. Garg, T. Mukhopadhyay, M.O. Belarbi, H.D. Chalak, A. Singh, A.M. Zenkour,
*"On accurately capturing the through-thickness variation of transverse shear and normal
stresses for composite beams using FSDT coupled with GPR"*, Compos. Struct. **305** (2023)
116551.

Garg et al. take FSDT (which gets σ13 badly wrong and cannot produce σ33 at all), compute
the difference against Pagano's 3-D elasticity solution for 500 Sobol-sampled laminates,
and fit a Gaussian-Process-Regression surrogate to that difference. The corrected FSDT
then matches elasticity — **inside the training box only** (their Table 1: `l/h ∈ [4,100]`,
sinusoidal load, simply-supported ends, ≤20 layers, and only those material bounds).

This folder asks whether MSG/VAM gets the same through-thickness distributions **by
asymptotic construction instead of by regression on the answer**.

---

## Problem statement

Laminate of total thickness `h`, `N` plies, infinite in `y` (∂/∂y = 0), simply supported
at `x = 0, L`. Top face loaded by

```
sigma33(x, +h/2) = q0 sin(p x),      p = pi / L
sigma33(x, -h/2) = sigma13(x, ±h/2) = sigma23(x, ±h/2) = 0
```

This is exactly Pagano's cylindrical-bending problem and exactly Garg's set-up. At the
plate level the problem is **statically determinate**:

```
N11 = N12 = 0,   M11 = q0 / p^2,   M12 = 0,   Q1 = q0 / p,   Q2 = 0
```

so every model below sees the same stress resultants, and the only thing being compared is
**how each model distributes them through the thickness**.

## The four models

| file | model | what it is |
|---|---|---|
| `exact_cyl.py` | **3-D elasticity (reference)** | exact, analytical. Not FEA. |
| `cyl_models.fsdt_profile` | **FSDT** | Garg's baseline: single director + `k = 5/6` |
| `cyl_models.clt_equil_profile` | **CLT + equilibrium** | classical Whitney-1973 shear flow |
| `cyl_models.msg_profile` | **MSG-VAM (OpenSG)** | 1-D SG FE through the thickness |

### `exact_cyl.py` — exact 3-D elasticity, state-space form

Rather than Pagano's quartic-root construction, each layer is written as a first-order
system in the state vector

```
s(z) = [U, V, W, X, Y, Z]        u = U cos(px), v = V cos(px), w = W sin(px)
                                 sigma13 = X cos(px), sigma23 = Y cos(px), sigma33 = Z sin(px)
```

for which the 3-D equations reduce **exactly** to `s' = A s` with `A` constant per layer.
The layer propagator is `expm(A h_k)`; the laminate transfer matrix is their product, and
the three unknowns `[U,V,W]` at the bottom face come from a 3×3 solve.

Why this instead of Pagano's own construction:
* handles **arbitrary fibre angles** (monoclinic plies) — Pagano 1969 is restricted to
  specially-orthotropic (cross-ply) layers, so the angle-ply comparisons in Garg's Fig. 5
  do not actually have the reference they cite;
* interface continuity of `(u,v,w,σ33,σ13,σ23)` and both traction BCs hold to machine
  precision by construction — no root finding, no ill-conditioned `4m × 4m` system.

Verified in `validate_exact.py`: BC residual `≤ 6e-14`; the recovered stress integrates to
`N11 = 0`, `M11 = q0/p²`, `Q1 = q0/p` to 9 digits; isotropic σ13 sits `8e-4` from the
`1.5 Q/h` parabola; and σ11 → CLT at a clean `O(S⁻²)` rate over `S = 4 … 400`.

### The MSG side — which OpenSG pieces were used

The FEA in this study is the **MSG 1-D Structure Gene**: a finite element mesh *through the
laminate thickness*, cubic Lagrange, 3 warping DOF per node (≈110 DOF for a 3-ply). There
is no 2-D or 3-D mesh anywhere.

* `msg_rm_plate.py` — copied from `examples/TW-paper/xsec_paper/msg_rm_plate.py`
  (the module already validated in the CPB paper: isotropic ν = 0 → `G = 5/6 Gh` exact,
  `A6 == compute_ABD_matrix`). Only the import line was changed, plus two additions:
  `msgrm_recover_profile` (interface-aware sampling) and `sigma33_equilibrium`.
* `materials.py` — byte-compatible copies of `opensg_jax.fe_jax.msg_materials`
  `build_stiffness_6x6` / `rotation_6x6` / `_plate_B` / `_grad_ops`, so the folder runs
  standalone but sees exactly the same ply stiffness as the production pipeline.
* `cyl_models.py` — **new**: the cylindrical-bending driver (plate strains from the
  statically-determinate resultants) and the FSDT / classical baselines.

Recovery chain actually exercised:

1. zeroth-order warping `V0` → `A6` (classical ABD) → in-plane σ11, σ22, σ12;
2. first-order gradient warping `C1bar` driven by `dE/dx1 = p Ê` → σ13, σ23;
3. σ33 by integrating `dσ33/dz = p σ13` from the bottom face — which lands on the applied
   `q0` automatically, because the recovered σ13 integrates to `Q1`.

## Results

`pilot.py` (table), `diagnose.py` (error decomposition), `plots.py` (figures).

**Relative L2 error of the through-thickness profile vs exact 3-D elasticity:**

| case | S | σ13 FSDT | σ13 MSG | σ33 FSDT | σ33 MSG |
|---|---|---|---|---|---|
| [0/90/0] | 100 | 73.5 % | **0.05 %** | n/a | **0.01 %** |
| [0/90/0] | 10 | 70.8 % | **4.4 %** | n/a | **1.2 %** |
| [0/90/0] | 5 | 64.1 % | 15.0 % | n/a | **3.9 %** |
| [0/90/90/0] AS4 | 10 | 71.8 % | **1.2 %** | n/a | **0.3 %** |
| sandwich 0.1/0.8/0.1 | 20 | 492.8 % | **2.4 %** | n/a | **0.2 %** |
| sandwich | 10 | 473.1 % | **9.1 %** | n/a | **0.8 %** |
| sandwich | 5 | 397.9 % | 30.4 % | n/a | **3.2 %** |

FSDT's σ13 error is **flat in S** — it never converges, at any thickness. MSG converges at
`O(S⁻²)` on every component.

### Three honest findings

1. **The first-order VAM recovery is identical to the classical Whitney-1973 shear flow**
   — `max|σ13_MSG − σ13_CLTeq| / max|σ13_exact| ≈ 3e-13`. Correct (both are the
   asymptotically exact `O(h/L)` transverse shear) but it means the novelty at first order
   is *not* σ13 accuracy; it is that MSG delivers it, plus σ33, plus the anisotropic
   coupling, from one variational construction with no load or BC assumption.

2. **The thick regime (S = 4–5) needs the second-order warping.** There σ11 is 43–86 %
   off and σ13 15–42 % off. `diagnose.py` localises it: for the sandwich only 0.26 % of the
   exact σ11 is non-linear *within* a ply, yet the error is 86 % — the classical
   partition of `M11` between the plies is wrong (the zig-zag). Recovering it is Yu 2002
   IJSS §5 / Yu 2005: the second-order warping driven by `E,11`, `E,12`, `E,22`.

3. **OPEN — angle-ply σ23 is under-predicted by ~2.4× and does not converge in S**
   (65–76 % error for `[0/θ/0]`, θ = 15…60, at S = 10 and 20). MSG and the classical
   equilibrium route give *identical* wrong answers, so it is a theory-order issue, not an
   implementation slip in either. Needs investigation before any of this is written up.

## Running

```
python validate_exact.py     # exact-solver verification
python pilot.py              # the comparison table
python diagnose.py           # where the residual error lives
python plots.py              # figures/ (no titles; captions live in the paper)
```

Environment: `C:\conda_envs\opensg_2_0_env\python.exe` (numpy/scipy/matplotlib). No JAX
yet — the MSG module is still the numpy one; the JAX port is the next step.
