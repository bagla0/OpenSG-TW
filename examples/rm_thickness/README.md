# `rm_thickness` — asymptotically exact through-thickness recovery, without a surrogate

Branch: `rm-thickness-recovery`.  Everything here is **JAX** (float64).

## What this answers

A. Garg, T. Mukhopadhyay, M.O. Belarbi, H.D. Chalak, A. Singh, A.M. Zenkour,
*"On accurately capturing the through-thickness variation of transverse shear and normal
stresses for composite beams using FSDT coupled with GPR"*, Compos. Struct. **305** (2023)
116551, takes FSDT — which gets `sigma13` badly wrong and cannot produce `sigma33` at all —
computes the difference against Pagano's 3-D elasticity solution for 500 Sobol-sampled
laminates, and fits a Gaussian-process surrogate to that difference.  The corrected FSDT
then matches elasticity, **inside the training box only** (their Table 1: `l/h ∈ [4,100]`,
sinusoidal load, simply-supported ends, ≤20 plies, and only those material bounds).

This folder asks whether MSG/VAM reaches the same distributions **by asymptotic
construction rather than by regression on the answer**.

## Problem statement

Laminate of thickness `h`, `N` plies, infinite in `y` (∂/∂y = 0), simply supported at
`x = 0, L`, loaded on the top face by

```
sigma33(x, +h/2) = q0 sin(pi x / L),   sigma33(x,-h/2) = sigma13(x,±h/2) = sigma23(x,±h/2) = 0
```

Pagano's cylindrical bending; exactly Garg's set-up.  At the plate level the problem is
**statically determinate**:

```
N11 = N12 = 0,   M11 = q0/p^2,   M12 = 0,   Q1 = q0/p,   Q2 = 0
```

so every model here sees the same stress resultants and the comparison is purely about how
each **distributes them through the thickness**.

## Models

| module | model | role |
|---|---|---|
| `exact_cyl.py` | 3-D elasticity | reference — **analytical**, not FEA |
| `models.fsdt` | FSDT, `k = 5/6` | Garg's baseline |
| `models.clt_equil` | CLT + equilibrium | classical Whitney-1973 shear flow |
| `models.msg` | MSG-VAM | the 1-D SG finite element model |

### `exact_cyl.py` — exact elasticity, state-space form

Each layer is written as a first-order system in

```
s(z) = [U, V, W, X, Y, Z]     u = U cos(px), v = V cos(px), w = W sin(px)
                              sigma13 = X cos(px), sigma23 = Y cos(px), sigma33 = Z sin(px)
```

for which the 3-D equations reduce **exactly** to `s' = A s`, `A` constant per layer.  The
layer propagator is `expm(A h_k)`; the laminate transfer matrix is their product; the three
unknowns at the bottom face come from a 3×3 solve.

Two things this buys over Pagano's own construction:
* **arbitrary fibre angles** — Pagano builds `f(z)` from the roots of a quartic that exists
  only for specially orthotropic (cross-ply) plies, so the angle-ply comparison in Garg's
  Fig. 5 cites a reference that does not cover that case;
* interface continuity and both traction BCs hold to **machine precision by construction**.

**Scaling matters.** In physical units `A` mixes `1/C ~ 1e-11` with `p^2 C ~ 1e10` — a
spread of ~1e21 — and `expm` of that returns NaN even though the eigenvalues are `O(p)`.
The solver therefore works in `z~ = z/h`, `(U,V,W)~ = (U,V,W)/h`, `(X,Y,Z)~ = (X,Y,Z)/E0`,
`p~ = p h`, which makes every entry of `A~` order unity.

### The MSG side — which OpenSG pieces are used

The FEA in this study is the **MSG 1-D Structure Gene**: a finite element mesh through the
laminate *thickness*.  Cubic Lagrange, 3 warping DOF per node, 111 DOF for a three-ply
laminate.  There is no 2-D or 3-D mesh anywhere.

`sg_plate.py` is the JAX port of `examples/TW-paper/xsec_paper/msg_rm_plate.py`, whose
`A6` is pinned to the shipped `opensg_jax.fe_jax.msg_materials.compute_ABD_matrix`
(agreement 2e-16, see `validate.py`).  Recovery chain:

1. zeroth-order warping `V0` → `A6` (classical ABD) → in-plane `sigma11, sigma22, sigma12`;
2. first-order gradient warping `C1bar` driven by `dE/dx1 = p*Ehat` → `sigma13, sigma23`;
3. `sigma33` by integrating `dsigma33/dz = p*sigma13` from the bottom face — which lands on
   the applied `q0` automatically, because the recovered `sigma13` integrates to `Q1`.

## Results

Relative L2 error of the through-thickness profile against exact 3-D elasticity:

| case | S | `sigma13` FSDT | `sigma13` MSG | `sigma33` FSDT | `sigma33` MSG |
|---|---|---|---|---|---|
| [0/90/0] | 100 | 73.5 % | **0.05 %** | n/a | **0.01 %** |
| [0/90/0] | 20 | 71.9 % | **1.16 %** | n/a | **0.31 %** |
| [0/90/0] | 10 | 70.8 % | **4.43 %** | n/a | **1.17 %** |
| [0/90/0] | 5 | 64.1 % | 15.0 % | n/a | **3.91 %** |
| [0/90/90/0] AS4 | 10 | 71.8 % | **1.20 %** | n/a | **0.32 %** |
| sandwich 0.1/0.8/0.1 | 20 | 492.8 % | **2.39 %** | n/a | **0.21 %** |
| sandwich | 10 | 473.1 % | **9.15 %** | n/a | **0.83 %** |
| sandwich | 5 | 397.9 % | 30.4 % | n/a | **3.15 %** |

FSDT's `sigma13` error is **flat in S** — it never converges, at any thickness.  MSG
converges at `O(S^-2)` on every component.

### The whole Garg box in one vmapped call

`sweep.py` draws the same Sobol sample from their Table 1 (512 laminates × 8 plies) and
runs the SG solve, the recovery and the exact solution over the population at once:

```
512 laminates in 6.8 s (13.3 ms each, CPU)
oracle integrity: worst exact traction-BC residual = 6.6e-14   ok

quantity                      median  90th pct       max
sigma13 FSDT                  70.43%   100.36%   165.55%
sigma13 MSG                    6.89%    17.67%    36.72%
sigma33 MSG                    2.32%     6.56%    20.30%
sigma11 MSG                   19.29%    54.89%   150.91%
sigma33 top-face closure       0.02%     0.05%     0.15%
```

Every one of these is an out-of-sample prediction, because there is no sample.

## Three honest findings

1. **The first-order VAM recovery is identical to the classical Whitney-1973 shear flow**
   — `max|sigma13_MSG - sigma13_CLTeq| / max|sigma13_exact| ~ 3e-13`.  Correct (both are
   the asymptotically exact `O(h/L)` transverse shear), but the novelty at first order is
   *not* `sigma13` accuracy; it is that one variational construction delivers it, plus
   `sigma33`, plus the anisotropic coupling, with no assumed load or boundary condition.

2. **The thick regime (S = 4–5) needs the second-order warping.**  There `sigma11` is
   43–86 % off and `sigma13` 15–42 % off.  `diagnose.py` localises it: for the sandwich
   only 0.26 % of the exact `sigma11` is non-linear *within* a ply, yet the error is 86 % —
   so it is the ply-to-ply partition of `M11` (the zig-zag) that is wrong, not the
   within-ply curvature.  Recovering it is Yu 2002 IJSS §5 / Yu 2005: warping driven by
   `E,11`, `E,12`, `E,22`.

3. **OPEN — angle-ply `sigma23` is under-predicted ~2.4× and does not converge in S**
   (65–76 % error for `[0/theta/0]`, theta = 15…60, at S = 10 and 20).  MSG and the
   classical equilibrium route give *identical* wrong answers, so it is a theory-order
   issue, not an implementation slip in either.  Must be resolved before publication.

## Running

```
python validate.py     # verification suite -- run this first
python run_study.py    # the case-by-case table -> results/table.csv
python diagnose.py     # where the residual error lives
python sweep.py        # the vmapped Garg-box sweep -> results/sweep.npz
python plots.py        # figures/
```

Environment: `C:\conda_envs\opensg_2_0_env\python.exe` (JAX 0.9.2, x64 enforced by
`jaxcfg.py` — import it before any other JAX import).

The paper lives in the Overleaf project `6a63c536a3f99dab46a52f1a`
(clone: `C:\Users\bagla0\ol_rm_thickness`).
