# `navier_plate` — replication of Mendonça & Ruviaro (2026) with OpenSG-RM

Reference: P.d.T.R. Mendonça, F. Ruviaro, *"Recovering 3D stress and displacement fields
from low-order finite element results for laminated composite plates with FSDT"*, Finite
Elem. Anal. Des. **260** (2026) 104609.

Their process: FE-FSDT (9-node quads) → **sequential SPR smoothing** to manufacture up to
4th derivatives of the low-order basis → equilibrium integration through the thickness for
τxz, τyz, σz → compliance integration for w and u, each with a **correction step** (linear
rotation Eq. 23, multiplicative scaling Eq. 32, shift Eq. 35, shift+rotation Eqs. 41–44).
Their converged FE curves coincide with the recovery applied to the **analytic** FSDT
solution — which is what `fsdt_mr()` here reproduces exactly.

OpenSG-RM answers with the same 5-DOF Navier problem closed by **G_MSG** and every
through-thickness field taken **directly from the structure-gene warping** — no smoothing
(the SG supplies exact derivatives), no correction/scaling/shift/rotation steps (closures
hold by construction).

## Problem

Square plate a=b=1000 mm, simply supported, q = q₁₁ sin(πx/a) sin(πy/a), q₁₁ = 0.01 MPa,
a/H = 4 and 100.  Laminates: [0/90/0], [0/90] (Pagano graphite/epoxy, E1=172.25 GPa), and
the Pagano-1970 sandwich (0.1H faces, 0.8H transversely-isotropic core).  Profiles at
their extraction points with their Eq.-52 normalizations: τ̄xz at (0,a/2), σ̄z and w̄ at
(a/2,a/2), ū at (0,a/2).

## Files

| file | role |
|---|---|
| `exact_navier.py` | exact 3-D elasticity (Pagano 1970 config), state-space/expm form, JAX |
| `navier_models.py` | 5×5 Navier solve; `msg_mr` (OpenSG-RM) and `fsdt_mr` (their process, analytic) |
| `validate_navier.py` | verification suite — ALL PASS |
| `plots_navier.py` | the 12 replication figures + `results/table_navier.csv` |

## Verification anchors

* exact solver: traction BCs ≤ 2e-14, σz closure ≤ 7e-15, **cylindrical limit b/a=1e5
  collapses onto ../exact_cyl.py to ≤ 1.3e-11**;
* [0/90/0] central deflection: 3D w̄ = **2.0010** (a/H=4) and **0.4347** (a/H=100) — the
  standard Pagano-1970 benchmark values; FSDT k=5/6 gives 1.7741 / 0.4337, exactly the
  values in their Fig. 4;
* the analytic-FSDT recovery closes σz on q₁₁ *before* its scaling step (their Theorem 1)
  to ≤ 4e-4; the MSG σz closes without any scaling step to the same order.

## Results (relative L2 vs exact 3-D)

| case | a/H | qty | M-R recovery | OpenSG-RM |
|---|---|---|---|---|
| [0/90/0] | 4 | τxz | 23.95% | **20.90%** |
| [0/90/0] | 4 | u | 45.31% | **32.64%** |
| [0/90/0] | 4 | w | 11.21% | **9.58%** |
| [0/90] | 4 | u | 8.93% | **3.73%** |
| [0/90] | 4 | w | **3.94%** | 6.77% |
| sandwich | 4 | **w** | 36.97% | **3.12%** |
| sandwich | 4 | u | 99.48% | **67.45%** |
| [0/90/0] | 100 | w | 0.231% | **0.005%** |
| sandwich | 100 | w | 0.804% | **0.002%** |

(full table: `results/table_navier.csv`; stresses at a/H=100 all ≤ 0.06% for both.)

Reading: on **stress** the two recoveries are essentially tied — both are the equilibrium
shear flow (Section Open-items of the paper).  The separation is on **displacement**:

* **sandwich w̄: 37% → 3.1%.**  The M-R w-recovery integrates εz from recovered stresses
  but is *gauged to the FSDT deflection* (w_ic = w0 at z=0, their Eq. 35), so it inherits
  the k=5/6 FSDT deflection error, which is catastrophic for the shear-soft sandwich
  (their own text: −31.2% at z=0).  OpenSG-RM's deflection comes from G_MSG, which is the
  asymptotically correct shear stiffness — the profile lands on the 3-D curve.
* thin-limit w̄: 0.002–0.005% vs 0.2–0.8% for the same reason.
* ū: OpenSG-RM better everywhere (2–4× thin, 1.4–2.4× thick).

## Two lessons (cost a day; recorded in `navier_models.msg_mr` docstring)

1. **Stress drivers must be the moment-consistent RM measures.**  The classical curvature
   p²W₀ contains the shear deflection, which carries no moment; at a/H=4 with E1/G13=50
   the shear deflection *dominates* W₀ (w̄: 2.20 total vs 0.43 bending), so classical
   -measure drivers overdrive the bending stress ~5× — τxz went to 567% before the revert.
2. **The displacement warping enters as its pure deviation** (least-squares linear part
   removed): the RM director ψx already carries the mean shear rotation; keeping both
   double-counts it (u at a/H=4 went to 479%).  This is the same rotation gauge M-R impose
   on their recovered u (Eqs. 43–44), so the comparison is like-for-like.

## Running

```
python validate_navier.py
python plots_navier.py
```

Environment: `C:\conda_envs\opensg_2_0_env\python.exe` (JAX x64 via ../jaxcfg.py).
