---
title: Tapered segments — the six-parameter (independent-$\omega_3$) RM model
---

# Tapered segments: the six-parameter (independent-$\omega_3$) RM model

On a **flat-walled** tapered section the classical RM shell — with the drilling
rotation $\omega_3$ *eliminated* through the in-plane symmetry — under-predicts the
beam transverse-shear stiffness by **$-24\%$** (isotropic) to **$-40\%$** ($[-45]$)
on a thin tapered square tube, an error that is mesh-converged, grows with the taper
rate squared, and is insensitive to any regularization of the $1/C_{33}$ drilling
reciprocal. This page shows the production resolution used by OpenSG-TW's tapered
pipeline: a **single six-parameter element** — the drilling rotation kept as an
independent DOF, the in-plane symmetry enforced exactly by element-wise Lagrange
multipliers, full shear integration on the segment (the rings retain the exact
γ₂₃ tie) — used for **both** the boundary rings and the tapered segment, with the
ring warping fields (including $\omega_3$) transferred to the segment as
Dirichlet data.

All inputs, the driver scripts, and the full-$6\times6$ `.dat` are bundled under
[`mitc_rm_segment/taper_indep_study/`](https://github.com/bagla0/OpenSG-TW/tree/main/mitc_rm_segment/taper_indep_study).

## Why the eliminated drilling fails on flat walls

The elimination $\omega_3=\big(S/2-y_\beta\omega_\beta\big)/C_{33}$ divides by
$C_{33}=\mathbf{a}_3\cdot\mathbf{b}_3$, which vanishes **identically over every flat
wall whose normal is perpendicular to $\mathbf{b}_3$** — exactly the walls that
carry the $V_3$ shear flow. On a smooth (circular) section $C_{33}=0$ only at
isolated points and nothing goes wrong; on the square the degeneracy covers whole
walls and the tapered transverse shear collapses. Keeping $\omega_3$ independent
removes every reciprocal from the strain operators (they become polynomial in the
direction cosines) and restores the symmetry condition as a finite constraint row
$g = y_k\omega_k - S/2 = 0$, enforced by one Lagrange multiplier per element.

## Results — all 8 cases at strong taper ($a_R=0.7$)

Tapered-segment diagonal errors vs the FEniCS 3-D solid (48×10 shell mesh;
$C_{22}=C_{33}$ is satisfied **identically** — the drilling boundary data enforce
the square's physical shear symmetry by construction):

| case | EA | GA₂ | GA₃ | GJ | EI₂ | EI₃ |
|---|---|---|---|---|---|---|
| square thin iso | +1.0% | −2.9% | −2.9% | −2.4% | +1.0% | +1.0% |
| square thin [-45] | +1.3% | **−1.7%** | **−1.7%** | −4.4% | +2.3% | +2.3% |
| square thick iso | +0.7% | −4.5% | −4.5% | −4.5% | −0.3% | −0.3% |
| square thick [-45] | +0.8% | +1.9% | +1.9% | −6.1% | +1.7% | +1.7% |
| circle thin iso | +1.2% | +3.8% | +3.8% | +1.2% | +1.3% | +1.3% |
| circle thin [-45] | +0.8% | +5.1% | +5.1% | +0.0% | +2.0% | +2.0% |
| circle thick iso | +1.0% | +2.2% | +2.2% | +0.7% | +0.2% | +0.2% |
| circle thick [-45] | +0.3% | +3.5% | +3.5% | −1.1% | +0.9% | +0.9% |

The eliminated-drilling operator on the same thin square gives GA₃ = −24.4%/−39.9%
with the coupling C₃₆ at −39.7% — the motivation for the six-parameter model.

## Cross-sections: 5-DOF MITC vs 6-DOF ring (square)

The same element solves the boundary rings on a wrapped strip. On the flat-walled
cross-section it matches the validated 5-DOF eliminated+MITC element on EA/GA/EI and
**repairs the ring torsion** (the floored drilling reciprocal injects a small
spurious prismatic GJ on flat walls):

| stiffness | 5-DOF MITC (iso) | 6-DOF (iso) | 5-DOF MITC ([-45]) | 6-DOF ([-45]) |
|---|---|---|---|---|
| EA | +0.0% | +0.0% | +0.8% | +0.8% |
| GA₂ | −3.6% | −3.3% | +1.1% | +1.5% |
| GA₃ | −3.7% | −3.3% | +0.9% | +1.5% |
| **GJ** | **+9.6%** | **−3.8%** | **+9.0%** | **+1.4%** |
| EI₂=EI₃ | −0.0% | −0.0% | +0.5% | +0.6% |

Both rings use their production shear treatments; on the span-invariant strip the
assumed γ₂₃ field reproduces the true shear exactly, so the treatment is exact
there by construction.

## Transverse shear: no MITC required (and canonical MITC is harmful)

The production scheme is **full 2×2 integration of both shear rows on the tapered
segment**, with the Dvorkin–Bathe **γ₂₃ tie retained only on the boundary rings**
(where it is exact by construction on the span-invariant strip — and verified
*inert*, see below). No assumed-strain protection is needed anywhere, for two
structural reasons:

1. **The discrete shear constraint restricts only the rotations.** The rows carry
   the rotations algebraically through the full-rank tangential traces
   ($2\gamma_{13}\supset x_{i;2}\,\omega_i$, $2\gamma_{23}\supset-x_{i;1}\,\omega_i$),
   so for any displacement field the thin-limit condition is solvable pointwise
   for the rotations — the constraint never eliminates a displacement mode.
2. **The SG fluctuation problem is never bending-dominated.** The Kirchhoff-mode
   content is carried analytically by the section-strain columns and by the ring
   Dirichlet data; the shear/twist load columns demand genuinely *finite* shear.

Certification (worst cases, all on the server):

- **Prismatic flat-wall identity** (`run_locksq.py`): the square segment
  reproduces its own ring to **±0.00% on every constant at t/R = 2·10⁻², 2·10⁻³,
  2·10⁻⁴**, both meshes, every shear scheme.
- **Boundary rings, square vs ellipse** (`run_ringboun.py`): full integration vs
  γ₂₃-tied agree within **0.05 points** on every constant, every mesh (48→384
  hoop), and every thickness over two decades — errors decay ∝ nc⁻² and are
  thickness-invariant, i.e. pure discretization, no locking signature on flat
  *or* curved boundaries. The MITC requirement reported for the 5-DOF
  eliminated-drilling cross-sectional element does **not** carry over to the
  6-DOF constrained element.
- **Prismatic circle probe**: errors vs closed form identical at t/R = 0.02 and
  0.002; a 5-DOF full-integration control is equally clean.

Conversely, **canonical MITC tying (whole rows, rotation columns included) must
not be used** with the 6-DOF element: it aliases the algebraic drilling content
(x₃;₂ω₃ — the role ω₂ plays prismatically) and collapses the flat-wall shears
(thin square −29/−47%; webbed ellipse −17/+29%, next section). A flux-only tie
(rotation columns kept at Gauss values) reproduces full integration to machine
precision — there is no flux-side locking to remove.

## Blade-like example: tapered ellipse with three webs

The discriminating validation case: a differentially tapered elliptical skin
(a: 1.0→0.65, b: 0.60→0.42 over L = 2.0, hoop curvature varying around *and*
along), three flat webs tapering with the section at x = ±a/2 and x = 0 (six
T-junction lines), t = 0.02, single [-45] ply — a four-cell blade-like layout,
compared against a fresh conforming 3-D solid (96×20×4, web end columns sharing
the skin through-thickness columns):

![webbed ellipse shell mesh (cutaway)](_img/taper_ell3w_shell_mesh.png)
![webbed ellipse solid mesh](_img/taper_ell3w_solid_mesh.png)

Segment diagonal %err vs solid (shell 48×10 skin + 6 elements/web, `run_ell3w.py`):

| scheme | EA | GA₂ | GA₃ | GJ | EI₂ | EI₃ |
|---|---|---|---|---|---|---|
| **full integration (production)** | +3.8 | +8.0 | +10.1 | **+0.0** | +6.7 | +2.3 |
| canonical MITC (both rows) | +4.4 | **−16.7** | **+28.9** | −0.1 | +2.6 | +3.3 |

Full integration stays coherent (GJ exact, shears +8/+10% — a scheme-independent
junction/model residual); canonical tying scatters the shears wildly. On webbed
sections the shear treatment is not cosmetic.

## Mesh convergence

Proportional refinement 24×5 → 96×20, thin wall, strong taper, fixed solid
reference:

![convergence](_img/taper6dof_convergence.png)

- **Circle**: converged — every curve moves < 0.4 points over 16× more elements;
  the +4–5% shear plateau is the shell-model error at this slenderness.
- **Square**: accurate at engineering resolution (+0.02% at 24×5, −2.9% at 48×10,
  isotropic) with a slow fold-line drift under further refinement (−8.6% iso /
  −17.4% [-45] at 96×20): the smooth-patch symmetry constraint over-constrains the
  C⁰-shared rotations across the four fold lines. The ω₃ boundary data halve the
  drift relative to free end drilling; a fold-consistent drilling treatment is the
  open question. GJ/EA/EI are mesh-insensitive throughout. Practical guidance:
  near-unit element aspect ratio, ~12 elements per wall.

## Computational cost

Wall-clock seconds per case, single core (32-core Linux server), reference mesh:

| case | extract | rings | segment | shell total | solid (boun+taper) |
|---|---|---|---|---|---|
| square thin iso | 1.0 | 2.0 | 4.9 | **7.9** | 6 |
| square thick iso | 0.9 | 1.5 | 4.6 | **7.1** | 44 |
| circle thin iso | 0.9 | 1.5 | 4.6 | **7.0** | 17 |
| circle thick m45 | 0.8 | 1.5 | 4.5 | **6.8** | 16 |

The shell cost is independent of geometry, thickness, and — unlike the solid, which
must resolve every ply through the thickness — of the layup count.

## Reproduce

```bash
# on the compute server (conda env opensg_2_0)
cd mitc_rm_segment
python run_taper_indep_study.py      # 8 cases -> taper_indep_results.dat (+ timing)
python run_paper_convergence.py      # 5-level mesh sweep -> paper_convergence.{dat,npz}
python run_extras.py                 # shear ablation + locking probe
python plot_paper_convergence.py     # -> fig_convergence.png + timing_summary.dat
python run_ring_indep.py             # 5-DOF vs 6-DOF ring comparison
python run_locksq.py                 # prismatic flat-wall identity, t/R -> 2e-4
python run_ringboun.py               # ring locking experiment: square vs ellipse
python run_ell3w.py                  # webbed-ellipse benchmark (+ fresh solid ref)
python render_ell3w.py               # cutaway mesh figures
```

Solver entry points: `run_indep.shell_solve_lagrange` (segment, all-6-DOF) and
`run_ring_indep.ring_indep` (ring); operators in `segment_indep.py`. Solid
references: `examples/data/benchmark/taper_{square,study}_solid_{iso,m45}.npz`.
