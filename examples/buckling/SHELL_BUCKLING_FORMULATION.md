# Abaqus/Ansys shell linear-buckling formulation and the correct isotropic-cylinder benchmark

*(Synthesis of a 4-agent research pass: Abaqus Theory/Analysis/Benchmarks manuals, Ansys Theory
Reference + Verification Manual, Timoshenko-Gere / Brush-Almroth / NASA SP-8007, and the MSG-RM
initial-stress derivation. Validates `shell_buckling.py`.)*

## (1) The shared eigenvalue-buckling formulation

Both Abaqus and Ansys solve the identical generalized symmetric eigenproblem — a **linear
perturbation about a prestressed base state**:

$$(K + \lambda_i\,K_G)\,\phi_i = 0$$

- **K** — ordinary elastic (material tangent) stiffness (→ K_E with no preload).
- **K_G** — the **initial-stress / geometric / stress-stiffness** matrix: the destabilizing work a
  compressive membrane force does through the quadratic rotation terms of the strain. For a shell it
  is a through-thickness-integrated **resultant** form, dominated by the membrane forces N_αβ.
- **λ_i** — a load multiplier; critical load = λ_i × (reference load).

| | Abaqus (`*BUCKLE`) | Ansys (`BUCOPT`/`PERTURB,BUCKLE`) |
|---|---|---|
| Statement | (K⁰ + λ_i K^Δ) v = 0 | ([K] + λ_i [S]){ψ} = 0 (Eq. 15-107) |
| Name for K_G | **K_Δ** (differential/initial-stress) | **[S]** (stress-stiffness) |
| Formed from | ∫_V Δσ (∂²ε/∂u∂u) dV, perturbation stresses | ∫[G]ᵀ[S₀][G]dV, [S₀]=stress, [G]=grad |
| Prestress | K⁰ = tangent + initial-stress + load stiffness | `PSTRES,ON` / static base step mandatory |
| Shell K_G | N_αβ **and** M_αβ (present; **inactive** for pure-membrane cylinder) | predominantly membrane N_x,N_y,N_xy |

For the axial cylinder the prebuckling state is pure membrane (N_xx=−σt, M_αβ=0), so both codes
reduce to **exactly** the code's form: `(K + λK_G)φ=0` with K_G from membrane resultants N̂ only.

## (2) VERDICT — the membrane-only facet K_G is CORRECT in the fine-mesh limit

`shell_buckling.py` sums K_G over **g = u, v, w**:
`KG = ∫_A Σ_g (∇u_g)ᵀ N̂ (∇u_g) dA`, N̂ = [[Nxx,Nxy],[Nxy,Nyy]].

This carries membrane prestress on **all three** translational gradients — the w-term **(a)** *and*
the in-plane u,v membrane-gradient terms **(b)**. On a flat facet, (a)+(b) is the **discrete image of
the covariant curvature geometric term** a curved element carries explicitly: the 1/R coupling is
supplied by inter-facet tilt in the polygon→circle limit. The omitted terms vanish/are negligible at
a **center (mid-surface) reference**: (c) the z² rotation-gradient term is O(h²/L²) (dropped in all
classical thin-shell buckling); (d) the M_αβ, Q_α cross terms are **identically zero** (M=∫σ⁰z dz=0
for symmetric membrane prestress). The SS flat-plate benchmark = 0.9996×(4π²D/a²) certifies K and the
(a)-term. **K_G is not the problem — do NOT add M/Q/z² terms.**

## (3) DIAGNOSIS of the original 0.38× — ranked

**(a) BC ≫ (c) mesh ≫ (b) missing term.** The classical σ_cr=Et/(R√(3(1−ν²))) assumes **SS3 at both
ends with tangential v=0**. The original test clamped the root and left the tip **free** (v,w,rot,
M_x,Q_x all unrestrained), so the buckle localizes in an O(√(Rt)) free-edge boundary layer.

| BC | v restraint | ratio to classical |
|---|---|---|
| SS3 / C4 (v=0) | restrained | ≈ 1.00 (–1.03) |
| SS1 / C1 (v free) | free | ≈ 0.50 |
| clamped–**free** cantilever | all free | ≈ 0.3–0.5 |

**Decisive:** both "bug" hypotheses push the ratio the *wrong way* — a missing K_G term or a coarse
mesh both **overpredict** (ratio > 1). Only a genuinely weaker BC drives it **below** classical. So
0.38× can only be the BC — and it is correct physics. Near-degenerate mode pairs are the O(2)
cos nθ / sin nθ Koiter-circle degeneracy: a *signature of a correct* computation, not a symptom.

## (4) Correct benchmark

- **SS3 at both ends:** w=0 (radial), **v=0 (tangential — the essential one)**, M_x free, axial u free;
  pin one node for rigid body. Uniform axial N_x so prebuckling is pure membrane.
- Geometry R=1,t=0.02,L=2,ν=0.3 → **Batdorf Z=190.8 ≫ 1 → long cylinder, classical applies directly.**
  Expected mode m≈1, **n≈8–12**.
- Mesh ≥ ~120 circ × ~60 axial; refine until λ₁ stable < 1%.
- Targets: σ_cr = 0.605 Et/R = **2.42 GPa**; **N_cr = 4.84×10⁷ N/m**; P_cr = N_cr·2πR ≈ **304 MN**.

## (5) PASS criterion

Passes iff: SS3 both ends + mesh-converged (λ₁ change < 1% on refinement) +
`0.95 ≤ N_FE/N_cr ≤ 1.05` (N_cr = 4.84×10⁷ N/m) + mode m≈1, n≈8–12 (degenerate pairs OK).
(NASA SP-8007 knockdown γ≈0.68 at R/t=50 is an imperfection *allowable*, not a linear-eigenvalue target.)

## RESULT (this code, `test_cyl_bc.py`, nc=160 nl=80)

| BC | FE/classical | verdict |
|---|---|---|
| **SS3 both ends** | **0.952** | PASS (in [0.95,1.05]) |
| clamp-clamp | 0.964 | PASS |
| clamp-free (cantilever) | 0.379 | correct free-edge physics (NOT a classical benchmark) |

Flat plate: 0.9996. **Formulation validated.** The blade uses clamped-root/free-tip (cantilever),
which is the physically-correct BC for a wind blade — its lower factor reflects genuine tip-edge
compliance, exactly as a real free end behaves.
