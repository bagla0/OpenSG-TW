# MSG Reissner–Mindlin thin-walled beam — rigorous derivation & implementation

A step-by-step derivation of the MSG (Mechanics of Structure Genome) thin-walled
**Reissner–Mindlin (RM)** shell→beam model and the recipe to implement it for the
6×6 Timoshenko / 4×4 Euler–Bernoulli beam stiffness.  Synthesised from:

- **Dr. Yu, "Shell Derivation — Kirchhoff"** (base-vector geometry, §1 below);
- **Dr. Yu, "RM derivation"** (rotation kinematics, drilling elimination, §3);
- **hand-derived "RM curvatures"** sheets (explicit κ/γ expansion, §4);
- **Opensg_MSG.pdf §3.3** (strain field eq 4.23, constitutive 4.24, solved
  fluctuations 4.25/4.28, validation Tables 3.1/3.2).

Index convention: Greek `α,β ∈ {1,2}` (surface), `η ∈ {2,3}` (cross-section),
Latin `i,j ∈ {1,2,3}`.  `(·)' = ∂/∂x₁` (beam axis), `˙(·) = ∂/∂ζ₂` (contour).
`x̃` is the skew (cross-product) matrix of `x`, `e_i` the unit vectors.

---

## 1. Geometry of the shell reference surface  (Dr. Yu Kirchhoff, p.1)

The reference line carries an orthonormal beam triad `b̂_i`, `b̂₁ = ∂r̄/∂x₁`.
A surface point is `r(x₁,y_α) = r̄(x₁) + ε y_α b̂_α`, `α=2,3`.  The covariant base
vectors of the surface, expressed in the beam triad, are

```
â_β = [b̂₁ b̂₂ b̂₃] [ (e₁ + k̃_b x_b) x_{1;β} + x^b_{;β} ] ,   a_β = |â_β|
â₃  = â₁ × â₂        (unit normal of the reference surface)
```

with `k̃_b` the skew of the **initial curvature** vector `k_b = [−k_{b2}, k_{b1}, k_{b3}]`.
For a **prismatic** segment `x_{1;β}=δ_{1β}` and the curvature reduces to the
contour curvature `k₂₂` of the cross-section line.  This fixes the direction-cosine
matrix between the shell triad `a_i` and the beam triad `b_j`:

```
C^{ab}_{ij} = a_i · b_j ,        C^{ab}_{;α} = − k̃^s_α C^{ab}
```

`C^{ab}_{33}` (normal·beam-normal) appears as a denominator throughout — it is the
projection that makes the drilling rotation solvable.

---

## 2. Beam vs shell frames

Global (macro) beam coords `x_i` (x₁ = reference line); local SG coords `y_i=x_i/ε`.
Beam triad `b_i` rotates to `B_i` on deformation, shell triad `a_i` to `A_i`:

```
C^{Bb}_{ij} = B_i · b_j ,   C^{Aa}_{ij} = A_i · a_j ,   C^{AB} = C^{ab}(Δ − ω̃)
```

The **last identity is the linearisation** (`C^{AB}` differs from `C^{ab}` only by
the small rotation `ω̃`; nonlinear `θ̃ω̃` dropped).

---

## 3. RM kinematics — displacements & rotations  (Dr. Yu RM, pp.1–3)

### 3.1 Displacement
With the warping/fluctuation `w=[w₁,w₂,w₃]` and beam rotation
`θ = [θ₁, −u₃', u₂']` (θ₁ = twist):

```
U  = u + ε w − x̃ θ ,            u^s = C^{ab} U
u^s_{;α} = C^{ab}_{;α} U + C^{ab} U_{;α} = − k̃^s_α u^s + C^{ab} U_{;α}
```

Constraints `⟨⟨w_i⟩⟩ = 0` (average fluctuation vanishes) make the split unique.

### 3.2 Rotation and the drilling constraint
From `Δ − ρ̃ = C^{ab}(Δ − θ̃ − ω̃) C^{ba}`:

```
ρ = C^{ab}(θ + ω)                                   (shell rotation)   (★)
```

The in-plane symmetry `ε₁₂ = ε₂₁` (`A₁·R_{,2}=A₂·R_{,1}`) gives the **drilling
rotation** `φ₃ = ρ₃` in closed form:

```
φ₃ = I_α C^{ab} U_{;α}            (I_α : the two row-operators from ε₁₂=ε₂₁)
```

Substituting `ρ₃ = φ₃` back into (★) **eliminates `ω₃`**:

```
ω₃ = (1/C^{ab}_{33}) ( φ₃ − e₃ᵀ C^{ab}(θ + e_α ω_α) )
ρ  = C^{ab} ( Δ − (e₃ e₃ᵀ / C^{ab}_{33}) C^{ab} ) (θ + e_α ω_α) + (e₃/C^{ab}_{33}) φ₃
```

> **Independent unknowns** = displacement fluctuations `w₁,w₂,w₃` **and the two
> rotation fluctuations `ω₁,ω₂`** → **5 DOF per node**.  This is the whole
> difference from Kirchhoff (which has only `w₁,w₂,w₃` and obtains the rotation
> from `w`'s derivative → C¹/penalty).

### 3.3 Curvatures from rotations
```
ρ_α = C^{ab} U_{;α} + ẽ_α ρ ,        κ^s_α = ρ_{α} ;_α + k̃^s_α ρ
```
Expanding `ρ_α` and keeping the asymptotically-leading terms (the underlined
`w'_s ∼ ε²` drop) yields the explicit `κ_{11}, κ_{22}, κ_{12}+κ_{21}` and the
transverse shears — this is exactly the algebra worked out on the hand-derived
sheets and collected as eq 4.23 below.

---

## 4. Shell strain field (prismatic)  — Opensg_MSG eq 4.23

`Γ_D = [ε₁₁, ε₂₂, 2ε₁₂, κ₁₁, κ₂₂, κ₁₂+κ₂₁]ᵀ` (plate),
`Γ_G = [2γ₁₃, 2γ₂₃]ᵀ` (transverse shear).  `R_n = x₂ẋ₃ − x₃ẋ₂`.

```
ε₁₁     = γ₁₁ + x₃κ₂ − x₂κ₃ + w₁'                          (underlined w₁' = O(ε²))
ε₂₂     = ẋ_η ẇ_η
2ε₁₂    = ẇ₁ + κ₁ R_n + ẋ_η w_η'
κ₁₁     = ẋ_η κ_η + ω₂'/ẋ₂ − (ẋ₃/2ẋ₂) ẇ₁' + (ẋ₃/ẋ₂) κ₁' R_n + …
κ₂₂     = − ω̇₁
κ₁₂+κ₂₁ = − κ₁ + ω̇₂/ẋ₂ + (k₂₂/2)(ẋ₃/ẋ₂)²(ẇ₁ − κ₁R_n)
          − k₂₂(ẋ₃/ẋ₂²)ω₂ − ω₁' − (ẋ₃/ẋ₂²)(k₂₂/2)w₃' + (ẋ₃/2ẋ₂) ẋ_η ẇ_η'
2γ₁₃    = κ₁[ x₂(ẋ₂+ẋ₃²/2ẋ₂) + x₃ẋ₃/2 ] − ẇ₁ ẋ₃/2ẋ₂ + ω₂/ẋ₂
          − w₂' ẋ₃/2 + w₃'(ẋ₂+ẋ₃²/2ẋ₂)
2γ₂₃    = (ẇ₃ ẋ₂ − ẇ₂ ẋ₃) − ω₁
```

**Key RM features** vs Kirchhoff:
- `κ₂₂ = −ω̇₁`, `κ₁₂+κ₂₁` contain only **first** derivatives of the fluctuations
  (Kirchhoff has second derivatives of `w` → C¹). → **C⁰ elements, no penalty.**
- `2γ₁₃, 2γ₂₃ ≠ 0` in general (carry `ω₁, ω₂`); they **vanish** for a straight
  prismatic beam at the **center reference** (eq 4.28), where RM ≡ Kirchhoff.

---

## 5. Constitutive  — Opensg_MSG eq 4.24

```
2Π = [Γ_D; Γ_G]ᵀ [[ D , Y ],[ Yᵀ , G ]] [Γ_D; Γ_G]
```
`D` = 6×6 plate stiffness `[[A,B],[Bᵀ,D_b]]` (from `compute_ABD_matrix`),
`G` = 2×2 transverse-shear stiffness (with shear-correction factors),
`Y` = D–G coupling (`Y = 0` for orthotropic laminates).  For symmetric laminates
`B = 0` and `Y = 0`.

---

## 6. VAM minimisation → Euler–Bernoulli beam stiffness

The zeroth-order (classical) energy is asymptotically correct to `O(μ ε²)`; drop
the underlined `O(ε²)` terms of eq 4.23 and minimise over the fluctuations.

### 6.1 Closed form — symmetric laminate, center ref (eq 4.25 / 4.28)
Isotropic (eq 4.28, `λ₁=0, λ₂=−ν`):
```
ε₁₁ = γ₁₁ + x₃κ₂ − x₂κ₃ ;  ε₂₂ = −ν ε₁₁ ;  2ε₁₂ = 0
κ₁₁ = ẋ_η κ_η ;  κ₂₂ = −ν(ẋ_η κ_η) ;  κ₁₂+κ₂₁ = −2κ₁ ;  2γ₁₃ = 2γ₂₃ = 0
```
Anisotropic symmetric (eq 4.25/4.27): adds
`λ₁ = −(A₂₂A₁₆+A₁₂A₂₆)/(A₂₂A₆₆−A₂₆²)`, `λ₂ = −(A₁₂A₆₆+A₁₆A₂₆)/(A₂₂A₆₆−A₂₆²)`
and the `D₁₂/D₂₂, D₂₆/D₂₂` curvature couplings, with the boundary term
`ζ₂|_boun` carrying the **closed-section (Bredt) shear flow** for torsion.

Then `Γ_D = B(s) [γ₁₁,κ₁,κ₂,κ₃]ᵀ` and the EB stiffness is the contour integral
```
C_EB = ∮ B(s)ᵀ D B(s) ds .
```
> Validated in `validate_tube_classical.py`: this reproduces **EA and bending**
> of Table 3.1 to <0.2%.  **GJ and the transverse-shear terms need the FE solve**
> (the Bredt shear flow lives in the `ζ₂|_boun` / `ω`,`w₁` fields, not in the
> pointwise `B`).

### 6.2 General — FE fluctuation solve  (the real implementation)
Discretise the unknown field `q = [w₁,w₂,w₃,ω₁,ω₂]` on the 1-D contour, build the
strain operators `Γ_D = B_D q + B̄_D ε_b`, `Γ_G = B_G q + B̄_G ε_b` (`ε_b` the 4
beam strains), and minimise
```
2Π(q;ε_b) = ∫ (B_D q + B̄_D ε_b)ᵀ D (B_D q + B̄_D ε_b)
          + (B_G q + B̄_G ε_b)ᵀ G (B_G q + B̄_G ε_b)  ds
```
→ `K_qq q = − K_qe ε_b` with the constraints below → `q = V₀ ε_b`, and
```
C_EB = K_ee + V₀ᵀ K_qe        (4×4 ;  D1 = V₀ᵀ K_qe ≤ 0 reduces stiffness)
```

---

## 7. Timoshenko (first-order) enhancement
Carry the `O(ε²)` (underlined) terms as a perturbation driven by the beam-strain
derivatives `ε_b'`; solve a second fluctuation field `V₁` and condense to the
6×6 Timoshenko stiffness `C = (Q^T A^{-1} C_EB A^{-1} Q)^{-1}`-type relation,
exactly as in the Kirchhoff Hermite code (`finalize_v1_and_compute_deff`).  RM's
advantage is that the transverse-shear flexibility is now **inside** the zeroth
order via `G`, fixing the ~1% shear error noted under Table 3.1.

---

## 8. Finite-element implementation  (`rm/msg_rm.py`)

1. **Mesh**: the cross-section contour as C⁰ Lagrange line elements (linear or
   quadratic). Geometry per element: node coords, unit tangent `ẋ_η`, contour
   curvature `k₂₂`, `R_n` at the quadrature points.
2. **DOF layout**: 5 per node `[w₁,w₂,w₃,ω₁,ω₂]` (vs 6 Hermite DOF/node in the
   Kirchhoff C¹ code). Periodic/closed-loop: merge first=last node.
3. **Operators**: from eq 4.23 build, at each Gauss point,
   - `B_D` (6 × 5·n_node)  — plate strains from `q`;
   - `B̄_D` (6 × 4)         — plate strains from `ε_b` (the `x₃κ₂`, `ẋ_ηκ_η`, `−2κ₁`…);
   - `B_G` (2 × 5·n_node), `B̄_G` (2 × 4) — transverse shears.
4. **Shear locking**: the `2γ` rows lock for thin sections. Use **selective
   reduced integration** of the `G`-energy (one-point) or an **assumed-strain**
   (MITC-like) interpolation of `2γ₁₃,2γ₂₃`. (Cross-check: the locked solution
   over-stiffens GA — compare against the reduced one.)
5. **Constraints**: `⟨w_i⟩ = 0` (3), the derivative-twist condition, and the
   **closed-loop single-valuedness** of `w₁` (Bredt circulation → torsion).
   Impose via Lagrange multipliers / nullspace projection (KKT), as in
   `solve_fluctuation_field`.
6. **Assemble** `K_qq, K_qe, K_ee` by autodiff/explicit `Bᵀ[D|G]B` integration;
   solve `V₀`, form `C_EB`; then the `V₁` step (§7) for the 6×6.

---

## 9. Reference surface — center vs outer (the C13 connection)

- **Center reference** (eq 4.28): `2γ₁₃=2γ₂₃=0` for a straight prismatic beam, so
  RM ≡ Kirchhoff — and the EB stiffness is exact (validates the formulation).
- **Outer reference**: the transverse shears are **non-zero**; RM keeps the energy
  asymptotically correct where the Kirchhoff outer-reference model degrades. This
  is precisely the regime of the **C13 / tension-center** drift seen on
  `jax-msg-shell` (`debug/`), so the outer-reference RM tube is the decisive test
  of whether RM tightens the ext–bend coupling at the OML.

---

## 10. Validation plan
1. `validate_tube_classical.py` — closed-form EA/EI (done).
2. Full FE solve → **isotropic tube, center ref** vs **Table 3.1**
   (C11, C22=C33 shear, C44 GJ, C55=C66 bend); target the shear term Kirchhoff
   misses by ~1%.
3. **Anisotropic tube, center ref** vs **Table 3.2** (tension–torsion C12).
4. **Center vs outer reference** sweep (iso + aniso); compare to the Kirchhoff
   results and the VABS `.K`.
5. Thick-web st15 with RM (shear/torsion terms).

---

## 11. Symbol map to code
| theory | code |
|--------|------|
| `D` (plate ABD) | `compute_ABD_matrix(...)[0]` |
| `G` (transverse shear) | new: from ply `G₁₃,G₂₃` × shear-correction |
| `q=[w₁,w₂,w₃,ω₁,ω₂]` | element DOF vector (5/node) |
| `B_D, B̄_D, B_G, B̄_G` | `rm_strain_operators(...)` (eq 4.23) |
| `V₀`, `C_EB` | `solve_fluctuation_field`, `K_ee+V₀ᵀK_qe` |
| `k₂₂, ẋ_η, R_n` | `compute_element_geometry`, `mesh_curvature` |
