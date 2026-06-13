# MSG Thin-Walled Beam Formulation

**Mechanics of Structure Genome (MSG) — Kirchhoff Shell Cross-Section**

This document describes the variational formulation implemented in
`opensg_jax/fe_jax/msg_shell.py` for computing the Timoshenko beam stiffness
of a composite thin-walled cross-section.

---

## 1. Problem Definition

A thin-walled composite beam is homogenized in two nested stages:

| Stage | SG dimension | Input | Output |
|-------|-------------|-------|--------|
| 1 | 1D through-thickness | Ply sequence, angles, material props | 6×6 ABD plate stiffness per wall segment |
| 2 | 1D cross-section arc | ABD per segment + cross-section geometry | 6×6 Timoshenko beam stiffness |

Stage 1 is handled by `msg_materials.compute_ABD_matrix`.
Stage 2 is the subject of this document.

---

## 2. Coordinate Systems

### 2.1 Global Frame

```
        x₃
        │
        │       cross-section plane
        │      ╱
        └─────── x₂
       ╱
      x₁  (beam axis, out of the cross-section plane)
```

- **x₁**: beam axis direction (longitudinal)
- **(x₂, x₃)**: cross-section plane
- All cross-section geometry is described in the (x₂, x₃) plane.

### 2.2 Local Shell Frame

At each point along the cross-section contour (arc-length coordinate **s**):

```
                 n̂  (in-plane normal to wall, pointing inward)
                 │
     ────────────┼──────────────  wall midline
                 │   ↗ t̂ (tangent along arc, CCW positive)
                 s
```

| Symbol | Definition |
|--------|-----------|
| s | Arc-length coordinate along the wall midline (CCW positive) |
| t̂ = (ẋ₂, ẋ₃) | Unit tangent to the contour: ẋ_α = dx_α/ds |
| n̂ = (−ẋ₃, ẋ₂) | In-plane normal (rotated 90° CCW from tangent) |
| κ₂₂ | Geodesic curvature of the midline: κ₂₂ = (ẋ₂ẍ₃ − ẋ₃ẍ₂) |
| R_n | Moment arm: R_n(s) = x₂(s) ẋ₃(s) − x₃(s) ẋ₂(s) |

The moment arm R_n represents the perpendicular distance from the centroid to the tangent line;
it appears naturally in the twist-warping coupling.

### 2.3 Through-Thickness Coordinate (Stage 1)

For each wall segment, a local thickness coordinate ξ ∈ [−h/2, h/2] runs through the
laminate plies. The Kirchhoff plate assumption discards ε₃₃ and transverse shear εᵢ₃ within
each wall, yielding the 6×6 ABD stiffness from Stage 1.

---

## 3. Displacement Decomposition

The total displacement field in the cross-section SG is split into:

$$\mathbf{u}(x_1, s) = \mathbf{V}(x_1) + \mathbf{w}^{(0)}(x_1,s) + x_1\,\mathbf{V}^{(1)}(x_1) + x_1\,\mathbf{w}^{(1)}(x_1,s)$$

In the MSG framework at beam level:

| Symbol | Name | Size | Role |
|--------|------|------|------|
| ε = [γ₁₁, κ₁, κ₂, κ₃]ᵀ | Generalized strains | 4 | Euler-Bernoulli beam strains |
| γ = [γ₁₂, γ₁₃]ᵀ | Shear strains | 2 | Timoshenko shear corrections |
| **w** = [w₁, w₂, w₃]ᵀ | Warping field (V0) | N×4 | Fluctuation — cross-section DOFs |
| **v₁** = [v₁, v₂, v₃]ᵀ | Shear warping (V1) | N×4 | Timoshenko correction field |

DOFs at each cross-section node: `[w₁, w₂, w₃]` (3 per node), no rotation DOFs
(Kirchhoff shell — rotations are derived from displacement derivatives).

---

## 4. Strain Operators

Three strain operators map the respective fields to the 6-component Voigt strain vector
in the **local shell frame** [ε₁₁, εₛₛ, 2ε₁ₛ, 2ε₁ₙ, 2εₛₙ, κ_b]ᵀ:

| Index | Component | Physical meaning |
|-------|-----------|-----------------|
| 0 | ε₁₁ | Axial membrane strain |
| 1 | εₛₛ | Tangential membrane strain |
| 2 | 2ε₁ₛ | In-plane shear (x₁–s) |
| 3 | 2ε₁ₙ | Transverse shear (x₁–n), zero for Kirchhoff |
| 4 | 2εₛₙ | Out-of-plane shear / bending curvature |
| 5 | κ_b | Wall bending (twist curvature) |

### 4.1 Macroscale Strain Operator Γ_e  (4 beam strains → 6 shell strains)

At each quadrature point q on element e, the **Ge** matrix (6×4) maps
**ε** = [γ₁₁, κ₁, κ₂, κ₃]ᵀ to shell strains:

$$\boldsymbol{\Gamma}_e\,\boldsymbol{\varepsilon} = \mathbf{G}_e\,\boldsymbol{\varepsilon}$$

$$\mathbf{G}_e = \begin{bmatrix}
1       & 0                             & x_3     & -x_2    \\
0       & 0                             & 0       & 0       \\
0       & R_n                           & \dot{x}_2 & \dot{x}_3 \\
0       & 0                             & 0       & 0       \\
0       & 0                             & 0       & 0       \\
0       & -2 - \tfrac{\kappa_{22}}{2}R_n & 0       & 0
\end{bmatrix}$$

Row-by-row interpretation:

| Row | Strain | Expression |
|-----|--------|-----------|
| 0 | ε₁₁ | γ₁₁ + x₃ κ₂ − x₂ κ₃  (axial + bending) |
| 1 | εₛₛ | 0  (no macroscale tangential membrane from beam strains) |
| 2 | 2ε₁ₛ | R_n κ₁ + ẋ₂ κ₂ + ẋ₃ κ₃  (twist + bending shear) |
| 3 | 2ε₁ₙ | 0  (Kirchhoff: no transverse shear at macro scale) |
| 4 | 2εₛₙ | 0 |
| 5 | κ_b | (−2 − κ₂₂ R_n/2) κ₁  (wall bending from twist) |

### 4.2 Fluctuation Strain Operator Γ_h  (warping w → 6 shell strains)

The warping field **w** = [w₁(s), w₂(s), w₃(s)] introduces the fluctuation strains:

$$\boldsymbol{\Gamma}_h(\mathbf{w}) = \begin{bmatrix}
0 \\
\dot{x}_2\, w_2' + \dot{x}_3\, w_3' \\
w_1' \\
0 \\
-\kappa_{22}\dot{x}_2\, w_2' + \dot{x}_3\, w_2'' - \kappa_{22}\dot{x}_3\, w_3' - \dot{x}_2\, w_3'' \\
\tfrac{\kappa_{22}}{2}\, w_1'
\end{bmatrix}$$

Where $(\,)'$ denotes d/ds (arc-length derivative).

Row-by-row interpretation:

| Row | Strain | Expression |
|-----|--------|-----------|
| 0 | ε₁₁ | 0  (no axial fluctuation in shell kinematics) |
| 1 | εₛₛ | ẋ₂ w₂′ + ẋ₃ w₃′  (tangential membrane from in-plane displacements) |
| 2 | 2ε₁ₛ | w₁′  (in-plane shear from axial warping gradient) |
| 3 | 2ε₁ₙ | 0  (Kirchhoff assumption) |
| 4 | 2εₛₙ | −κ₂₂ẋ₂ w₂′ + ẋ₃ w₂″ − κ₂₂ẋ₃ w₃′ − ẋ₂ w₃″  (bending curvature) |
| 5 | κ_b | (κ₂₂/2) w₁′  (torsional wall bending) |

### 4.3 Timoshenko Shear Strain Operator Γ_l  (shear warping v₁ → 6 shell strains)

The Timoshenko V1 field **v₁** = [v₁(s), v₂(s), v₃(s)] adds the shear-induced strains:

$$\boldsymbol{\Gamma}_l(\mathbf{v}_1) = \begin{bmatrix}
v_1 \\
0 \\
\dot{x}_2\, v_2 + \dot{x}_3\, v_3 \\
0 \\
2\dot{x}_3\, v_2' - \tfrac{\kappa_{22}}{2}\dot{x}_2\, v_2 - 2\dot{x}_2\, v_3' - \tfrac{\kappa_{22}}{2}\dot{x}_3\, v_3 \\
0
\end{bmatrix}$$

Row-by-row interpretation:

| Row | Strain | Expression |
|-----|--------|-----------|
| 0 | ε₁₁ | v₁  (axial strain from shear warping) |
| 1 | εₛₛ | 0 |
| 2 | 2ε₁ₛ | ẋ₂ v₂ + ẋ₃ v₃  (in-plane shear projected onto tangent) |
| 3 | 2ε₁ₙ | 0 |
| 4 | 2εₛₙ | 2ẋ₃ v₂′ − (κ₂₂/2)ẋ₂ v₂ − 2ẋ₂ v₃′ − (κ₂₂/2)ẋ₃ v₃ |
| 5 | κ_b | 0 |

---

## 5. MSG Variational Principle

The MSG energy functional for the cross-section SG is:

$$\Pi(\mathbf{w}) = \frac{1}{2} \int_{\mathcal{S}} \bigl[\boldsymbol{\Gamma}_h(\mathbf{w}) + \boldsymbol{\Gamma}_e\boldsymbol{\varepsilon}\bigr]^T \mathbf{A}_{BD} \bigl[\boldsymbol{\Gamma}_h(\mathbf{w}) + \boldsymbol{\Gamma}_e\boldsymbol{\varepsilon}\bigr]\, ds$$

where **A**_BD is the 6×6 plate stiffness from Stage 1 (ABD homogenized).

Expanding and collecting in matrix form:

$$\Pi = \frac{1}{2} \mathbf{w}^T \mathbf{D}_{hh} \mathbf{w} + \mathbf{w}^T \mathbf{D}_{he} \boldsymbol{\varepsilon} + \frac{1}{2} \boldsymbol{\varepsilon}^T \mathbf{D}_{ee} \boldsymbol{\varepsilon}$$

| Block | Size | Definition |
|-------|------|-----------|
| D_hh | N×N | ∫ Γ_hᵀ A_BD Γ_h ds  (fluctuation stiffness) |
| D_he | N×4 | ∫ Γ_hᵀ A_BD Γ_e ds  (coupling) |
| D_ee | 4×4 | ∫ Γ_eᵀ A_BD Γ_e ds  (direct macroscale energy) |

Minimization with respect to **w** subject to the periodicity and zero-mean
constraints gives the **V0 equation**:

$$\mathbf{D}_{hh}\,\mathbf{V}_0 = -\mathbf{D}_{he}$$

---

## 6. Euler-Bernoulli Stiffness (4×4)

The 4×4 effective beam stiffness is:

$$\mathbf{C}_{EB} = \mathbf{D}_{ee} + \mathbf{V}_0^T \mathbf{D}_{he}$$

The second term is always negative semidefinite (reduces stiffness) because:

$$\mathbf{V}_0^T \mathbf{D}_{he} = -\mathbf{D}_{he}^T\mathbf{D}_{hh}^{-1}\mathbf{D}_{he} \leq 0$$

Columns/rows of C_EB:

| Index | Generalized strain | Stiffness term |
|-------|--------------------|---------------|
| 0 | γ₁₁ (extension) | EA |
| 1 | κ₁ (twist) | GJ |
| 2 | κ₂ (bending x₂) | EI₂ |
| 3 | κ₃ (bending x₃) | EI₃ |

---

## 7. Timoshenko Shear Correction (6×6)

### 7.1 Extended Energy with Shear Warping

Including the Timoshenko shear strains, the energy at the V1 level involves three additional operators:

| Block | Size | Definition |
|-------|------|-----------|
| D_ll | N×N | ∫ Γ_lᵀ A_BD Γ_l ds |
| D_hl | N×N | ∫ Γ_hᵀ A_BD Γ_l ds |
| D_le | N×4 | ∫ Γ_lᵀ A_BD Γ_e ds |

### 7.2 V1 Right-Hand Side with Null-Space Correction

The raw V1 RHS is:

$$\mathbf{b}_{raw} = \mathbf{D}_{hl}\,\mathbf{V}_0 - \bigl(\mathbf{D}_{hl}^T\mathbf{V}_0 + \mathbf{D}_{le}\bigr)$$

Before solving, the null-space component of **b**_raw (spanned by the rigid-body modes Ψ)
must be removed. This is the **Psi/Dc projection**:

$$\mathbf{b} = \mathbf{D}_c \bigl(\Psi^T \mathbf{D}_c\bigr)^{-1} \Psi^T \mathbf{b}_{raw} - \mathbf{b}_{raw}$$

Where:
- **Ψ** ∈ ℝ^{N×4}: null-space matrix of rigid-body modes
  - Columns: [**e₁** translation, **e₂** translation, **e₃** translation, twist mode (0, −x₃, x₂)]
- **D_c = Cᵀ**: transpose of the Lagrange constraint matrix C ∈ ℝ^{4×N}
  - C enforces: ∫w₁ ds = ∫w₂ ds = ∫w₃ ds = ∫(x₃w₂ − x₂w₃) ds = 0

The KKT system for V1 is:

$$\begin{bmatrix} \mathbf{D}_{hh} & \mathbf{C}^T \\ \mathbf{C} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{V}_1 \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{b} \\ \mathbf{0} \end{bmatrix}$$

The same KKT matrix factored for V0 is **reused** here (no refactoring needed).

### 7.3 V1 Null-Space Projection (Post-Solve)

After solving, the residual Ψ-component is removed from V1:

$$\mathbf{V}_1 = \mathbf{V}_{1,raw} - \Psi \bigl(\mathbf{D}_c^T\,\Psi\bigr)^{-1} \mathbf{D}_c^T\,\mathbf{V}_{1,raw}$$

### 7.4 Timoshenko Sub-Matrices

Define the intermediate 4×4 blocks:

$$B_{Tim} = \bigl(\mathbf{D}_{hl}^T\mathbf{V}_0 + \mathbf{D}_{le}\bigr)^T \mathbf{V}_0$$

$$C_{Tim} = \mathbf{V}_0^T \mathbf{D}_{ll} \mathbf{V}_0 + \mathbf{V}_1^T\bigl(\mathbf{D}_{hl}\mathbf{V}_0 + \mathbf{D}_{hl}^T\mathbf{V}_0 + \mathbf{D}_{le}\bigr)$$

C_Tim is symmetrized: $C_{Tim} = \tfrac{1}{2}(C_{Tim} + C_{Tim}^T)$.

### 7.5 Shear Correction Matrix G (2×2)

The shear modes are activated by the Q_base matrix that selects shear load cases:

$$\mathbf{Q}_{base} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & -1 \\ 1 & 0 \end{bmatrix} \in \mathbb{R}^{4\times 2}$$

Columns correspond to γ₁₂ and γ₁₃ shear strains.

The Timoshenko shear flexibility matrix:

$$\mathbf{G}_{Tim}^{-1} = \mathbf{Q}_{Tim}^T \bigl(C_{Tim} - B_{Tim}^T \mathbf{C}_{EB}^{-1} B_{Tim}\bigr) \mathbf{Q}_{Tim}$$

where $\mathbf{Q}_{Tim} = \mathbf{C}_{EB}^{-1}\,\mathbf{Q}_{base}$.

The shear stiffness matrix (2×2): $\mathbf{G}_{Tim} = \bigl(\mathbf{G}_{Tim}^{-1}\bigr)^{-1}$.

### 7.6 Coupling and Modified EB Stiffness

The shear-bending coupling (4×2):

$$\mathbf{Y}_{Tim} = B_{Tim}^T\,\mathbf{Q}_{Tim}\,\mathbf{G}_{Tim}$$

The modified 4×4 Euler-Bernoulli stiffness including Timoshenko correction:

$$\mathbf{A}_{Tim} = \mathbf{C}_{EB} + \mathbf{Y}_{Tim}\,\mathbf{G}_{Tim}^{-1}\,\mathbf{Y}_{Tim}^T$$

### 7.7 Final 6×6 Stiffness Matrix

The full Timoshenko beam stiffness in the ordering
**[γ₁₁, γ₁₂, γ₁₃, κ₁, κ₂, κ₃]**:

$$\mathbf{S}_{6\times6} = \begin{bmatrix}
A_{Tim}^{(00)} & Y_{Tim}^{(0,:)} & A_{Tim}^{(0,1:4)} \\
Y_{Tim}^{(:,0)} & \mathbf{G}_{Tim} & Y_{Tim}^{(:,1:4)} \\
A_{Tim}^{(1:4,0)} & Y_{Tim}^{(1:4,:)} & A_{Tim}^{(1:4,1:4)}
\end{bmatrix}$$

Diagonal entries:

| Index | Symbol | Physical meaning |
|-------|--------|-----------------|
| S[0,0] | EA | Axial (extension) stiffness |
| S[1,1] | GA₁₂ | Shear stiffness in x₂ direction |
| S[2,2] | GA₁₃ | Shear stiffness in x₃ direction |
| S[3,3] | GJ | Torsional stiffness |
| S[4,4] | EI₂ | Bending stiffness about x₂ |
| S[5,5] | EI₃ | Bending stiffness about x₃ |

---

## 8. Periodicity and Constraint Enforcement

### 8.1 Periodic Boundary Conditions

For a **closed** cross-section, the first and last nodes of the CCW chain are merged
(same physical point): the last node's DOFs are redirected to node 0 via
`build_periodic_dof_map`. This reduces N_total → N_unique = N_total − 1.

### 8.2 Lagrange Constraints (Rigid-Body Suppression)

Four integral constraints eliminate the null space of D_hh:

$$\int_{\mathcal{S}} w_1\, ds = 0 \qquad \int_{\mathcal{S}} w_2\, ds = 0 \qquad \int_{\mathcal{S}} w_3\, ds = 0 \qquad \int_{\mathcal{S}} (x_3 w_2 - x_2 w_3)\, ds = 0$$

The constraint matrix C ∈ ℝ^{4×N} is assembled element-wise:

$$\mathbf{C}_{e,\text{row }i,\,\text{dofs}} = \int_{\ell_e} \phi_n(s)\, ds \quad (i = 0,1,2: \text{ translation})$$
$$\mathbf{C}_{e,3,\,w_2\text{-dofs}} = \int_{\ell_e} x_3(s)\,\phi_n(s)\, ds, \qquad \mathbf{C}_{e,3,\,w_3\text{-dofs}} = -\int_{\ell_e} x_2(s)\,\phi_n(s)\, ds$$

### 8.3 KKT System

The constrained V0 solve is:

$$\begin{bmatrix} \mathbf{D}_{hh} & \mathbf{C}^T \\ \mathbf{C} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{V}_0 \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} -\mathbf{D}_{he} \\ \mathbf{0} \end{bmatrix}$$

Solved once via pypardiso (Intel MKL PARDISO). The factored matrix is reused for the V1 solve.

---

## 9. Element Discretization

### 9.1 Quadratic Lagrange Line Elements

Three-node elements on the cross-section arc, with ξ ∈ [0, 1]:

$$N_1(\xi) = 2\xi^2 - 3\xi + 1, \qquad N_2(\xi) = -4\xi^2 + 4\xi, \qquad N_3(\xi) = 2\xi^2 - \xi$$

Physical derivatives: $N_\alpha' = \frac{1}{L_e} \frac{dN_\alpha}{d\xi}$,
$N_\alpha'' = \frac{1}{L_e^2} \frac{d^2N_\alpha}{d\xi^2}$.

Node ordering within each element: [corner₁ (ξ=0), midside (ξ=0.5), corner₂ (ξ=1)].

### 9.2 DOF Layout

| DOF index within element | Quantity |
|--------------------------|----------|
| 0, 3, 6 | w₁ at nodes 1, 2, 3 |
| 1, 4, 7 | w₂ at nodes 1, 2, 3 |
| 2, 5, 8 | w₃ at nodes 1, 2, 3 |

9 DOFs per element, 3 DOFs per node (no rotational DOFs — Kirchhoff constraint enforced
analytically through the second-derivative shape functions N″).

### 9.3 Quadrature

4-point Gauss-Legendre on [0,1]. Exact for polynomial degree up to 7.
Assembly uses JAX `vmap` over elements with energy-based autodiff
(`jax.hessian` for D_hh/D_ll; `jax.jacfwd` composed with `jax.grad` for D_he/D_hl/D_le).

---

## 10. Sign Conventions and Implementation Notes

| Convention | Value |
|------------|-------|
| κ₂₂ for CCW circle (radius R) | −1/R  (negative for convex outward) |
| Twist mode in Ψ column 3 | [0, −x₃, x₂] per node |
| Constraint row 3 | +x₃ on w₂-DOFs, −x₂ on w₃-DOFs |
| D1 formula | D1 = V0ᵀ D_he  (not −V0ᵀ D_he) |
| F_load sign | F_load = −D_he  so  D_hh V0 = F_load |

**ABD matrix index map** (6×6, Voigt ordering for shell):

| Row/Col | 0 | 1 | 2 | 3 | 4 | 5 |
|---------|---|---|---|---|---|---|
| **0** | A₁₁ | A₁₂ | A₁₆ | B₁₁ | B₁₂ | B₁₆ |
| **1** | A₁₂ | A₂₂ | A₂₆ | B₁₂ | B₂₂ | B₂₆ |
| **2** | A₁₆ | A₂₆ | A₆₆ | B₁₆ | B₂₆ | B₆₆ |
| **3** | B₁₁ | B₁₂ | B₁₆ | D₁₁ | D₁₂ | D₁₆ |
| **4** | B₁₂ | B₂₂ | B₂₆ | D₁₂ | D₂₂ | D₂₆ |
| **5** | B₁₆ | B₂₆ | B₆₆ | D₁₆ | D₂₆ | D₆₆ |

The upper-left 3×3 block is the extensional stiffness A, lower-right 3×3 is the bending
stiffness D, and off-diagonal 3×3 blocks are the coupling stiffness B.
