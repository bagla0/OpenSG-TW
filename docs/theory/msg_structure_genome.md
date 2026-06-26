# MSG and the Structure Genome

## The idea

A slender composite beam is a 3-D elastic body, but engineering analysis wants a **1-D beam** with a
$6\times6$ stiffness. The classical route (Euler–Bernoulli, Timoshenko, VABS) postulates a kinematic
ansatz for how the cross-section deforms. The **Mechanics of Structure Genome (MSG)** instead *derives*
that ansatz: it asks for the cross-sectional **warping field** that makes the 1-D model's strain energy
equal to the original 3-D strain energy, to the asymptotic order requested.

The smallest repeating piece that carries all the material/geometric heterogeneity is the
**Structure Gene (SG)**. For a prismatic beam the SG is the **cross-section** — either modeled as a
filled **2-D solid** domain, or, for thin walls, as a **1-D arc** (the shell wall) with a through-thickness
plate constitutive law.

## Kinematics: macro strain + warping

Let $x_1$ be the beam axis and $(x_2,x_3)$ the cross-section. The 3-D displacement is split into a
**macroscopic** beam motion plus a **warping** fluctuation $\chi(x_2,x_3)$ that the SG problem solves for:

$$
u_i(x_1,x_2,x_3) \;=\; \underbrace{\bar u_i(x_1) + (\text{rotation}\times \text{position})}_{\text{beam d.o.f.}}
\;+\; \chi_i(x_2,x_3).
$$

The generalized 1-D strain measures are

$$
\boldsymbol{\epsilon} \;=\; \big[\,\gamma_{11},\;\kappa_1,\;\kappa_2,\;\kappa_3\,\big]^{\!\top}
\quad\text{(Euler–Bernoulli, 4)},
$$

augmented by the two **transverse shears** $2\gamma_{12},2\gamma_{13}$ for the Timoshenko (6) model.
The 3-D strain at a point is linear in these and in the warping:

$$
\boldsymbol{\varepsilon}_{3D} \;=\; \Gamma_h\,\chi \;+\; \Gamma_\epsilon\,\boldsymbol{\epsilon},
$$

where $\Gamma_h$ (the *warping* operator) and $\Gamma_\epsilon$ (the *macro* operator) are the kinematic
maps assembled element-by-element. In the code these are `gamma_h`, `gamma_e` (and for the first-order
Timoshenko step, `gamma_l`).

## The variational statement

MSG minimizes the SG strain energy over the warping subject to constraints that remove rigid-body
freedom (otherwise $\chi$ is non-unique):

$$
\min_{\chi}\; \tfrac12 \int_{\text{SG}} \boldsymbol{\varepsilon}_{3D}^{\top}\, \mathbf{C}\, \boldsymbol{\varepsilon}_{3D}\, \mathrm{d}\Omega
\qquad\text{s.t.}\qquad \langle \chi \rangle = 0 .
$$

Discretizing the warping with finite elements ($\chi \to V$, nodal d.o.f.) gives the block system

$$
\begin{aligned}
D_{hh}\,V_0 &= -\,D_{he}, & D_{\text{eff}}^{(4)} &= D_{ee} + V_0^{\top} D_{he},
\end{aligned}
$$

where

- $D_{hh} = \int \Gamma_h^{\top}\mathbf{C}\,\Gamma_h$ — warping stiffness (singular: rigid-body null space),
- $D_{he} = \int \Gamma_h^{\top}\mathbf{C}\,\Gamma_\epsilon$ — warping–macro coupling load,
- $D_{ee} = \int \Gamma_\epsilon^{\top}\mathbf{C}\,\Gamma_\epsilon$ — bare macro stiffness,
- $V_0$ — the **zeroth-order (Euler–Bernoulli) warping**.

$D_{\text{eff}}^{(4)}$ is the **4×4 classical (EB) stiffness** $[EA,GJ,EI_2,EI_3]$.

## Removing the rigid-body null space

$D_{hh}$ has a 4-dimensional kernel (3 rigid translations + the section twist), so the solve is a
**saddle-point** problem. OpenSG-TW builds two operators:

- $\Psi$ — the rigid-body modes (3 translations + $[\,0,-x_3,x_2,\dots]$ twist),
- $D_c$ — the conjugate constraint rows ($\langle w_1\rangle,\langle w_2\rangle,\langle w_3\rangle$ and the
  twist-rate constraint).

The augmented system $\begin{bmatrix}D_{hh}&D_c\\ D_c^{\top}&0\end{bmatrix}\begin{bmatrix}V\\\lambda\end{bmatrix}
=\begin{bmatrix}b\\0\end{bmatrix}$ is solved directly with **PARDISO** (`solve_fluctuation_field`). The load
$b$ is orthogonal to $\Psi$, so the constraint multiplier $\lambda\to 0$ and $V$ is the energy minimizer in
the $D_c^{\top}V=0$ frame — the discrete form of the VABS *Eq. 85* structural-frame projection
($V \leftarrow (\mathbf I - \Psi(D_c^{\top}\Psi)^{-1}D_c^{\top})\,V$).

## Euler–Bernoulli → Timoshenko (the V1 step)

The 4×4 EB model has no transverse shear. Timoshenko's two shear stiffnesses come from a **first-order**
correction: a second warping field $V_1$ driven by the shear-warping operator $\Gamma_l$ (`gamma_l`),
giving the extra matrices $D_{ll},D_{hl},D_{le}$. After the same rigid-body projection, the $6\times6$ is
assembled by the standard MSG/VABS condensation:

$$
\begin{aligned}
\mathbf A &= D_{\text{eff}}^{(4)}, &
\mathbf B &= (D_{hl}^{\top}V_0 + D_{le})^{\top} V_0,\\
\mathbf C_{\!t} &= V_0^{\top}D_{ll}V_0 + V_1^{\top}\big(D_{hl}V_0 + D_{hl}^{\top}V_0 + D_{le}\big), &
Q &= \mathbf A^{-1}\!\begin{bmatrix}0&0\\0&0\\0&-1\\1&0\end{bmatrix},
\end{aligned}
$$

$$
G = \Big(Q^{\top}\big(\mathbf C_{\!t}-\mathbf B^{\top}\mathbf A^{-1}\mathbf B\big)Q\Big)^{-1},
\qquad
Y = \mathbf B^{\top} Q\, G,
\qquad
\mathbf A_t = \mathbf A + Y\,G^{-1}Y^{\top}.
$$

$G$ is the $2\times2$ **shear** block ($GA_2,GA_3$), $Y$ the shear–(extension/bending) coupling, and
$\mathbf A_t$ the shear-corrected classical block. The reordering into
$[EA,GA_2,GA_3,GJ,EI_2,EI_3]$ is `finalize_v1_and_compute_deff`.

```{note}
This EB→Timoshenko condensation (`solve_fluctuation_field` → `prepare_v1_rhs` →
`finalize_v1_and_compute_deff`) is **identical** across the KL, RM and 2-D-solid solvers — only the
kinematic operators $\Gamma_h,\Gamma_\epsilon,\Gamma_l$ and the constitutive law $\mathbf C$ differ. That
shared back-end is why all three return the same 6×6 layout.
```

## What changes between the three models

```{list-table}
:header-rows: 1
:widths: 20 26 26 28

* -
  - $\mathbf C$ (constitutive)
  - SG / element
  - kinematic operators
* - **KL shell**
  - plate **ABD** (3+3)
  - 1-D arc, Hermite-$C^1$ cubic
  - $\Gamma$ from $w$ + arc-slope; no shear
* - **RM shell**
  - **ABD** + transverse-shear $G$
  - 1-D arc, $C^0$ Lagrange + director
  - extra director $\omega$ ⇒ $\gamma_{13},\gamma_{23}$
* - **2-D solid**
  - full **3-D** $6\times6$ $\mathbf C$
  - 2-D triangle/quad, $P_1$
  - $\Gamma$ from the in-plane gradient of $u$
```

See the next three pages for each.

## References

The Mechanics of Structure Genome framework is due to W. Yu (Purdue); the beam reduction follows Yu, Hodges & Ho (2012). Full bibliography with DOIs: {doc}../references.
