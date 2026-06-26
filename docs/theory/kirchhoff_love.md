# Kirchhoff–Love shell model

```{contents}
:local:
:depth: 2
```

## 1. Scope and the two-stage reduction

For a **thin-walled** section the wall is a 1-D **arc** carrying a through-thickness **plate (ABD)**
constitutive law. The Kirchhoff–Love (KL) hypothesis — *a normal to the wall mid-surface stays straight
and normal* — drops transverse shear, so KL is exact in the classical limit ($EA$, $EI$, $GJ$) and is the
right model when the walls are genuinely thin. The homogenization is two nested MSG problems:

```{list-table}
:header-rows: 1
:widths: 8 26 33 33

* - Stage
  - SG dimension
  - input
  - output
* - 1
  - 1-D through-thickness
  - ply sequence, angles, materials
  - $6\times6$ plate **ABD** per wall (`compute_ABD_matrix`)
* - 2
  - 1-D cross-section arc
  - ABD per segment + geometry
  - $6\times6$ Timoshenko beam stiffness
```

Stage 1 uses 3-node quadratic Lagrange elements per ply and recovers the CLT $\mathbf A,\mathbf D$ to
machine precision plus the MSG transverse-normal correction. This page is Stage 2
(`opensg_jax/fe_jax/msg_shell.py`; source `docs/MSG_TW_Beam_Formulation.md`).

## 2. Geometry and the local shell frame

Along the contour arc-length $s$ (CCW positive): unit tangent $\hat t=(\dot x_2,\dot x_3)$, in-plane
normal $\hat n=(-\dot x_3,\dot x_2)$, geodesic curvature $\kappa_{22}=\dot x_2\ddot x_3-\dot x_3\ddot x_2$,
and the moment arm $R_n=x_2\dot x_3-x_3\dot x_2$ (perpendicular distance from the centroid to the tangent —
it appears in every twist–warping coupling). The plate strain is written in the local shell Voigt order
$[\varepsilon_{11},\varepsilon_{ss},2\varepsilon_{1s},2\varepsilon_{1n},2\varepsilon_{sn},\kappa_b]$, with
the transverse-shear slot $2\varepsilon_{1n}\equiv0$ (the KL constraint).

## 3. The three strain operators

KL has **3 d.o.f. per node** $[w_1,w_2,w_3]$ (no rotation d.o.f. — rotations come from displacement
derivatives). Three operators map the fields to the 6-component shell strain.

**Macro** $\mathbf G_e$ (beam strains $[\gamma_{11},\kappa_1,\kappa_2,\kappa_3]\to$ shell):

$$
\mathbf G_e=\begin{bmatrix}
1&0&x_3&-x_2\\ 0&0&0&0\\ 0&R_n&\dot x_2&\dot x_3\\ 0&0&0&0\\ 0&0&0&0\\
0&-2-\tfrac{\kappa_{22}}{2}R_n&0&0\end{bmatrix}
\;\Rightarrow\;
\begin{cases}
\varepsilon_{11}=\gamma_{11}+x_3\kappa_2-x_2\kappa_3,\\
2\varepsilon_{1s}=R_n\kappa_1+\dot x_2\kappa_2+\dot x_3\kappa_3,\\
\kappa_b=(-2-\tfrac{\kappa_{22}}{2}R_n)\,\kappa_1.
\end{cases}
$$

**Warping** $\Gamma_h(w)$ (fluctuation $w(s)\to$ shell) — note the **second** $s$-derivatives, the source
of the $C^1$ requirement:

$$
\Gamma_h(w)=\big[\,0,\;\dot x_2 w_2'+\dot x_3 w_3',\;w_1',\;0,\;
-\kappa_{22}\dot x_2 w_2'+\dot x_3 w_2''-\kappa_{22}\dot x_3 w_3'-\dot x_2 w_3'',\;\tfrac{\kappa_{22}}{2}w_1'\,\big]^{\top}.
$$

**Shear-warping** $\Gamma_l(v_1)$ (the Timoshenko first-order field $v_1(s)$):

$$
\Gamma_l(v_1)=\big[\,v_1,\;0,\;\dot x_2 v_2+\dot x_3 v_3,\;0,\;
2\dot x_3 v_2'-\tfrac{\kappa_{22}}{2}\dot x_2 v_2-2\dot x_2 v_3'-\tfrac{\kappa_{22}}{2}\dot x_3 v_3,\;0\,\big]^{\top}.
$$

(The slot-3 transverse shear is zero in all three — that is what "Kirchhoff" means.)

## 4. Variational principle and the EB stiffness

The cross-section energy with $\mathbf C=\text{ABD}$ is

$$
\Pi(w)=\tfrac12\!\int_{\mathcal S}\!\big[\Gamma_h(w)+\mathbf G_e\,\varepsilon\big]^{\top}\mathbf{ABD}
\big[\Gamma_h(w)+\mathbf G_e\,\varepsilon\big]\,ds
=\tfrac12 w^{\top}D_{hh}w+w^{\top}D_{he}\varepsilon+\tfrac12\varepsilon^{\top}D_{ee}\varepsilon,
$$

with $D_{hh}=\!\int\Gamma_h^{\top}\mathbf{ABD}\,\Gamma_h$, $D_{he}=\!\int\Gamma_h^{\top}\mathbf{ABD}\,\mathbf G_e$,
$D_{ee}=\!\int\mathbf G_e^{\top}\mathbf{ABD}\,\mathbf G_e$. Minimizing over $w$ (constraints in §6) gives
$D_{hh}V_0=-D_{he}$ and the **4×4 Euler–Bernoulli** stiffness

$$
\mathbf C_{EB}=D_{ee}+V_0^{\top}D_{he},\qquad V_0^{\top}D_{he}=-D_{he}^{\top}D_{hh}^{-1}D_{he}\preceq0
$$

(the warping always *reduces* stiffness). Diagonal $\mathbf C_{EB}=\mathrm{diag}(EA,GJ,EI_2,EI_3)$.

## 5. The Hermite-$C^1$ element (why, and what it buys)

Because $\Gamma_h$ contains $w''$, the membrane/bending energy is a **4th-order** problem in $w$: a naïve
$C^0$ interpolation admits spurious zero-energy ($C^0$) warping modes that must be killed by an **interior
penalty**. OpenSG-TW instead uses **Hermite-$C^1$ cubic** elements — each corner node carries the
displacement *value and its arc-derivative* (6 d.o.f./node for the 3-component $w$). Continuous slope
removes the spurious modes **without any penalty**, and the derivative-form twist constraint keeps the
$V_1$ right-hand side orthogonal to the rigid kernel.

The cross-section *geometry* uses the 3-node quadratic line so that wall **curvature** $\kappa_{22}$ is
represented; a flat 2-node geometry forces $\kappa_{22}=0$ and loses the curvature contribution to the
closed-section shear/twist coupling. Quadrature: 4-point Gauss–Legendre on $[0,1]$ (exact to degree 7),
assembled with JAX `vmap` + energy autodiff (`jax.hessian` for $D_{hh},D_{ll}$; `jacfwd∘grad` for the
couplings).

```{admonition} Two KL discretizations live in the repo
:class: note
`msg_shell.py` documents a **3-node quadratic-Lagrange** version (3 d.o.f./node, with the interior penalty
for the $C^0$ modes), while the production driver `gradient_kirchhoff.gradient_junction_kirchhoff` uses the
**Hermite-$C^1$** version (no penalty) and adds **multi-cell / web–skin junction** handling. The tutorials
use the latter.
```

## 6. Rigid kernel, constraints, and the KKT solve (the nullspace)

$D_{hh}$ has a 4-mode kernel. The rigid modes and conjugate constraints are

$$
\Psi=\big[e_1,\;e_2,\;e_3,\;(0,-x_3,x_2)\big],\qquad
\textstyle\int w_1=\int w_2=\int w_3=\int(x_3w_2-x_2w_3)=0 ,
$$

assembled element-wise as $C\in\mathbb R^{4\times N}$ ($C_{3}$ uses $+x_3$ on the $w_2$-d.o.f., $-x_2$ on the
$w_3$-d.o.f.). The constrained solve is the saddle-point system

$$
\begin{bmatrix}D_{hh}&C^{\top}\\C&0\end{bmatrix}\begin{bmatrix}V_0\\\lambda\end{bmatrix}
=\begin{bmatrix}-D_{he}\\0\end{bmatrix},
$$

factored once with PARDISO and **reused** for $V_1$. For a closed contour the first/last node merge
(`build_periodic_dof_map`, $N_{\text{unique}}=N-1$). Unlike {doc}`reissner_mindlin`, the KL kernel has
**no director mode** — the twist mode is $(0,-x_3,x_2)$ in displacement only, and there is no soft-core
$\omega_2$ null vector because there is no $\omega_2$.

## 7. Timoshenko condensation (the V1 step)

The first-order field $v_1$ adds $D_{ll},D_{hl},D_{le}$. After the same Eq.85 rigid-body projection of the
RHS and the post-solve $V_1$ projection, the $6\times6$ is the standard MSG/VABS condensation:

$$
\begin{aligned}
B_{T}&=(D_{hl}^{\top}V_0+D_{le})^{\top}V_0, &
C_{T}&=V_0^{\top}D_{ll}V_0+V_1^{\top}(D_{hl}V_0+D_{hl}^{\top}V_0+D_{le}),\\
Q&=\mathbf C_{EB}^{-1}\!\begin{bmatrix}0&0\\0&0\\0&-1\\1&0\end{bmatrix}, &
G&=\big(Q^{\top}(C_{T}-B_{T}^{\top}\mathbf C_{EB}^{-1}B_{T})Q\big)^{-1},
\end{aligned}
$$

$Y=B_{T}^{\top}QG$, $\mathbf A_{T}=\mathbf C_{EB}+YG^{-1}Y^{\top}$, reordered into
$[EA,GA_2,GA_3,GJ,EI_2,EI_3]$ (`finalize_v1_and_compute_deff` — the **same** routine as RM and the 2-D
solid). The $2\times2$ $G$ is the shear block; because KL injects no transverse-shear strain, $G$ is fed
only through the warping and is therefore **under-stiff** on composites — the gap RM closes.

## 8. What differs from RM and from the 2-D solid

```{list-table}
:header-rows: 1
:widths: 22 26 26 26

* - aspect
  - **KL shell**
  - {doc}`reissner_mindlin`
  - {doc}`jax_solid`
* - d.o.f./node
  - 3 $[w_1,w_2,w_3]$
  - 5 (+ directors $\omega_1,\omega_2$)
  - 3 $[u_1,u_2,u_3]$
* - element
  - Hermite-$C^1$ ($w''$ → 4th-order)
  - $C^0$ Lagrange (1st-order)
  - $P_1$ 2-D triangle/quad
* - transverse shear
  - **none** (normal stays normal)
  - explicit director + MITC
  - from in-plane gradient
* - stabilization
  - $C^1$ slope (no penalty)
  - MITC (anti-locking)
  - none needed
* - best at
  - classical $EA,EI,GJ$ (thin)
  - + $GA_2,GA_3$ (thin)
  - everything (the oracle)
```

## 9. Accuracy and when to leave KL

```{list-table}
:header-rows: 1
:widths: 26 24 24 26

* - term
  - iso box (thin)
  - $[-45]$ tube
  - two-cell $[-45]$
* - $EA,EI$
  - ≤ 0.6%
  - 0.03 / −16…−19%
  - ≤ 1%
* - $GJ$
  - −1.9%
  - −23%
  - −0.4%
* - $GA_2,GA_3$
  - −1.2 / −3.0%
  - **−44 / −69%**
  - **−13.8 / −11.2%**
```

KL nails the classical stiffnesses on thin sections and is cheap and robust. The moment the
**transverse-shear** terms matter (composites, short beams, dynamics) the $GA$ collapse shows — switch to
{doc}`reissner_mindlin`; for thick walls / soft cores / the full 3-D response use {doc}`jax_solid`.

```{seealso}
Run it: {doc}`../tutorials/rm_timo_from_yaml` (the KL subsection), {doc}`../tutorials/twocell_m45_asc`.
```

## References

The Kirchhoff-Love / Hermite-C1 reduction follows the MSG/VABS theory of Yu, Hodges & Ho (2012) and the MSG-TW blueprint of Deo & Yu. Full bibliography with DOIs: {doc}../references.
