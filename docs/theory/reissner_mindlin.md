# Reissner–Mindlin shell model

```{contents}
:local:
:depth: 2
```

## 1. Why RM — the transverse shear KL throws away

The Kirchhoff–Love model ties the wall rotation to the slope of the displacement
($C^1$/Hermite), so it carries **no independent transverse-shear strain**. For composite walls, short
beams, and dynamics that is too stiff: the two **transverse-shear stiffnesses** $GA_2,GA_3$ are
under-predicted — by tens of percent on the $[-45]$ tube and the two-cell composite
({doc}`../tutorials/rm_timo_from_yaml`).

The **Reissner–Mindlin (RM)** shell adds an **independent director rotation**: the wall normal may
rotate relative to the mid-surface. That extra freedom *is* the transverse-shear kinematics. The
curvature strains then contain only **first** derivatives of the fluctuations (vs second for KL), so the
element is a plain **$C^0$ Lagrange** line with **no penalty** — at the cost of **shear locking**, which
§4 handles with MITC. (Source: `scripts/rm_research/RM_DERIVATION.md`, `scripts/rm_research/RM_FORMULATION.md`, Opensg_MSG §3.3.)

```{note}
**Material orientation convention** (the axes drawn by `orient_plot` and used by every solver): $e_2$
(blue) is the in-plane ply-flow direction, $e_3$ (green) is the wall normal taken **OML → IML** — from the
outer mould line toward the inner mould line, i.e. into the section interior — and $e_1$ is the out-of-plane
beam axis. The same convention applies to the 1-D shell and the 2-D solid meshes.
```

## 2. Kinematics: five d.o.f. per node and the drilling elimination

The shell carries displacement fluctuations $w=[w_1,w_2,w_3]$ **and** rotation fluctuations $\omega_i$.
The shell rotation is $\rho^s = C^{ab}(\theta + \varepsilon\,\omega)$ with $C^{ab}=a_i\!\cdot\!b_j$ the
shell-to-beam direction cosines. The in-plane symmetry $\varepsilon_{12}=\varepsilon_{21}$
($A_1\!\cdot\!R_{,2}=A_2\!\cdot\!R_{,1}$) fixes the **drilling rotation** $\rho_3=\varphi_3$ in closed
form, which **eliminates $\omega_3$**:

$$
\omega_3=\frac{1}{C^{ab}_{33}}\Big(\varphi_3 - e_3^{\top}C^{ab}(\theta+e_\alpha\omega_\alpha)\Big).
$$

So the **independent unknowns are $w_1,w_2,w_3,\omega_1,\omega_2$ → 5 d.o.f./node** (vs 3 for KL). That
single fact — two extra rotation d.o.f. with the drilling one condensed — is the whole difference from
Kirchhoff at the kinematic level.

## 3. The strain field (Opensg_MSG eq 4.23)

With $\dot{(\,)}=\partial/\partial\zeta_2$ (contour), $R_n=x_2\dot x_3-x_3\dot x_2$, beam strains
$\varepsilon_b=[\gamma_{11},\kappa_1(\text{twist}),\kappa_2,\kappa_3]$, the plate strain
$\Gamma_D=[\varepsilon_{11},\varepsilon_{22},2\varepsilon_{12},\kappa_{11},\kappa_{22},\kappa_{12}{+}\kappa_{21}]$
and the **two transverse shears** $\Gamma_G=[2\gamma_{13},2\gamma_{23}]$ are

$$
\begin{aligned}
\kappa_{22} &= -\dot\omega_1, &
2\gamma_{13} &= \tfrac{\omega_2}{\dot x_2} + \kappa_1\big[x_2(\dot x_2+\tfrac{\dot x_3^2}{2\dot x_2})+\tfrac{x_3\dot x_3}{2}\big]-\tfrac{\dot x_3}{2\dot x_2}\dot w_1 - \tfrac{\dot x_3}{2}w_2' + (\dot x_2+\tfrac{\dot x_3^2}{2\dot x_2})w_3',\\
\kappa_{12}{+}\kappa_{21} &= -\kappa_1+\tfrac{\dot\omega_2}{\dot x_2}+\dots, &
2\gamma_{23} &= (\dot w_3\dot x_2-\dot w_2\dot x_3)-\omega_1.
\end{aligned}
$$

The key features vs Kirchhoff: $\kappa_{22},\kappa_{12}{+}\kappa_{21}$ contain only **first** derivatives
of the fluctuations ($\Rightarrow C^0$, no penalty), and $2\gamma_{13},2\gamma_{23}\neq 0$ carry the new
directors $\omega_1,\omega_2$. The constitutive law is the block

$$
2\Pi=\begin{bmatrix}\Gamma_D\\\Gamma_G\end{bmatrix}^{\!\top}
\begin{bmatrix}\mathbf D&\mathbf Y\\\mathbf Y^{\top}&\mathbf G\end{bmatrix}
\begin{bmatrix}\Gamma_D\\\Gamma_G\end{bmatrix},
$$

with $\mathbf D$ the plate ABD ($6\times6$, from `compute_ABD_matrix`), $\mathbf G$ the $2\times2$
transverse-shear stiffness, and $\mathbf Y=0$ for orthotropic laminates. The leading-order section-shear
energy is $\mathcal O(\zeta^2)$ (ASC `bagla2025asc` Eq. 17).

## 4. Shear locking and the MITC cure (the detail, incl. junctions)

In the code the two shear rows are

```
2*gamma13 = omega2                  (+ geometric/curvature terms)   <- ALGEBRAIC in the DOF
2*gamma23 = n . dw/ds - omega1                                       <- LOCKING-PRONE
```

`gamma23` pairs a **differentiated** displacement ($\dot w$, one order lower) against an
**undifferentiated** rotation ($\omega_1$): in the thin/stiff limit the discrete field cannot drive
$\gamma_{23}\to0$ pointwise without spurious constraints — **transverse-shear locking**.

**MITC / assumed-natural-strain** (Dvorkin–Bathe 1984/86; Bathe–Dvorkin MITC4) is the full-rank cure that
does *not* rely on under-integration (so no hourglass). Sample the locking-prone strain at the
**tying = Barlow = optimal-sampling points**, re-interpolate an assumed lower-order shear, tie it back to
the nodal d.o.f., then **fully integrate** (a Hu–Washizu mixed form with the assumed-strain parameters
statically condensed; Simo–Hughes 1986). Tying points (Barlow 1976):

- $p{=}1$ (linear): one tying point $\xi=0$ → assumed shear **constant**;
- $p{=}2$ (quadratic): two tying points $\xi=\pm1/\sqrt3$ → assumed shear **linear** between them.

OpenSG-TW uses a **selective** scheme (`opensg_jax/fe_jax/msg_rm_timo.py::assemble_all`, `shear="mitc"`, the default):

```{list-table}
:header-rows: 1
:widths: 26 18 56

* - shear row
  - integration
  - why
* - $2\gamma_{13}=\omega_2$
  - **full**
  - algebraic, does not lock; reduced-int would leave the $\omega_2$ antisymmetric mode unpenalized → soft-core hourglass
* - $2\gamma_{23}=n\!\cdot\!\dot w-\omega_1$
  - **assumed-strain**
  - locking-prone → sample at the tying point(s), re-interpolate $\bar\gamma_{23}$, then full-integrate
```

```{admonition} Field-consistency caveat ($p=1$) and why integration is not the composite cure
:class: note
For the **2-node linear** element the assumed-*constant* $\gamma_{23}$ integrated fully is *algebraically
identical* to 1-point reduced integration (Prathap–Bhashyam 1982) — so at $p{=}1$ MITC is a provably
anti-locking **refactor** that reproduces the validated `reduced` answer (guardrail drift ≤ 0.01% on every
TW case) and only diverges from reduced at $p{=}2$. Across the entire `tube_thesis_314` $R/h$ sweep
(`scripts/rm_research/debug_sweep_lock.py`) `full == reduced` to 0.00%, i.e. **no locking was ever actually triggered** in a
validated case; the reduced rule only ever *under-integrated the soft core*.
```

**Junctions (multi-cell / web–skin).** Where several walls meet (the internal web of the
{doc}`../tutorials/rm_timo_from_yaml` two-cell tube, or a blade spar–skin T-junction), each wall is its own
$C^0$ strip with its own tying-point assumed $\gamma_{23}$; the strips share the junction node's
$[w_1,w_2,w_3,\omega_1,\omega_2]$. MITC is applied **per element**, so the junction inherits a
field-consistent shear from every incident wall without a penalty — this is why RM holds GA2/GA3 across the
two-cell junction (KL −13.8/−11.2% → RM −1.1/−0.15%) where the KL gradient-junction model has no shear
director to share.

## 5. The RM rigid kernel and constraints (the nullspace, derived)

$D_{hh}$ is singular: rigid-body modes cost no energy. RM's kernel and the conjugate constraints
(`opensg_jax/fe_jax/msg_rm_timo.py::build_C_Psi`) are:

$$
\Psi=\Big[\underbrace{[1,0,0,0,0]}_{w_1},\;\underbrace{[0,1,0,0,0]}_{w_2},\;\underbrace{[0,0,1,0,0]}_{w_3},\;
\underbrace{[0,-y_3,\,y_2,\,-1,\,0]}_{\text{twist}}\Big],
$$

i.e. **3 translations + the section twist** $(w_2{=}-y_3,\,w_3{=}y_2,\,\omega_1{=}-1)$ — note the twist
mode now also rotates the **director** $\omega_1$, which the 3-d.o.f. KL kernel cannot. The conjugate
constraints (Lagrange rows $C$, $D_c=C^{\top}$) pin the averages

$$
\langle w_1\rangle=\langle w_2\rangle=\langle w_3\rangle=\langle\omega_1\rangle=0,
$$

plus, for a closed cell, the **single-valuedness of $w_1$** around the loop — the **Bredt circulation** that
carries closed-section torsion (so $GJ$ comes from the $w_1,\omega$ fields, not the pointwise operator).
The V0 and V1 fluctuation solves use the same saddle-point + Eq.85 projection as every other solver
({doc}`msg_structure_genome`).

```{admonition} A 5th nullspace mode for soft cores — the omega2 near-null vector
:class: important
For a **soft-core sandwich** wall the transverse-shear $\mathbf G$ drops ~100×, and the director $\omega_2$
acquires a **near-null** mode that lies in **no** rigid-body mode (it is purely a director hourglass). If the
Eq.85 $V_1$ projection misses it, the soft wall-$G_s$ leaks into the section shear. `build_C_Psi(...,
w2null=True)` augments $\Psi$ with a **constant-$\omega_2$ column** and adds a $\langle\omega_2\rangle$
constraint row (`w2null="id"` constrains every $\omega_2$ d.o.f.). This is the one nullspace piece that has
**no analogue in the 2-D solid model** — it exists only because RM carries an explicit director d.o.f.
```

## 6. What differs from the 2-D solid model

```{list-table}
:header-rows: 1
:widths: 26 36 38

* - aspect
  - RM shell
  - {doc}`jax_solid`
* - SG / element
  - 1-D contour, $C^0$ Lagrange, **5 d.o.f./node** $[w_1,w_2,w_3,\omega_1,\omega_2]$
  - 2-D filled mesh, $P_1$ triangle/quad, **3 d.o.f./node** $[u_1,u_2,u_3]$
* - constitutive
  - plate **ABD** $\mathbf D$ + transverse-shear $\mathbf G$ (+ shear-correction)
  - full **3-D** $6\times6$ $\mathbf C$ (no thin-wall reduction)
* - transverse shear
  - explicit director $\Rightarrow$ **MITC** needed (locking)
  - emerges from the in-plane gradient; **no locking, no MITC**
* - extra unknowns
  - directors $\omega_1,\omega_2$; closed-loop Bredt constraint; soft-core $\omega_2$ null mode (§5)
  - none beyond the 4 rigid modes
* - validity
  - thin walls; degrades for thick/soft-core
  - all regimes; **the oracle** (matches VABS to ~1e-6)
```

**The extra you find vs the solid** is exactly the director machinery: (i) MITC to defeat the
$\gamma_{23}$ locking the solid never has; (ii) the closed-loop $w_1$ single-valuedness for Bredt torsion;
(iii) the $\omega_2$ soft-core null vector of §5 — none of which the 2-D solid needs, because the solid
resolves the through-wall shear directly with the in-plane displacement gradient.

```{admonition} The soft-core limit: MITC is necessary but not sufficient
:class: warning
MITC cures the *numerical* pathology (locking + reduced-int hourglass) **within single-director FSDT
kinematics**. It does **not** repair the *physical* inadequacy of one director for a soft-core sandwich, where
the true through-thickness shear is zig-zag (concentrated in the core). The composite mh104 GA2 over-softens
~20% for this reason, and it is a documented homogenization-coupling artifact: the warping routes the soft
wall-$G_s$ into the membrane-carried section shear. Commercial codes (Abaqus 3.6.8, Ansys SHELL181) keep the
transverse-shear block **separate** from the membrane $A_{66}$ and steer soft-core sandwiches to solid
elements. So for soft cores the **2-D solid is the correct oracle**; do not inherit $\kappa=5/6$ and do not
blindly floor $\mathbf G$. (Full discussion + citations: `docs/MITC_transverse_shear.md`.)
```

## 7. Center vs outer reference

At the **center reference** of a straight prismatic beam the transverse shears vanish (eq 4.28) and
RM $\equiv$ KL — which validates the formulation (EA/EI exact). At the **outer (OML) reference** the shears
are non-zero and RM keeps the energy asymptotically correct where KL degrades — the regime that matters for
the extension–bending (C13) coupling at the OML.

## 8. Accuracy and the regime guard

On the $[-45]$ tube (vs 2-D solid): KL $GA_2,GA_3 = -44.5\%,-68.7\%$ → RM $= -12.9\%,-12.9\%$; on the
two-cell composite KL $-13.8\%,-11.2\%$ → RM $-1.1\%,-0.15\%$. RM is **never worse than KL** on any term —
the guardrail `scripts/rm_research/tw_regression_guardrail.py` enforces exactly that (RM ≤ KL on $GA_2,GA_3$) on every TW
benchmark before a solver change ships. For thick walls ($t/h\gtrsim8$), soft cores, and the hardest
junctions, fall back to the 2-D solid (an RM-regime guard, shipped in [OpenSG_io](https://github.com/bagla0/OpenSG_io), flags this per-station).

## References

Every formulation on this page is taken from the published literature — nothing here is original. The RM
kinematics and section-shear ordering follow the MSG/VABS reduction (Yu, Hodges & Ho 2012; the MSG-TW
blueprint of Deo & Yu); the transverse-shear treatment follows the assumed-natural-strain / **MITC** family
(Dvorkin & Bathe 1984, 1985, 1986; Bucalem & Bathe 1993; Lee & Bathe 2010) on the variational basis of
Simo & Hughes (1986); the tying-point / field-consistency analysis is Barlow (1976) and Prathap & Bhashyam
(1982); and the soft-core limitations are Pagano (1970), Altenbach et al. (2015) and Tessler et al. (2009).
Full bibliography with DOIs: {doc}`../references`.

```{seealso}
Run it: {doc}`../tutorials/rm_timo_from_yaml`.
```
