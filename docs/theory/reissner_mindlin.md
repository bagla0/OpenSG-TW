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

## 2. Kinematics: the director d.o.f. and the drilling rotation

The shell carries displacement fluctuations $w=[w_1,w_2,w_3]$ **and** rotation fluctuations $\omega_i$.
The shell rotation is $\rho^s = C^{ab}(\theta + \varepsilon\,\omega)$ with $C^{ab}=a_i\!\cdot\!b_j$ the
shell-to-beam direction cosines. The **drilling rotation** $\omega_3$ is not free: the in-plane symmetry
$\varepsilon_{12}=\varepsilon_{21}$ ($A_1\!\cdot\!R_{,2}=A_2\!\cdot\!R_{,1}$) constrains it. *How* that
constraint is imposed is the one design decision that splits the two RM drivers in this repository.

### 2.1 Drilling elimination — the 5-d.o.f. strip

The symmetry condition can be solved for $\omega_3$ in closed form,

$$
\omega_3=\frac{1}{C^{ab}_{33}}\Big(\varphi_3 - e_3^{\top}C^{ab}(\theta+e_\alpha\omega_\alpha)\Big),
$$

which **eliminates $\omega_3$** and leaves the **independent unknowns $w_1,w_2,w_3,\omega_1,\omega_2$ →
5 d.o.f./node** (vs 3 for KL). That single fact — two extra rotation d.o.f. with the drilling one condensed —
is the whole difference from Kirchhoff at the kinematic level. This is the element of
`opensg_jax/fe_jax/msg_rm_timo.py` and `opensg_jax/fe_jax/strip_RM.py`, and of the general operator
`mitc_rm_segment/segment_element_general.py`; §3–§8 below describe it.

### 2.2 Independent $\omega_3$ — the 6-d.o.f. drilling-Lagrange ring

The elimination carries a $1/C^{ab}_{33}$, and $C^{ab}_{33}=n\!\cdot\!b_3$ **vanishes on a wall parallel to
the beam-frame $b_3$** — precisely the shear webs and flat walls that carry $GA_3$. For a closed multi-cell
section (a blade with webs) the eliminated element therefore degenerates exactly where it is needed most.

The cross-section formulation used by the *Composites Part B* RM paper and by the IEA-22 tutorials keeps
$\omega_3$ as a **genuine sixth nodal d.o.f.**, $[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$, so that no
reciprocal of $C_{33}$ ever appears (`mitc_rm_segment/segment_indep.py::quad_ops_indep`,
driver `mitc_rm_segment/run_ring_indep.py::ring_indep`, user-facing wrapper
`examples/TW-paper/xsec_paper/xsec_5v6_master.py::ring_6dof`). $\omega_3$ enters **directly**:

$$
\begin{aligned}
\text{curvature:}\quad &\Lambda_a=\omega_{3|a}+x_{1;a}\,\omega_3'
  \;\Rightarrow\; \kappa_{11}\!+\!=x_{3;2}\Lambda_1,\;\;
  \kappa_{22}\!+\!=-x_{3;1}\Lambda_2,\;\;
  \kappa_{12}\!+\!=x_{3;2}\Lambda_2-x_{3;1}\Lambda_1,\\
\text{shear:}\quad &2\gamma_{13}\!+\!=C_{23}\,\omega_3,\qquad 2\gamma_{23}\!+\!=-C_{13}\,\omega_3,\\
\text{membrane:}\quad &\text{unchanged (no }\omega_3).
\end{aligned}
$$

The in-plane symmetry that *defined* $\omega_3$ is re-imposed in its **finite (undivided) form** as the
drilling residual

$$
\mathrm{DR}=C_{33}\,\omega_3+C_{3b}\,\omega_b-\tfrac12 S
=C_{33}\big(\omega_3-\omega_3^{\text{elim}}\big),
$$

$$
\tfrac12 S=\tfrac12\Big[\kappa_1\big(x_{1;1}R_{n2}-x_{1;2}R_{n1}\big)
+w_i'\big(x_{1;1}x_{i;2}-x_{1;2}x_{i;1}\big)
+\big(w_{i|1}x_{i;2}-w_{i|2}x_{i;1}\big)\Big],
$$

which stays finite as $C_{33}\to0$: on a healthy wall ($C_{33}\approx1$) it pins $\omega_3$ to its eliminated
value and reproduces the 5-d.o.f. result; on a flat wall it drops out of the $\omega_3$ column and constrains
$\omega_b$ instead, and $\omega_3$ is then set by its own curvature stiffness — **no singularity**.

$\mathrm{DR}=0$ is enforced **weakly, by a Lagrange multiplier that is constant on each element**
($\langle\mathrm{DR}\rangle_e=0$, `assemble_constraint(..., lam_space="elem")`), not by a penalty. The
multiplier space matters: an **equal-order nodal** multiplier violates the inf–sup (LBB) condition and
over-constrains under refinement — on the thin square the $GA_2/GA_3$ error drifts from $-1/-2\%$ at
$N_C{=}24$ to $-17/-9\%$ at $N_C{=}96$ — while the element-constant space removes the drift entirely.

```{list-table} The two RM elements in this repository
:header-rows: 1
:widths: 20 38 42

* -
  - **5-d.o.f. strip** (§2.1)
  - **6-d.o.f. ring** (§2.2)
* - nodal d.o.f.
  - $[w_1,w_2,w_3,\omega_1,\omega_2]$
  - $[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$
* - drilling condition
  - solved in closed form ($\div\,C_{33}$)
  - element-wise Lagrange multiplier on the finite residual $\mathrm{DR}$
* - flat / web walls
  - degenerates as $C_{33}\to0$
  - regular
* - shear tying
  - selective MITC (§4)
  - MITC on $\gamma_{23}$ only, `mitc4_g23` (§4)
* - used for
  - open and simple closed sections, taper studies
  - **closed multi-cell** sections (blade with webs) — the paper + IEA-22 tutorials
```

Both are RM: same directors, same $8\times8$ wall law, same EB→Timoshenko condensation. Everything in §3–§8
applies to both unless a row explicitly says otherwise.

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

### 3.1 The wall law: the MSG-RM $8\times8$

Written out, the wall constitutive law the ring integrates is the $8\times8$

$$
\begin{bmatrix}N\\M\\Q\end{bmatrix}
=\begin{bmatrix}\mathbf A&\mathbf B&0\\ \mathbf B&\mathbf D&0\\ 0&0&\mathbf G\end{bmatrix}
\begin{bmatrix}\Gamma_D^{\,\text{mem}}\\ \Gamma_D^{\,\text{bend}}\\ \Gamma_G\end{bmatrix}.
$$

$\mathbf G$ is **not** taken from a Whitney / complementary-energy shear-flow closure. It comes from the
**VAM route on the same through-thickness SG** that produced $\mathbf A,\mathbf B,\mathbf D$ — Yu's
Reissner–Mindlin plate model, implemented in `examples/TW-paper/xsec_paper/msg_rm_plate.py::rm_plate_msg`:

1. **zeroth order** — solve $\mathbf K V_0^{p}=-\mathbf F$ over the ply stack
   ($\mathbf K=\int \mathbf B^{\top}\mathbf C\mathbf B$, $\mathbf F=\int\mathbf B^{\top}\mathbf C\Gamma_e$);
   $\mathbf A_6=D_{ee}^{p}+V_0^{p\top}\mathbf F$ reproduces `compute_ABD_matrix` to machine precision;
2. **first order** — the in-plane **gradient** operators $M_1,M_2$ (which route $w_{,1}$ into
   $\varepsilon_{11},2\gamma_{13},\gamma_{12}$ and $w_{,2}$ into $\varepsilon_{22},2\gamma_{23},\gamma_{12}$)
   drive $\mathbf K\bar C_a=-R_a$ with $R_a=T_aV_0^{p}-(U_aV_0^{p}+P_a)$, giving the $12\times12$ gradient
   energy $H_{ab}=V_0^{p\top}W_{ab}V_0^{p}+R_a^{\top}\bar C_b$ over $[\mathcal E_{,1};\mathcal E_{,2}]$;
3. **RM projection** — substitute the Reissner constitutive relation (the equilibrium swap
   $\mathcal E\to R-D_1\gamma_{,1}-D_2\gamma_{,2}$) and minimize the residual gradient energy in
   **least squares** over $\mathbf X=\mathbf G^{-1}$ plus the relaxed-constraint constants; $\mathbf G=\mathbf X^{-1}$.

Two self-checks ship with the module: a homogeneous isotropic plate returns the classical
$\mathbf G=\tfrac56 Gh$, and the projection reports its own relative residual $U^{*}_{\text{rel}}$ — a
built-in measure of how well the Reissner form fits that particular stack. Swapping the legacy Whitney
$\mathbf G$ for the MSG one moves the section $6\times6$ by $\le0.02\%$ at the IEA $r/R=0.2$ station; the
payoff is not in the $6\times6$ but in the **recovered through-thickness field**, because the same SG stores
the first-order columns $\bar C_1,\bar C_2$ that the recovery reuses
({doc}`dehomogenization`).

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

### 4.1 The 6-d.o.f. ring ties $\gamma_{23}$ only

The production scheme for the ring of §2.2 is `shear="mitc4_g23"` — **tie $\gamma_{23}$, leave $\gamma_{13}$
at its Gauss value** — for a reason specific to the independent drilling d.o.f. Under span invariance
$\gamma_{13}$ carries no fluctuation *gradient*: it is **algebraic** in the directors
($2\gamma_{13}=C_{2a}\omega_a+C_{23}\omega_3+\dots$), exactly the role $\omega_2$ plays in the 5-d.o.f.
element. Only $\gamma_{23}$ pairs a differentiated displacement against undifferentiated rotations, so only
$\gamma_{23}$ can lock. Tying $\gamma_{13}$ as well (`mitc4_both`) would **de-penalize the drilling content**
that now lives in that row — the assumed-strain interpolation aliases $\omega_3$ and a director hourglass
appears on flat walls. The selective logic of the table above is therefore preserved, with $\omega_3$ joining
$\omega_2$ on the "algebraic, full integration" side.

**Junctions (multi-cell / web–skin).** Where several walls meet (the internal web of the
{doc}`../tutorials/rm_timo_from_yaml` two-cell tube, or a blade spar–skin T-junction), each wall is its own
$C^0$ strip with its own tying-point assumed $\gamma_{23}$; the strips share the junction node's
$[w_1,w_2,w_3,\omega_1,\omega_2]$ — and, for the 6-d.o.f. ring, $\omega_3$ as well. MITC is applied
**per element**, so the junction inherits a field-consistent shear from every incident wall without a penalty
— this is why RM holds GA2/GA3 across the two-cell junction (KL −13.8/−11.2% → RM −1.1/−0.15%) where the KL
gradient-junction model has no shear director to share. Sharing all six d.o.f. at the junction node is also
what makes the recovered **displacement continuous across a T-junction**, the demanding test in
{doc}`dehomogenization`.

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

### 5.1 The ring's augmented KKT: rigid modes **and** drilling multipliers

The 6-d.o.f. ring has to satisfy two independent constraint families, so its saddle-point system carries two
multiplier blocks. The rigid kernel is unchanged — drilling is *not* a rigid-body mode, so the validated
5-d.o.f. $\Psi$ is embedded per node into the six-d.o.f. layout with a **zero $\omega_3$ column**
(and the $\omega_1$ sign of the twist mode flipped to match the validated kernel). Adding the $P$
element-constant drilling rows $G_c$ of §2.2 gives

$$
\begin{bmatrix}
D_{hh} & G_c^{\top} & D_c\\
G_c    & 0          & 0\\
D_c^{\top} & 0      & 0
\end{bmatrix}
\begin{bmatrix}V\\ \mu\\ \lambda\end{bmatrix}
=\begin{bmatrix}b\\ b_\mu\\ 0\end{bmatrix},
$$

with $\mu$ the drilling multipliers and $\lambda$ the four rigid-body multipliers. The ring is built as a
one-quad-deep prismatic strip whose top node row is **d.o.f.-mapped onto the bottom row** (so all matrices are
per unit length, $/h$), the factorization is formed **once** and reused for the $V_0$ and $V_1$ right-hand
sides, and $V_0,V_1$ are returned with the multiplier rows stripped — this is what
{doc}`dehomogenization` consumes.

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
  - 1-D contour, $C^0$ Lagrange, **5 d.o.f./node** $[w_1,w_2,w_3,\omega_1,\omega_2]$ (strip) or
    **6** with independent $\omega_3$ (ring) — §2
  - 2-D filled mesh, $P_1$ triangle/quad, **3 d.o.f./node** $[u_1,u_2,u_3]$
* - constitutive
  - the MSG-RM $8\times8$: plate **ABD** $\mathbf D$ + VAM transverse-shear $\mathbf G$ (§3.1)
  - full **3-D** $6\times6$ $\mathbf C$ (no thin-wall reduction)
* - transverse shear
  - explicit director $\Rightarrow$ **MITC** needed (locking)
  - emerges from the in-plane gradient; **no locking, no MITC**
* - extra unknowns
  - directors $\omega_1,\omega_2$ (+ $\omega_3$ and its element-wise multipliers $\mu$ for the ring);
    closed-loop Bredt constraint; soft-core $\omega_2$ null mode (§5)
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

For a real composite section the choice is not free, because an offset reference activates the
extension–bending coupling $\mathbf B$ of the $8\times8$ (§3.1) with the full offset as lever. That degrades
the flapwise $GA_3$ and the in-plane web shear first: over the 51 IEA-22 stations the mean $|\%\text{err}|$ of
$GA_3$ is $1.57$ at the center reference and $\approx11$ at the OML. The reference is therefore a **single
argument** carried by the 1-D YAML and propagated identically into the contour geometry, the wall law and the
recovery depth; the paper and the tutorials adopt the **center (mid-surface)** reference. See §6 of
{doc}`dehomogenization` for the full table and the propagation rules.

## 8. Accuracy and the regime guard

On the $[-45]$ tube (vs 2-D solid): KL $GA_2,GA_3 = -44.5\%,-68.7\%$ → RM $= -12.9\%,-12.9\%$; on the
two-cell composite KL $-13.8\%,-11.2\%$ → RM $-1.1\%,-0.15\%$. RM is **never worse than KL** on any term —
the guardrail `scripts/rm_research/tw_regression_guardrail.py` enforces exactly that (RM ≤ KL on $GA_2,GA_3$) on every TW
benchmark before a solver change ships. For thick walls ($t/h\gtrsim8$), soft cores, and the hardest
junctions, fall back to the 2-D solid (an RM-regime guard, shipped in [OpenSG_io](https://github.com/bagla0/OpenSG_io), flags this per-station).

On a **closed multi-cell blade section** the 6-d.o.f. ring of §2.2 is measured against VABS rather than
against KL: at the IEA-22 $r/R=0.2$ station every $6\times6$ diagonal term is within $\sim2.7\%$ (Frobenius
$3.15\%$), and across all **51 stations** the mean $|\%\text{err}|$ of the diagonal is
$EA\,0.85$, $GA_2\,1.93$, $GA_3\,1.57$, $GJ\,1.57$, $EI_2\,0.23$, $EI_3\,2.54$ — at the center reference,
from a 1-D contour model. Run it: {doc}`../tutorials/iea_r020_homo_dehom`, {doc}`../tutorials/iea_spanwise`.

## References

Every formulation on this page is taken from the published literature — nothing here is original. The RM
kinematics and section-shear ordering follow the MSG/VABS reduction (Yu, Hodges & Ho 2012; the MSG-TW
blueprint of Deo & Yu); the $8\times8$ wall law of §3.1 is Yu's variational-asymptotic Reissner–Mindlin plate
construction; the transverse-shear treatment follows the assumed-natural-strain / **MITC** family
(Dvorkin & Bathe 1984, 1985, 1986; Bucalem & Bathe 1993; Lee & Bathe 2010) on the variational basis of
Simo & Hughes (1986); the tying-point / field-consistency analysis is Barlow (1976) and Prathap & Bhashyam
(1982); and the soft-core limitations are Pagano (1970), Altenbach et al. (2015) and Tessler et al. (2009).
Full bibliography with DOIs: {doc}`../references`.

```{seealso}
Run it: {doc}`../tutorials/rm_timo_from_yaml` (5-d.o.f. strip),
{doc}`../tutorials/mitc_5dof_vs_6dof` (head-to-head element comparison),
{doc}`../tutorials/iea_r020_homo_dehom` and {doc}`../tutorials/iea_spanwise` (6-d.o.f. ring on the blade).
What the converged $V_0,V_1$ are then used for: {doc}`dehomogenization`.
```
