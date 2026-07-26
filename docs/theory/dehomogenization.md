# Dehomogenization (3-D recovery)

```{contents}
:local:
:depth: 2
```

## 1. Recovery is the other half of the reduction, not a post-process

The MSG solve of {doc}`msg_structure_genome` does **not** compute a $6\times6$ and discard the rest. It
computes the **warping field** $V_0$ (and the first-order $V_1$) that minimizes the SG strain energy, and the
Timoshenko stiffness is a *contraction* of that solution,
$\mathbf C_{EB}=D_{ee}+V_0^{\top}D_{he}$. The pointwise 3-D displacement, strain and stress of the original
heterogeneous body are therefore **already contained** in the fields the homogenization solved for:

$$
\boldsymbol\varepsilon_{3D}=\Gamma_h\,\chi+\Gamma_\epsilon\,\boldsymbol\epsilon ,
\qquad \chi = V_0\,\boldsymbol\epsilon+\dots
$$

**Dehomogenization** is the evaluation of that expression at a query point. It is *not* a stress-smoothing or
super-convergent-patch post-process: nothing is fitted, extrapolated or averaged to invent a field the analysis
did not have. Two consequences follow, and they are the reason this page exists:

- the recovered field is **energy-consistent by construction** — if you integrate it back you get exactly the
  stiffness that was reported (§3.3), because the same $V_0$ appears in both;
- the recovery inherits **whatever the homogenization assumed**. A recovery built on a different wall law, a
  different reference surface, or a different element than the one that produced the $6\times6$ is not a
  recovery of that model. This is why `examples/TW-paper/xsec_paper/dehom_rm.py` rebuilds step 1 on the **RM
  ring** (the 6-d.o.f. element of {doc}`reissner_mindlin`) rather than reusing the Kirchhoff–Love bundle in
  `opensg_jax/fe_jax/msg_dehom.py`.

For a thin-walled section the reduction was two nested SGs (contour arc, then wall thickness), so the recovery
is **two steps run in reverse**: beam $\to$ shell/plate along the contour, then shell $\to$ 3-D through the
wall.

```{list-table}
:header-rows: 1
:widths: 8 26 30 36

* - Step
  - from
  - to
  - operator reused from homogenization
* - 1
  - beam resultants $F$ (6)
  - plate strains $\mathcal E,\mathcal G$ (6+2) at $(s)$
  - the ring warping $V_0,V_1$ + the element operators $B_{D\bullet},B_{G\bullet}$
* - 2
  - plate strains at $(s)$
  - $\boldsymbol\Sigma(s,z)$ (6) ply-by-ply
  - the through-thickness plate-SG warping $V_0^{p}$ that produced the ABD
```

## 2. Step 1 — beam resultants to shell strains along the contour

### 2.1 Beam strain and the derivative recoveries

The load enters as the beam force/moment resultant in VABS order
$F=[F_1,F_2,F_3,M_1,M_2,M_3]$ (the section integrals $\int\!\boldsymbol\sigma\,\mathrm dA$ that BeamDyn or GEBT
reports). Invert the section stiffness that the RM ring just produced,

$$
\boldsymbol\epsilon=\mathbf C_6^{-1}F,
\qquad
\boldsymbol\epsilon=[\gamma_{11},\,2\gamma_{12},\,2\gamma_{13},\,\kappa_1,\,\kappa_2,\,\kappa_3]^{\top}.
$$

A Timoshenko beam carries shear, so the axial variation of the resultants is not zero: recovery needs the
**axial derivatives** of the classical strain as well as its value. With $R_1$ the fixed 1-D equilibrium
recovery operator ($F'=R_1F$ for the prismatic, unloaded segment) the derivative chain is

$$
F^{(k+1)}=R_1F^{(k)},\qquad
\boldsymbol\epsilon^{(k)}=\mathbf C_6^{-1}F^{(k)},
$$

and each is split into its four classical components plus the two shears, the shears being folded back through
$Q$ (`msg_dehom._macro_recovery`):

$$
\bar{\boldsymbol\epsilon}=[\gamma_{11},\kappa_1,\kappa_2,\kappa_3]^{\top}+Q\,\boldsymbol\gamma^{(1)},
\qquad
\bar{\boldsymbol\epsilon}^{(1)},\;\bar{\boldsymbol\epsilon}^{(2)}\ \text{likewise from }\boldsymbol\gamma^{(2)},\boldsymbol\gamma^{(3)} .
$$

### 2.2 The contour warping and the eight shell strains

The nodal warping and its axial rate are the **same** $V_0,V_1$ the ring solved (`dehom_rm._macro_fields`):

$$
w=V_0\,\bar{\boldsymbol\epsilon}+V_1\,\bar{\boldsymbol\epsilon}^{(1)},
\qquad
w'=V_0\,\bar{\boldsymbol\epsilon}^{(1)}+V_1\,\bar{\boldsymbol\epsilon}^{(2)} ,
$$

each node carrying the six RM d.o.f. $[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$. At element $e$, arc parameter
$\xi$, the **eight** generalized shell strains follow from the *identical* element operators that assembled
$D_{hh},D_{he},D_{hl}$ (`segment_indep.quad_ops_indep`, `_mitc_shear_indep`):

$$
\begin{aligned}
\mathcal E&=[\varepsilon_{11},\varepsilon_{22},2\varepsilon_{12},\kappa_{11},\kappa_{22},2\kappa_{12}]^{\top}
 = B_{De}\,\bar{\boldsymbol\epsilon}+B_{Dh}\,w+B_{Dl}\,w',\\
\mathcal G&=[2\gamma_{13},2\gamma_{23}]^{\top}
 = B_{Ge}\,\bar{\boldsymbol\epsilon}+\tilde B_{Gh}\,w+B_{Gl}\,w' ,
\end{aligned}
$$

with $\tilde B_{Gh}$ the **MITC-tied** shear rows — the same assumed-strain tying used in the stiffness
assembly, so step 1 is the exact adjoint of the RM homogenization rather than a look-alike. Because the ring
carries the wall transverse shears explicitly, $\mathcal G$ is a genuine per-element wall quantity and not a
section-level average (see the caveat in §7.3 on what may be done with it).

### 2.3 Why the contour gradients are evaluated per laminate region

The ring is a **2-node linear** element. Any strain row containing the *contour derivative* of the warping
($2\varepsilon_{12}$, row 2, and $2\kappa_{12}$, row 5) is therefore one order lower than the fields it comes
from: it is element-piecewise and oscillates about the smooth solution. Measured on the IEA station,
element-to-element jumps reach $32\%$ of the mean on row 2 and $182\%$ on row 5, against $\sim10\%$ for the
macro-driven rows (`dehom_rm._flow_nodal_avg`).

The classical derivative-field cure applies: sample those two rows at the element midpoint (the Barlow point of
the linear element, §4 of {doc}`reissner_mindlin`) and **patch-average at shared nodes**, then interpolate
linearly. The restriction is what makes it *gradient-consistent* rather than a smoothing:

- averaging is applied **only where exactly two elements meet** (nodal valence $2$);
- at a **junction node** (valence $\geq3$) the element's own midpoint value is kept;
- rows $0,1,3,4$ are **never** touched — their jumps at a region boundary are physical (different $A,B,D$).

Averaging blindly across a T-junction would smear the genuine discontinuity in the wall law into a spurious
contour-gradient spike, which is exactly the artefact the valence restriction removes.

```{note}
The implementation (`dehom_rm._flow_nodal_avg`) tests **valence only** — it does not compare the layups of the
two incident elements, so a ply-drop node with valence $2$ *is* averaged. Only genuine multi-wall junctions are
protected. Rows $0,1,3,4$ are untouched either way, so the wall-law discontinuity itself is never smeared; the
approximation is confined to the two derivative rows across a thickness change. Note also that the averaging is
**off by default** (`stress_at_points(..., flow_avg=False)`); the spanwise driver enables it explicitly.
```

## 3. Step 2 — shell strains to pointwise 3-D stress

### 3.1 The through-thickness plate SG, reused

The wall constitutive law came from a 1-D through-thickness SG over the ply stack: a warping $V_0^{p}$
satisfying $\mathbf K\,V_0^{p}=-\mathbf F$, with

$$
\mathbf K=\!\int_0^h\! \mathbf B^{\top}\mathbf C\,\mathbf B\,\mathrm dz,\quad
\mathbf F=\!\int_0^h\! \mathbf B^{\top}\mathbf C\,\Gamma_e\,\mathrm dz,\quad
D_{ee}^{p}=\!\int_0^h\!\Gamma_e^{\top}\mathbf C\,\Gamma_e\,\mathrm dz ,
$$

where $\mathbf B(z)$ is the through-thickness strain–displacement operator and $\Gamma_e(z)$ maps the plate
strains onto the 3-D Voigt strain,

$$
\Gamma_e(z):\quad \varepsilon_{11}\!\leftarrow\!\varepsilon_{11}+z\kappa_{11},\quad
\varepsilon_{22}\!\leftarrow\!\varepsilon_{22}+z\kappa_{22},\quad
2\varepsilon_{12}\!\leftarrow\!2\varepsilon_{12}+z\,2\kappa_{12}.
$$

Recovery evaluates the very same objects at the requested depth. The shipped cross-section path calls
`msg_materials.plate_stress_at_depth`; `msg_rm_plate.msgrm_strain_at_depth` is the gradient-augmented
counterpart, which additionally consumes the first-order columns $\bar C_1,\bar C_2$ and the in-plane strain
gradients $\mathcal E_{,1},\mathcal E_{,2}$ (those terms are not active in the blade recovery below, where the
gradients are not formed):

$$
\boxed{\;
\boldsymbol\Gamma(z)=\big(\mathbf B(z)\,V_0^{p}+\Gamma_e(z)\big)\,\mathcal E,
\qquad
\boldsymbol\Sigma(z)=\mathbf C_{\text{layer}}(z)\,\boldsymbol\Gamma(z) \; }
$$

with $\mathbf C_{\text{layer}}$ the rotated $6\times6$ stiffness of **the ply at that depth**. The stress
therefore jumps ply-to-ply exactly as the material does; nothing is homogenized through the thickness.

### 3.2 Where the sample sits, and the reference shift

A query point $(y_2,y_3)$ is projected onto the contour to get $(e,\xi)$ and a signed depth $z$ measured from
the **reference surface**. The plate SG measures depth from the OML ($z\in[0,h]$), so a mid-surface ring must
transport both the depth and the membrane strain (`dehom_rm.stress_at_points`):

$$
z_{\text{OML}}=z+\mathrm{frac}\cdot h,
\qquad
\mathcal E_{1:3}\;\mapsto\;\mathcal E_{1:3}-\mathrm{frac}\cdot h\,\mathcal E_{4:6},
$$

so that the $z\kappa$ term in $\Gamma_e$ stays consistent with the shifted membrane part. Omitting it clamps
every outer-half point to the OML ply — a thick cap then reports the gelcoat instead of the carbon.

### 3.3 Energy consistency (why "the same $V_0^{p}$" matters)

Because $V_0^{p}$ is *the* solution of $\mathbf K V_0^{p}=-\mathbf F$ and $\mathbf K$ is symmetric,

$$
\int_0^h\!\big(\mathbf BV_0^{p}+\Gamma_e\big)^{\!\top}\mathbf C\big(\mathbf BV_0^{p}+\Gamma_e\big)\mathrm dz
= V_0^{p\top}\mathbf KV_0^{p}+2V_0^{p\top}\mathbf F+D_{ee}^{p}
= D_{ee}^{p}+\mathbf F^{\top}V_0^{p}
= \mathbf{ABD}.
$$

Hence

$$
\int_0^h \boldsymbol\Gamma(z)^{\top}\boldsymbol\Sigma(z)\,\mathrm dz \;=\; \mathcal E^{\top}\,\mathbf{ABD}\,\mathcal E ,
$$

**identically** — the recovered pointwise field integrates back to the plate energy that the homogenization
used. Any recovery built on a *different* through-thickness ansatz breaks this identity, and its stress is then
inconsistent with the stiffness it was paired with, by an amount nobody tracks.

## 4. The MSG-RM $8\times8$ wall law, and why not a Whitney closure

The RM ring needs a wall law with a transverse-shear block. OpenSG-TW uses the **MSG-RM $8\times8$**

$$
\begin{bmatrix}N\\M\\Q\end{bmatrix}
=\begin{bmatrix}\mathbf A&\mathbf B&0\\ \mathbf B&\mathbf D&0\\ 0&0&\mathbf G\end{bmatrix}
\begin{bmatrix}\mathcal E_{1:3}\\ \mathcal E_{4:6}\\ \mathcal G\end{bmatrix},
$$

in which $\mathbf G$ ($2\times2$) is obtained by the **VAM route** on the *same* through-thickness SG that gave
$\mathbf A,\mathbf B,\mathbf D$ (Yu's Reissner–Mindlin plate model; `msg_rm_plate.rm_plate_msg`, construction
summarized in §3.1 of {doc}`reissner_mindlin`): a zeroth-order solve gives the classical ABD, gradient-driven
first-order warping columns $\bar C_1,\bar C_2$ give a gradient energy $H$, and $\mathbf G$ follows from a
least-squares projection of the residual energy onto the Reissner form ($\mathbf X=\mathbf G^{-1}$).

The alternative — a **Whitney / complementary-energy shear-flow closure** — postulates the through-thickness
shear *distribution* (a cylindrical-bending parabola with traction-free faces), integrates it to a compliance,
and pairs the result with an ABD it was never derived from. Three things break, all of which matter for
recovery rather than for the $6\times6$:

1. **It is a load assumption, not a solution.** Its shape is fixed before the section is known, so it cannot
   respond to the actual gradient state $\mathcal E_{,1},\mathcal E_{,2}$ at the sample point.
2. **It has no junction notion.** At a web/skin T-junction the local state is nowhere near cylindrical
   bending; the assumed flow is simply wrong there, and there is no residual to tell you so.
3. **It is not the companion of the ABD.** The identity of §3.3 holds for the MSG pair by construction; for a
   grafted closure it holds only accidentally.

The MSG route buys, in exchange: an $8\times8$ that is one asymptotically-ordered object; a **residual**
$U^{*}_{\mathrm{rel}}$ that reports how well the Reissner form actually fits the wall; and — the point for this
page — the stored first-order columns $\bar C_1,\bar C_2$, which let the *same* SG deliver an
equilibrium-consistent through-thickness recovery including the gradient terms,

$$
\boldsymbol\Gamma(z)=\mathbf B\big(V_0^{p}\mathcal E+\bar C_1\mathcal E_{,1}+\bar C_2\mathcal E_{,2}\big)
+\Gamma_e\mathcal E+M_1V_0^{p}\mathcal E_{,1}+M_2V_0^{p}\mathcal E_{,2},
$$

with $M_1,M_2$ the in-plane gradient operators. Nothing is re-postulated at a junction; the wall simply reports
the state it is handed.

```{admonition} Two checks the module itself runs
:class: note
`python examples/TW-paper/xsec_paper/msg_rm_plate.py` verifies that (i) the zeroth-order $\mathbf A_6$
reproduces `compute_ABD_matrix` to machine precision, and (ii) a homogeneous isotropic plate returns the
classical $\mathbf G=\tfrac56 Gh$ shear-correction factor. Switching the wall $\mathbf G$ from the legacy
Whitney value to the MSG value moves the section $6\times6$ by $\le0.02\%$ at the IEA $r/R=0.2$ station — the
*section* stiffness is insensitive, which is precisely why the argument above is about the **recovered
through-thickness field**, not about the $6\times6$.
```

## 5. Displacement recovery and the warping gauge

### 5.1 Warping plus beam kinematics

The recovered $w$ is the **fluctuation**. The physical local displacement of a material point at section
position $r$ adds the 1-D beam motion — translation $u_g$ and rotation $C$ of the beam node at that station,
from the same 1-D solve that produced $F$:

$$
u \;=\; u_g \;+\; C\,(w+r)\;-\;r .
$$

For an RM wall the fluctuation itself is depth-dependent: the ring carries a mid-surface displacement *and* a
director, so a point at depth $z$ moves as

$$
u(z)=u_{\text{mid}}+z\,\big(\boldsymbol\omega\times e_3\big),
$$

which `dehom_rm.disp_at_points(..., director=True)` includes. On the contour ($z\approx0$) the term is
inactive, which is why a circumferential path is insensitive to it and a through-thickness path is not.

### 5.2 The gauge

The warping is only defined up to the rigid modes that were constrained away
($\langle w\rangle=0$ over the SG, §5 of {doc}`reissner_mindlin`). **The constraint set defines a gauge**, and
two codes need not share it: the RM ring normalizes over the **contour**, VABS over the **2-D section area**.
The two differ by a *constant per component* — at the IEA $r/R=0.2$ station the $u_3$ offset is $\sim0.7$ mm,
which is large compared with the warping itself.

The fix is a **gauge transport**, not a correction: before comparing, shift the recovered warping by the
difference of the two averages taken over a **common node set**. It changes no strain, no stress and no energy
(a constant is in the kernel of $\Gamma_h$); it only removes an arbitrary additive constant. Comparing warping
displacement between codes without doing this is meaningless.

## 6. The reference surface propagates into the recovery

The reference surface is a **single argument** — `frac` $=0.5$ (center/mid-surface), $0.0$ (OML), $1.0$ (IML) —
and it must propagate to *all three* places or the model is inconsistent
(`dehom_rm.build_rm_bundle` derives it once from the YAML's `reference` field):

1. the **contour geometry** (which surface the 1-D nodes lie on),
2. the **wall law** — the $z_{\text{ref}}$ of the ABD / $8\times8$,
3. the **recovery depth** — the $z_{\text{OML}}$ transport of §3.2.

Physically the choice sets the **extension–bending coupling $\mathbf B$**. At the mid-surface of a symmetric
stack $\mathbf B\approx0$; at an offset reference $\mathbf B$ is activated by the full offset lever, and its
effect is not uniform across the $6\times6$: the flapwise transverse shear $GA_3$ and the in-plane web shear
degrade first. Measured on the IEA-22 blade over all **51 stations**, mean $|\%\text{err}|$ of the RM
diagonal against the VABS `.K`:

```{list-table}
:header-rows: 1
:widths: 22 13 13 13 13 13 13

* - reference
  - $EA$
  - $GA_2$
  - $GA_3$
  - $GJ$
  - $EI_2$
  - $EI_3$
* - **center** (adopted)
  - 0.85
  - 1.93
  - 1.57
  - 1.57
  - 0.23
  - 2.54
* - OML
  - —
  - —
  - $\approx11$
  - —
  - —
  - —
```

The paper and every tutorial on this site therefore **adopt the center (mid-surface) reference**. The IML is
strictly worse than the OML — it is the same offset lever applied across the full thickness — and is not used.

## 7. Practical validity of a sampling path

### 7.1 The recovery is a through-**thickness** reconstruction

Step 2 reconstructs the field of **one wall, across its own thickness**. A sampling path is valid when it is a
genuine through-thickness column of a single wall (or lies on the contour). It is *not* a general-purpose 2-D
field interpolator: nothing in the model spans the interior of a cell.

### 7.2 The junction is where a naive column fails

Nearest-point projection sends a query point to the closest contour element. Near a **web/skin T-junction**
that element may belong to the *other* wall: a carbon spar-cap point can land on a web element and be evaluated
with the **web layup** (and vice versa). The result is a spurious hot spot precisely at the junction — a model
error, not a rendering artefact.

The fix used in the paper driver is a **material-aware projection**: each query point carries the material of
the source element, and is allowed to project only onto ring elements whose layup *contains* that material
(matched on $E_1$). The cap point can no longer be evaluated on the web. The lightweight alternative used by
the tutorial's circumferential path is to detect and mask the outliers by residual (4 of 124 contour points at
$r/R=0.2$).

This is exactly why the junction is **the demanding test** of a cross-section recovery, and why the shipped
validation walks a *connected* cap $\to$ T-junction $\to$ web polyline instead of two separate paths.

### 7.3 The transverse-shear stresses

$\sigma_{13},\sigma_{23}$ are **not** recovered constitutively by default (`rm_shear=False`). The ring does
carry $\mathcal G$, but a spar cap's transverse shear is an **equilibrium (shear-flow)** effect, and
$\sigma_{13}=G_{13}\gamma_{13}$ from the local wall shear over-predicts it by roughly an order of magnitude with
the wrong through-thickness shape, whatever tying scheme is used (`sweep_rm_shear.py`). The shipped stress
therefore reports the in-plane field plus the plate plane-stress limit; a correct $\sigma_{13},\sigma_{23}$
needs the equilibrium shear flow $q(s)$ and is a separate development. The in-plane $\sigma_{11},\sigma_{22},
\sigma_{12}$ and $\sigma_{33}$ are the validated outputs.

## 8. What is validated and shipped

Both tutorials run standalone from data committed under `examples/data/iea_all_stations/` (51 center-reference
1-D shell YAMLs, the BeamDyn load set, and a small pre-extracted VABS landmark file).

**{doc}`../tutorials/iea_r020_homo_dehom` — the $r/R=0.2$ station, three paths.**
Homogenization first: the RM ring $6\times6$ against the VABS `.K` — every diagonal term within $\sim2.7\%$,
full-matrix Frobenius error $3.15\%$. Then recovery under the station's beam resultant:

```{list-table}
:header-rows: 1
:widths: 30 34 36

* - path
  - what it tests
  - result vs VABS
* - circumferential (on the contour)
  - step 1 all the way round, incl. layup transitions
  - $\sigma_{11}$ 0.6%; $u_1,u_2,u_3$ 0.71 / 0.17 / 0.16% (124 pts, 4 masked)
* - LP spar-cap OML $\to$ IML
  - step 2 through a thick multi-ply wall
  - $\sigma_{11}$ 0.5%, $\sigma_{22}$ 1.5%, $\sigma_{12}$ 4.4%; disp $\le0.22$%
* - cap $\to$ T-junction $\to$ web
  - $C^0$ continuity through the junction
  - 149 samples over 604 mm (T at 458 mm); max adjacent jump $|\Delta u_1|=0.097$, $|\Delta u_2|=|\Delta u_3|=0.025$ mm
```

The third path is the one that matters for the formulation: the displacement is continuous across the
T-junction — natural for the $C^0$ 6-d.o.f. ring, which shares all six nodal d.o.f. at the junction node —
while the stress correctly steps between the two materials.

**{doc}`../tutorials/iea_spanwise` — all 51 stations.** The homogenization $\%$-error panel behind the table of
§6, plus spanwise stress and displacement recovery against the VABS landmark ($\sigma_{11}<1\%$; flapwise tip
deflection $\approx17.7$ m).

```{admonition} Reproducing a number requires reproducing the gauge and the reference
:class: warning
A recovered stress is meaningful only together with (i) the reference surface, (ii) the wall law, and (iii) —
for displacement — the warping gauge. All three are recorded in the 1-D YAML and carried by the bundle; the
`reference` field is the single source of truth and both the homogenization and the recovery read it. Mixing a
center-reference $6\times6$ with an OML-reference recovery is the most common way to get a plausible-looking
wrong answer.
```

## References

The two-step recovery is the standard MSG/VAM dehomogenization (Yu, Hodges & Ho 2012; Yu, Volovoi, Hodges &
Hong 2002 for the validation methodology), applied here on the thin-walled two-SG chain of Deo & Yu. The
Reissner–Mindlin plate law and its least-squares projection follow Yu's RM plate construction; the
complementary-energy shear closure it replaces is the Whitney-type family. Nothing on this page is original.
Full bibliography with DOIs: {doc}`../references`.

```{seealso}
Formulation of the element that supplies $V_0,V_1$: {doc}`reissner_mindlin`.
Run it: {doc}`../tutorials/iea_r020_homo_dehom`, {doc}`../tutorials/iea_spanwise`.
```
