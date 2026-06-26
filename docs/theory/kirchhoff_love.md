# Kirchhoff–Love shell model

## Scope

For a **thin-walled** section the wall is modeled as a 1-D **arc** in the cross-section plane carrying a
through-thickness **plate (ABD)** constitutive law. The Kirchhoff–Love (KL) hypothesis — *a normal to the
wall mid-surface stays straight and normal* — drops transverse shear, so KL is exact in the classical limit
(EA, $EI$, $GJ$) and is the right model when the walls are genuinely thin.

The SG is the wall arc $s$; at each arc point the plate constitutive matrix is the $6\times6$ **ABD**

$$
\begin{bmatrix}\mathbf N\\ \mathbf M\end{bmatrix}
=\begin{bmatrix}\mathbf A & \mathbf B\\ \mathbf B^{\top} & \mathbf D\end{bmatrix}
\begin{bmatrix}\boldsymbol\epsilon^0\\ \boldsymbol\kappa\end{bmatrix},
$$

built from the layup with the MSG **1-D plate** through-thickness solve (`compute_ABD_matrix`), which uses
3-node quadratic Lagrange elements per ply and recovers the classical CLT $\mathbf A,\mathbf D$ to machine
precision while adding the MSG transverse-normal correction.

## Hermite-$C^1$ arc element

The wall displacement along the arc must have a **continuous slope** to represent bending without spurious
zero-energy ($C^0$) warping modes and without an interior-penalty stabilization. OpenSG-TW therefore uses
**Hermite-$C^1$ cubic** elements: each corner node carries the displacement *value* and its *arc-derivative*
(6 d.o.f. per node for the 3-component wall displacement). The cross-section *geometry* uses the 3-node
quadratic line so that wall **curvature** $\kappa_{22}$ is represented (a flat 2-node geometry forces
$\kappa_{22}=0$ and loses the curvature contribution to the closed-section shear/twist coupling).

The warping strain operator $\Gamma_h$ maps the Hermite d.o.f. to the plate strain
$[\,\epsilon_{11},\epsilon_{22},2\epsilon_{12},\kappa_{11},\kappa_{22},\kappa_{12}\,]$ along the arc, with
the wall-curvature $\kappa_{22}$ entering through the geometric term. The macro operator $\Gamma_\epsilon$
maps the beam strains $[\gamma_{11},\kappa_1,\kappa_2,\kappa_3]$ to the same plate-strain space using the
arc position $(x_2,x_3)$ and tangent.

## Assembly and solve

With $\Gamma_h,\Gamma_\epsilon$ and $\mathbf C=\text{ABD}$ the KL solver assembles

$$
D_{hh}=\!\int\!\Gamma_h^{\top}\mathbf C\,\Gamma_h\,\mathrm ds,\quad
D_{he}=\!\int\!\Gamma_h^{\top}\mathbf C\,\Gamma_\epsilon\,\mathrm ds,\quad
D_{ee}=\!\int\!\Gamma_\epsilon^{\top}\mathbf C\,\Gamma_\epsilon\,\mathrm ds,
$$

(4-point Gauss per element), then runs the **shared** MSG condensation from the
{doc}`msg_structure_genome` page: the saddle-point EB solve gives $V_0$ and the 4×4 $D_{\text{eff}}$, and the
$\Gamma_l$ first-order step gives the 6×6. Because the Hermite basis already has continuous slope, the
twist constraint is imposed in derivative form and **no interior penalty is needed**.

```{admonition} Driver
:class: tip
`gradient_kirchhoff.gradient_junction_kirchhoff(yaml, frac, dshift)` returns the KL `(C6, …)`.
`frac` selects the through-thickness reference surface (0 = OML); `dshift` offsets the cross-section line to
the wall mid-surface when the YAML nodes are on the OML. A multi-cell / junction-aware variant
(`gradient_junction_…`) handles web–skin junctions where several walls meet.
```

## Accuracy and limits

KL reproduces the classical stiffnesses essentially exactly:

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - term
  - single-cell box (iso)
  - $[-45]$ tube (composite)
* - $EA$
  - +0.0%
  - +0.03%
* - $EI_2,EI_3$
  - ±0.6%
  - −16% / −19%
* - $GJ$
  - −1.9%
  - −23%
* - $GA_2,GA_3$
  - −1.2% / −3.0%
  - **−44% / −69%**
```

For the **isotropic box** every term — including shear — is within a few percent because the walls are very
thin and shear-flexibility is negligible. For the **composite tube** the bending/twist terms drift and the
**transverse-shear** terms collapse: KL has *no* transverse-shear kinematics, so $GA_2,GA_3$ are recovered
only through the warping and are badly under-predicted. That is precisely the gap the
{doc}`reissner_mindlin` model closes.

```{seealso}
Run it: {doc}`../tutorials/kl_timo_from_yaml`.
```
