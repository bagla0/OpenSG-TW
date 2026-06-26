# 2-D solid Timoshenko model

## Scope

When the walls are **not thin** — thick spar caps, foam/balsa cores, blunt trailing edges, multi-wall
junctions — the plate (ABD) idealization breaks down. The **2-D solid** model drops the thin-wall
assumption entirely: the SG is the **filled** cross-section, meshed with 2-D triangles/quads, and each
element carries the full **3-D** orthotropic constitutive law. This is the high-fidelity reference, and it
reproduces **VABS** to machine precision.

It is **pure JAX**: `basix` only tabulates the $P_1$ basis and quadrature, `pypardiso` (MKL PARDISO) solves
the sparse saddle-point system, and the orientation, kinematics, assembly and 6×6 finalize are all JAX.
No FEniCSx, no MPI.

## Constitutive law and orientation

Each element's material is built from the orthotropic engineering constants

$$
\big[E_1,E_2,E_3,\;G_{12},G_{13},G_{23},\;\nu_{12},\nu_{13},\nu_{23}\big]
\;\xrightarrow{\;\text{build\_orthotropic\_C}\;}\; \mathbf C_{\text{mat}}\in\mathbb R^{6\times6}
$$

(Voigt order $[11,22,33,23,13,12]$, $S_{44}=1/G_{23},S_{55}=1/G_{13},S_{66}=1/G_{12}$), then rotated to the
global frame by the per-element material triad $[\mathbf e_1,\mathbf e_2,\mathbf e_3]$:

$$
\mathbf C_{\text{glob}} \;=\; \mathbf T(\mathbf R)\,\mathbf C_{\text{mat}}\,\mathbf T(\mathbf R)^{\top},
\qquad \mathbf R=[\,\mathbf e_1\,|\,\mathbf e_2\,|\,\mathbf e_3\,].
$$

$\mathbf T$ is the $6\times6$ Voigt **stress** rotation built with the basis vectors as the **columns** of
$\mathbf R$ (`rotate_C_with_matrix`). The orientation enters two ways in an OpenSG YAML: a per-element
in-plane ply angle (`rotate_C_matrix`) and the per-element frame `elementOrientations`.

```{admonition} Orientation convention (the subtle bit)
:class: important
The YAML stores each frame in cross-section axis order $(x,y,\text{beam})$, while the homogenizer's global
order is $(\text{beam},x,y)$. The reader (`segment.py`) **permutes** each 3-vector
$(x,y,z)\!\to\!(z,x,y)$ before handing it to `rotate_C_with_matrix`. Building $\mathbf T$ from the *columns*
$[\mathbf e_1,\mathbf e_2,\mathbf e_3]$ (not the transpose) is essential — the transpose is the **inverse**
rotation, which is invisible on an isotropic mesh but produces a few-percent error on composites.
```

## Kinematics on the 2-D mesh

The beam axis is out of the mesh plane, so the in-plane gradient of the $P_1$ displacement gives the warping
strain (`gamma_h`, Voigt $[11,22,33,23,13,12]$):

$$
\Gamma_h\mathbf v=\big[\,0,\;v_{2,2},\;v_{3,3},\;v_{2,3}+v_{3,2},\;v_{1,3},\;v_{1,2}\,\big],
$$

the macro operator carries the section position $(x_2,x_3)$,

$$
\Gamma_\epsilon=\begin{bmatrix}1&0&x_3&-x_2\\0&0&0&0\\0&0&0&0\\0&0&0&0\\0&x_2&0&0\\0&-x_3&0&0\end{bmatrix},
$$

and the first-order shear-warping operator is $\Gamma_l\mathbf v=[\,v_1,0,0,0,v_3,v_2\,]$ (`gamma_l`). These
are the exact 2-D-solid operators used by the VABS-matched OpenSG core.

## Rigid-body projection and solve

$D_{hh}$ is singular with a 4-mode rigid kernel (3 translations + section twist
$[\,0,-x_3,x_2\,]$). OpenSG-TW assembles $\Psi$ (the modes) and $D_c$ (the conjugate constraints
$\langle v_1\rangle,\langle v_2\rangle,\langle v_3\rangle$ and the twist-rate
$\langle v_{2,3}-v_{3,2}\rangle$), and solves the augmented saddle-point system with PARDISO. The discrete
**Eq. 85** projection $V \leftarrow (\mathbf I - \Psi(D_c^{\top}\Psi)^{-1}D_c^{\top})V$ pins the warping to
the VABS structural frame, so the result is directly comparable to a VABS `.K`.

The EB→Timoshenko condensation that follows is the **shared** MSG back-end
({doc}`msg_structure_genome`).

## Validation — the full 6×6 vs VABS

On the **MH-104 airfoil** and a **9-web airfoil**, every term of the Timoshenko $6\times6$ — diagonals
*and* couplings — matches the VABS `.K`:

```{list-table}
:header-rows: 1
:widths: 34 22 22 22

* - case
  - max diagonal
  - worst coupling$^\dagger$
  - every term / EA
* - MH-104 (1 airfoil, 21 213 tri)
  - 0.0002%
  - 0.023%
  - ≤ 0.00003%
* - 9-web airfoil
  - 0.0002%
  - 0.0014%
  - ≤ 0.00013%
```

$^\dagger$ over terms with $|\text{VABS}|\ge 10^{6}$; the larger relative values sit on structurally
near-zero couplings (denominator noise). Reproduce with `opensg_jax/fe_jax/benchmark_vabs.py`.

```{seealso}
Run it: {doc}`../tutorials/solid_timo_from_yaml`.
```

## References

The 2-D-solid Timoshenko reduction follows the MSG/VABS variational-asymptotic theory of Yu, Hodges & Ho (2012) and Yu et al. (2002). Full bibliography with DOIs: {doc}../references.
