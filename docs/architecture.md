# Software framework

OpenSG-TW computes the Timoshenko $6\times6$ through **three independent branches** — the Reissner–Mindlin
shell, the Kirchhoff–Love shell, and the 2-D solid — that share one MSG core. Each branch is a driver
`.py` exposing a single entry function that returns the stiffness in the order
$[\,EA,\,GA_2,\,GA_3,\,GJ,\,EI_2,\,EI_3\,]$, so the three are drop-in interchangeable. The RM branch
ships **two elements** — a 5-DOF drilling-eliminated strip and a 6-DOF drilling-Lagrange ring — and the
homogenization is followed by a **two-step dehomogenization** that recovers the pointwise 3-D field.

```{image} _img/architecture_timo.svg
:alt: OpenSG-TW three-branch Timoshenko architecture (RM, KL, 2-D solid)
:width: 100%
:align: center
```

| Branch | Driver `.py` | Entry function | Input | Engine / key methods |
|---|---|---|---|---|
| **Reissner–Mindlin shell** — 5-DOF strip | `strip_RM.py` | `rm_timoshenko_6x6` | 1-D shell SG YAML | `msg_rm_timo.assemble_all`, `build_C_Psi`, `transverse_shear.transverse_shear_stiffness` (MITC) |
| **Reissner–Mindlin ring** — 6-DOF, independent $\omega_3$ | `mitc_rm_segment/run_ring_indep.py` (wrapper `xsec_paper/xsec_5v6_master.py`) | `ring_indep` (wrapped as `ring_6dof`) | 1-D shell SG YAML | `segment_indep.quad_ops_indep` / `assemble_constraint` (element-$\lambda$ drilling constraint), `shear="mitc4_g23"` tying |
| **Kirchhoff–Love shell** | `gradient_kirchhoff.py` | `gradient_junction_kirchhoff` | 1-D shell SG YAML | `msg_hermite.solve_tw_from_yaml`, the $\Gamma_e/\Gamma_h/\Gamma_l$ Hermite-$C^1$ operators |
| **2-D solid** | `solid_timo.py` | `compute_timo_from_yaml` | 2-D solid SG YAML | `segment.read_solid_yaml` (tri + quad), `get_heterogeneous_C_matrix` (R_sig), KKT solve |

## The three branches

- **Reissner–Mindlin shell** — a $C^0$ Lagrange contour with an independent director carrying transverse
  shear, fixed against locking by **MITC** assumed strain. This is the branch that recovers $GA_2,GA_3$
  and so **replaces the 2-D solid for thin walls** (see {doc}`theory/reissner_mindlin`). Two element
  variants ship; see [below](rm-elements).
- **Kirchhoff–Love shell** — cubic Hermite $C^1$ contour (3 DOF/node, value + arc-slope), no transverse
  shear; the wall curvatures carry the second contour derivatives and the $V_1$ condensation alone yields
  the beam shear $GA$ (see {doc}`theory/kirchhoff_love`).
- **2-D solid** — a filled $P_1$ tri/quad mesh with the full 3-D $6\times6$ material law; no thin-wall
  reduction, matched to **VABS** (see {doc}`theory/jax_solid`).

(rm-elements)=
## The two RM elements: 5-DOF strip vs 6-DOF ring

Both elements are $C^0$ Lagrange lines on the same 1-D contour and both tie the locking-prone
$\gamma_{23}$ with an MITC assumed strain. They differ in **what happens to the drilling rotation
$\omega_3$**.

```{list-table}
:header-rows: 1
:widths: 22 39 39

* -
  - 5-DOF strip (`strip_RM.py`)
  - 6-DOF ring (`run_ring_indep.ring_indep`)
* - nodal DOF
  - $[w_1,w_2,w_3,\omega_1,\omega_2]$
  - $[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$
* - drilling $\omega_3$
  - **eliminated** in closed form from the in-plane symmetry $\varepsilon_{12}=\varepsilon_{21}$, using
    $1/C^{ab}_{33}$
  - kept **independent**; the same symmetry condition is imposed *weakly* by an **element-wise Lagrange
    multiplier** (`assemble_constraint`, `lam_space="elem"`)
* - shear tying
  - selective MITC (`shear="mitc"` in `msg_rm_timo.assemble_all`)
  - `shear="mitc4_g23"` — tie $\gamma_{23}$ only, since under span invariance $\gamma_{13}$ is algebraic
    in the directors and carries no fluctuation gradient
* - use it for
  - single-wall / smooth contours, and as the historical reference implementation
  - **multi-wall sections with web–skin T-junctions** — the paper and the IEA tutorials
```

Why that matters: the closed-form elimination carries a $1/C^{ab}_{33}$ factor
({doc}`theory/reissner_mindlin`, §2.1), with $C^{ab}_{33}$ the wall normal projected on the beam-frame
$e_3$. It degrades on walls whose normal is nearly orthogonal to that axis — the wall-parallel
orientation a blade shear web takes at a T-junction. Keeping $\omega_3$ as a real DOF and imposing the
symmetry weakly removes that division entirely, so every wall meeting at a junction contributes its own
drilling residual and the shared node stays well conditioned. The element-*constant* multiplier space is
the inf-sup-stable choice: an equal-order nodal multiplier over-constrains under refinement (documented
in `assemble_constraint`'s docstring as an LBB failure that shows up as a drift in the thin-square
$GA_2/GA_3$).

`xsec_5v6_master.py` is the user-facing wrapper: `load_ring(yaml)` builds the ring arrays from a 1-D
shell SG YAML and `ring_6dof(...)` returns the symmetrized Timoshenko $6\times6$ in the usual VABS order.
Running the module directly reproduces the paper's 5-vs-6-DOF comparison against the 2-D solid on the
single-cell tube, the two-cell (webbed) tube and the IEA-22 $r/R=0.2$ and $0.3$ rings.

## Wall constitutive law: the MSG-RM $8\times8$

Every RM wall carries an $8\times8$ plate law

$$
\begin{bmatrix}\mathbf A&\mathbf B&0\\\mathbf B&\mathbf D&0\\0&0&\mathbf G\end{bmatrix},
$$

built by `xsec_paper/msg_rm_plate.py::rm_plate_msg` on the *same* 1-D through-thickness SG
discretization that `msg_materials.compute_ABD_matrix` uses. The $2\times2$ transverse-shear block
$\mathbf G$ is **not** a Whitney / assumed-shear-flow closure. It comes from the MSG/VAM
Reissner–Mindlin plate route:

1. a **zeroth-order** plate-SG solve gives the warping $V_0$ and the classical ABD (it reproduces
   `compute_ABD_matrix` exactly);
2. **first-order**, gradient-driven warping columns are formed and condensed into a **gradient energy**;
3. $\mathbf G$ follows from a **least-squares projection** of the residual energy onto the Reissner form
   (the shear compliance $X=\mathbf G^{-1}$ plus the relaxed-constraint constants).

Sanity checks live in the module: a homogeneous isotropic wall returns exactly $\mathbf G = \tfrac56 Gh$,
and the ABD block reproduces `compute_ABD_matrix`. The section $6\times6$ is fairly insensitive to which
$\mathbf G$ is used ($\le 0.02\%$ on the IEA $r/R=0.2$ ring), but the MSG $\mathbf G$ is the
theory-consistent one and it is what the recovery is built on, so it is the default (`g_source="msg"`).

## The reference surface is one argument

Where the 1-D contour sits through the wall thickness is a **single choice that must propagate
everywhere**, and in OpenSG-TW it does: one `fraction` / `ref` argument sets

- the **contour geometry** (which surface the nodes lie on),
- the **wall law** — the laminate `z_ref` for the ABD *and* for the plate-SG warping behind $\mathbf G$,
- the **recovery depth** in dehomogenization (`stress_at_points` converts the reference-surface depth to
  the plate's OML depth using the same fraction).

| `fraction` | `ref` | surface |
|---|---|---|
| `0.5` | `"center"` | laminate mid-surface — **what the paper adopts** |
| `0.0` | `"oml"` | outer mould line |
| `1.0` | `"iml"` | inner mould line |

`build_rm_bundle` defaults `ref=None`, which reads the `reference` field recorded in the YAML when it was
created — the single source of truth, so homogenization and dehomogenization cannot drift apart.

At an offset reference the extension–bending coupling $\mathbf B$ is switched on, and that is not free:
it degrades the flapwise transverse shear $GA_3$ and the in-plane shear carried by the webs. Measured
over the 51 IEA-22 stations against the VABS $\mathbf K$, the mean $|\%\,\text{err}|$ of the RM $6\times6$
diagonal at the **center** reference is

| | $EA$ | $GA_2$ | $GA_3$ | $GJ$ | $EI_2$ | $EI_3$ |
|---|---|---|---|---|---|---|
| center-ref mean $\lvert\%\,\text{err}\rvert$ | 0.85 | 1.93 | 1.57 | 1.57 | 0.23 | 2.54 |

whereas the **OML** reference pushes the mean $GA_3$ error to roughly **11 %**. IML is strictly worse — a
full-thickness extrapolation lever — and is not used. Reproduce the table in
{doc}`tutorials/iea_spanwise`.

## Dehomogenization: the two-step 3-D recovery

Homogenization gives the beam a $6\times6$; **dehomogenization** is the inverse map, taking a beam
force/moment resultant back to the pointwise 3-D stress, strain and displacement at any cross-section
coordinate. It is two steps, and the second one deliberately reuses machinery from the first.

```{image} _img/dehom_flowchart.png
:alt: Two-step dehomogenization — beam resultants to shell strains to pointwise 3-D stress
:width: 100%
:align: center
```

**Step 1 — beam $\rightarrow$ shell.** Invert the section stiffness to get the beam strains
$\varepsilon = C_6^{-1}F$ (VABS ordering of $F$), then evaluate the generalized **shell/plate strains
along the contour** from the converged warping influence coefficients: the zeroth-order $V_0$ and the
first-order $V_1$ returned by the same ring solve
(`ring_indep(..., return_fields=True)`). For the RM element that is the 6-row
$s_6=[\varepsilon_{11},\varepsilon_{22},2\varepsilon_{12},\kappa_{11},\kappa_{22},2\kappa_{12}]$ **plus**
the 2-row wall transverse shear $s_2=[2\gamma_{13},2\gamma_{23}]$, assembled with the *same* element
operators (`quad_ops_indep`, `_mitc_shear_indep`) that built the $6\times6$ — so step 1 is the exact
adjoint of the homogenization. The evaluation is **gradient-consistent per laminate region**, so the
recovered contour gradients do not spike at layup transitions.

**Step 2 — shell $\rightarrow$ 3-D.** Through the wall thickness, reuse the *same* plate-SG
through-thickness warping that produced the ABD/$8\times8$, so the pointwise stress is

$$
\Sigma(z) = C_{\text{layer}}(z)\,\bigl[\mathbf B(z)\,V_0 + G_e(z)\bigr]\,\varepsilon .
$$

Because the warping is the one that was homogenized with, the recovery is **energy-consistent**:
$\int \Gamma\!:\!\Sigma\,dz$ reproduces $\varepsilon^{\top}\mathbf{ABD}\,\varepsilon$. This is the reason
the through-thickness distribution beats a load-assumed (cylindrical-bending / Whitney-type) closure,
which has no notion of a junction at all.

**Displacement.** The recovered field is the warping plus the beam kinematics,

$$
u = u_g + C\,(w + r) - r ,
$$

with $u_g$ the beam displacement, $C$ the beam rotation tensor, $w$ the warping and $r$ the position in
the section. Within the wall, `disp_at_points` adds the RM director term $z\,(\omega\times e_3)$, which
is what makes a through-thickness path come out right (it is inert on a path lying on the contour).

| Module | What it is |
|---|---|
| `xsec_paper/dehom_rm.py` | the **RM-consistent** dehomogenization — `build_rm_bundle` (homogenize + package $C_6$, $V_0$, $V_1$, geometry, layup/material DBs), `stress_at_points`, `disp_at_points` |
| `opensg_jax/fe_jax/msg_dehom.py` | the Kirchhoff–Love (Hermite) counterpart — `recover_shell_strains`, `dehomogenize`, `stress_at_points`; `dehom_rm` reuses its `_macro_recovery` / `_project_point` helpers, and both share the `msg_materials` plate step |

```{note}
`stress_at_points(..., rm_shear=False)` by default. The RM warping *does* carry the wall transverse
shear $s_2$, but the local constitutive recovery $\sigma_{13}=G_{13}\gamma_{13}$ is not physical for a
spar cap, whose transverse shear is an equilibrium (shear-flow) effect. The shipped stress is therefore
the validated in-plane field, with $\sigma_{13},\sigma_{23}$ at the plate plane-stress limit as in the
KL dehomogenization.
```

Both steps are exercised end-to-end in {doc}`tutorials/iea_r020_homo_dehom` — the IEA-22 $r/R=0.2$
station, RM $6\times6$ vs the VABS $\mathbf K$ (about 3.15 % Frobenius), then recovery along three paths:
circumferential, spar-cap through-thickness OML$\rightarrow$IML (with $\sigma_{11}$ within about 0.5 % of
VABS), and a connected cap $\rightarrow$ T-junction $\rightarrow$ web polyline that demonstrates $C^0$
displacement continuity through the junction. {doc}`tutorials/iea_spanwise` runs all 51 stations. Both
run standalone from data committed under `examples/data/iea_all_stations/` (51 center-referenced 1-D
shell YAMLs, BeamDyn loads, VABS benchmarks and a pre-extracted 51-station VABS landmark file).

## Solver stack

- **JAX** in **float64** (`jax_enable_x64`) provides the element assembly and the JIT'd linear algebra.
  The KL-Hermite and 2-D-solid branches assemble with `jax.vmap` over elements; the RM ring uses
  array-batched element operators (`quad_ops_indep_batch`, `_tie_rows_batch`) and JIT'd kernels for the
  $V_1$ right-hand side and the final $6\times6$ (`msg_solver.prepare_v1_rhs`,
  `finalize_v1_and_compute_deff`, both `@jax.jit`).
- The **constrained (KKT / saddle-point) warping systems** are sparse and go to **pypardiso**
  (Intel MKL PARDISO) — `msg_solver.assemble_kkt` / `solve_fluctuation_field` for the shell branches,
  the same call inside `solid_timo`.
- The cross-section ring KKT is small and dense: `ring_indep` builds the augmented
  $[\,D_{hh}\;D_c;\;D_c^{\top}\;0\,]$ system and factorizes it **once** with a dense LU
  (`scipy.linalg.lu_factor`), reusing the factorization for both the $V_0$ and the $V_1$ solves.
- The tapered-segment reduced block deliberately uses **SuperLU**, not PARDISO: PARDISO's static
  pivoting silently returns a wrong factorization on the ill-conditioned block at a web/skin T-junction
  (see the note in `mitc_rm_segment/segment_element.py`).

## Shared MSG core

All branches reuse the same back-end: `msg_materials` (1-D structure-genome → plate ABD and the
through-thickness warping the recovery reuses), `msg_mesh` (line/area mesh, curvature), `msg_solver`
(the saddle-point KKT solve and the Eq.85 projection), and `timo_report` (the full-$6\times6$,
every-$C_{ij}$ benchmarking). The RM cross-section modules sit outside the installed package, under
`mitc_rm_segment/` and `examples/TW-paper/xsec_paper/`. The full call signatures for all of it are in
the {doc}`api`.
