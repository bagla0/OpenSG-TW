# Reissner–Mindlin shell model

## Why RM

The Kirchhoff–Love model ties the wall rotation to the slope of the displacement, so it carries **no**
independent transverse-shear strain. For composite walls, short beams, and dynamics that is too stiff: the
two **transverse-shear stiffnesses** $GA_2,GA_3$ are under-predicted (badly — see {doc}`kirchhoff_love`).

The **Reissner–Mindlin (RM)** shell relaxes the normality constraint: it adds an **independent director
rotation** so the wall normal may rotate relative to the mid-surface. That extra freedom is exactly the
transverse-shear kinematics, and recovering $GA_2,GA_3$ is the RM payoff.

## Kinematics: five d.o.f. per node

Each $C^0$ Lagrange node on the wall arc carries five fluctuation d.o.f.

$$
\big[\,w_1,\;w_2,\;w_3,\;\omega_1,\;\omega_2\,\big],
$$

the three wall displacements plus two director rotations ($\omega_1$ about the beam axis,
$\omega_2$ the through-thickness director tilt). The membrane/bending strain operator $\Gamma_h$ (`BDq` in
`msg_rm_timo`) maps these to the plate strain, and a **separate** transverse-shear operator $\Gamma_g$
(`BGq`) produces the two shear strains

$$
\gamma_{13}=\omega_2,\qquad
\gamma_{23}= \mathbf n\!\cdot\!\frac{\partial \mathbf w}{\partial s}-\omega_1 ,
$$

penalized by the plate **transverse-shear stiffness** $\mathbf G$ (a $2\times2$ block; isotropic limit
$\mathbf G=\tfrac56\,\frac{E}{2(1+\nu)}\,h\,\mathbf I$). The RM constitutive input is therefore the ABD
matrix **plus** $\mathbf G$ per layup. The rigid kernel gains the conjugate constraints
$\langle w_1\rangle=\langle w_2\rangle=\langle w_3\rangle=\langle \omega_1\rangle=0$ (and, for a soft core,
an $\omega_2$ near-null mode — see below).

## Shear locking and the MITC fix

A naive full integration of the $\mathbf G$-energy with low-order elements **locks**: as the wall thins,
the spurious constraint $\gamma_{23}\to 0$ over-stiffens the section and $GA$ blows up. Uniform **reduced**
integration removes locking but leaves the $\omega_2$ antisymmetric mode unpenalized — a *soft-core
hourglass* that **over-softens** $GA_2$.

OpenSG-TW uses the field-consistent **MITC** (Dvorkin–Bathe) **selective assumed-strain** scheme
(`shear="mitc"`, the default):

- $\gamma_{13}=\omega_2$ is algebraic in the d.o.f. and does **not** lock → **full** integration;
- $\gamma_{23}=\mathbf n\!\cdot\!\partial_s\mathbf w-\omega_1$ **is** locking-prone → sample it at the
  Barlow/**tying** points ($\xi=0$ for $p{=}1$; $\pm 1/\sqrt3$ for $p{=}2$), re-interpolate the assumed
  $\bar\gamma_{23}$, then full-integrate.

This passes the **thin-wall limit** (no locking) *and* keeps the soft-core $GA_2$ physical (no hourglass).
The derivation and the field-consistency argument are in `docs/MITC_transverse_shear.md`.

```{admonition} Driver
:class: tip
`strip_RM.rm_timoshenko_6x6(yaml, frac, dshift, curved, shear="mitc")` returns the RM 6×6.
`curved=True` evaluates the wall curvature $\kappa_{22}$ from the geometry (needed for closed/curved
sections); `shear` selects `"mitc"` (default), `"reduced"` (legacy), or `"full"` (locks).
```

## The V1 step is shared

Once $\Gamma_h$ (with the $\mathbf G$-energy folded into the warping stiffness), $\Gamma_\epsilon$ and the
shear-warping $\Gamma_l$ are assembled, RM runs the **same** EB→Timoshenko condensation as every other
solver ({doc}`msg_structure_genome`): saddle-point $V_0$, the $\Gamma_l$-driven $V_1$, and the
`finalize_v1_and_compute_deff` reorder into $[EA,GA_2,GA_3,GJ,EI_2,EI_3]$.

## Accuracy: the shear recovery

On the single-ply $[-45]$ tube, against the 2-D solid reference:

```{list-table}
:header-rows: 1
:widths: 22 26 26 26

* - term
  - KL %err
  - RM (mitc) %err
  - what happened
* - $EA$
  - +0.03
  - −0.02
  - both fine (classical)
* - $GA_2$
  - **−44.5**
  - **−12.9**
  - shear recovered
* - $GA_3$
  - **−68.7**
  - **−12.9**
  - shear recovered
* - $EI_2,EI_3$
  - −16 / −19
  - −12 / −12
  - improved
```

RM cuts the transverse-shear error by **3–5×** and is never worse than KL on any term — the guardrail
`rm/tw_regression_guardrail.py` enforces exactly that (RM ≤ KL on $GA_2,GA_3$) on every thin-walled
benchmark before any solver change ships.

```{admonition} When RM is *not* enough
:class: warning
RM is still a *thin-wall* model. For thick walls ($t/h \gtrsim 8$), soft-core sandwiches, and
multi-wall junctions the reduced shell drifts and you should fall back to the {doc}`jax_solid` model.
`rm/…rm_regime_guard` flags per-station when RM may replace the 2-D solid versus when it will break.
```

```{seealso}
Run it: {doc}`../tutorials/rm_timo_from_yaml`.
```
