---
title: Independent-$\omega_3$ transverse-shear (GA3) fix
---

# Independent-$\omega_3$ transverse-shear (GA3) fix

On a **flat-walled** tapered section the RM shell under-predicts the beam transverse-shear
stiffness $GA_3$ ($C_{33}$) — by **$-24\%$** (isotropic) to **$-40\%$** ($[-45]$) on a thin
tapered **square** tube — while every other $6\times6$ term stays within a few percent. This
page shows the root cause, the fix (carry the drilling rotation $\omega_3$ as an **independent
DOF** and impose the in-plane symmetry with a **Lagrange multiplier**), and the validation on
all eight cases $\{\text{square},\text{circle}\}\times\{\text{thin},\text{thick}\}\times\{\text{iso},[-45]\}$
at strong taper ($a_R=0.7$) against the FEniCS 3-D solid.

All inputs, the driver script, and the full-$6\times6$ `.dat` are bundled under
[`mitc_rm_segment/taper_indep_study/`](https://github.com/bagla0/OpenSG-TW/tree/main/mitc_rm_segment/taper_indep_study).

## Why $GA_3$ collapses on a flat wall

The general RM taper operator eliminates the drilling rotation $\omega_3$ algebraically from the
in-plane symmetry $\epsilon_{12}=\epsilon_{21}$,

$$
\omega_3=\frac{S}{2\,C_{33}}-\frac{C_{3\beta}}{C_{33}}\,\omega_\beta,\qquad C_{33}=n\cdot b_3 ,
$$

and substitutes it into the curvature and transverse-shear strains. On the two square walls that
**carry the $V_3$ shear flow** the normal is $n\parallel b_2$, so $C_{33}=n\cdot b_3\equiv 0$ over
the *entire* wall and the elimination is degenerate. The resulting eliminated-then-regularised
drilling is a frame-non-objective, $\sim\!-24..-40\%$ mis-representation of the flat-wall tapered
transverse shear. It is a **converged formulation error** (flat under mesh refinement), it grows
$\propto(\mathrm{d}R/\mathrm{d}z)^2$, and it is **specific to flat walls** — the **circular** tube,
whose walls are curved and cross $C_{33}=0$ only at isolated points, is clean and symmetric.

:::{note}
The value of the $1/C_{33}$ regularisation does **not** set $GA_3$ (floor / $\varepsilon$ /
frame-rotation sweeps all leave it at $\sim\!-24\%$). The defect is the *eliminated* drilling
being singular/non-objective on a flat wall, not the size of the regularised denominator.
:::

## The fix — independent $\omega_3$ with a Lagrange constraint

Keep the pre-elimination kinematics: $\omega_3$ is a genuine nodal DOF
$[w_1,w_2,w_3,\omega_1,\omega_2,\omega_3]$ used **directly** (no $1/C_{33}$ anywhere),

$$
\Lambda_\alpha=\omega_{3|\alpha}+x_{1;\alpha}\,\omega_3',\qquad
2\gamma_{13}\mathrel{+}= x_{3;2}\,\omega_3,\quad 2\gamma_{23}\mathrel{-}= x_{3;1}\,\omega_3,
$$

and re-impose the in-plane symmetry in its **finite** (undivided) product form as a constraint

$$
DR \;=\; C_{33}\,\omega_3 + C_{3\beta}\,\omega_\beta - \tfrac{1}{2}S \;=\; 0
\qquad(=C_{33}(\omega_3-\omega_3^{\text{elim}}),\ \text{finite even at }C_{33}=0).
$$

Where $C_{33}\neq0$ the constraint pins $\omega_3$ to its eliminated value (recovering the general
operator); where $C_{33}=0$ it drops $\omega_3$ and the energy sets it from its own curvature
stiffness — no singularity. `DR=0` is imposed **exactly** by one **nodal Lagrange multiplier** per
node (weak form $\langle N_a\,DR\rangle=0$), assembled into the augmented KKT system

$$
\begin{bmatrix} D_{hh} & G^{T}\\ G & 0\end{bmatrix}
\begin{Bmatrix} w\\ \lambda\end{Bmatrix}
=\begin{Bmatrix} -D_{he}\\ -G_e\end{Bmatrix},
$$

after which the existing MSG $V_0/V_1$ condensation runs unchanged. **No penalty, no tuning
parameter.**

- operator + constraint: [`segment_indep.py`](https://github.com/bagla0/OpenSG-TW/blob/main/mitc_rm_segment/segment_indep.py) (`quad_ops_indep`, `assemble_constraint`)
- solve: [`run_indep.py`](https://github.com/bagla0/OpenSG-TW/blob/main/mitc_rm_segment/run_indep.py) → `shell_solve_lagrange`

## Results — all eight cases at strong taper ($a_R=0.7$)

Transverse shear $GA_3$ ($C_{33}$) and $GA_2$ ($C_{22}$), **%err vs the 3-D solid**, general
(eliminated $\omega_3$) vs the fix (independent $\omega_3$, Lagrange):

```{list-table}
:header-rows: 1
:widths: 30 17 17 17 17

* - case ($a_R=0.7$)
  - $GA_3$ general
  - $GA_3$ **indep**
  - $GA_2$ general
  - $GA_2$ indep
* - square · thin · iso
  - $-24.4\%$
  - **$-4.0\%$**
  - $-1.1\%$
  - $-6.3\%$
* - square · thin · $[-45]$
  - $-39.9\%$
  - **$-4.3\%$**
  - $+5.5\%$
  - $-11.5\%$
* - square · thick · iso
  - $-5.7\%$
  - $-5.5\%$
  - $-5.5\%$
  - $-5.0\%$
* - square · thick · $[-45]$
  - $-1.3\%$
  - $+0.2\%$
  - $+8.6\%$
  - $-1.4\%$
* - circle · thin · iso
  - $+3.7\%$
  - $+3.7\%$
  - $+4.0\%$
  - $+4.1\%$
* - circle · thin · $[-45]$
  - $+4.9\%$
  - $+5.0\%$
  - $+5.1\%$
  - $+5.2\%$
* - circle · thick · iso
  - $+2.1\%$
  - $+2.1\%$
  - $+3.6\%$
  - $+3.7\%$
* - circle · thick · $[-45]$
  - $+3.1\%$
  - $+3.0\%$
  - $+4.3\%$
  - $+4.3\%$
```

- **Pathological square-thin case fixed**: $GA_3$ $-24.4\%/-39.9\%\to-4.0\%/-4.3\%$.
- **No regression**: the circle (any regime) is *unchanged* by the fix (it never hit the flat-wall
  degeneracy), and the thick square stays good.
- **$GA_2$ trade**: on the thin square $GA_2$ softens a few percent (the discretisation cost of
  representing $\omega_3$ as a field), but $GA_2\approx GA_3$ symmetry — physically required for the
  square — is *restored* (general gave $-1.1/-24.4\%$; the fix gives $-6.3/-4.0\%$).

### Full $6\times6$ — worst case (square, thin, $[-45]$)

Diagonal (×$10^9$) and the non-zero couplings; the **$GA_3$–$EI_2$ coupling $C_{36}$** is fixed
alongside $GA_3$ itself:

```{list-table}
:header-rows: 1
:widths: 20 20 20 20 12 12

* - term
  - solid
  - general
  - indep-lag
  - gen %err
  - ind %err
* - $C_{11}$ EA
  - 1.5435
  - 1.5676
  - 1.5626
  - $+1.6\%$
  - $+1.2\%$
* - $C_{22}$ GA2
  - 0.3404
  - 0.3591
  - 0.3012
  - $+5.5\%$
  - $-11.5\%$
* - $C_{33}$ **GA3**
  - 0.3404
  - 0.2048
  - 0.3257
  - $\mathbf{-39.9\%}$
  - $\mathbf{-4.3\%}$
* - $C_{44}$ GJ
  - 0.6887
  - 0.7016
  - 0.6269
  - $+1.9\%$
  - $-9.0\%$
* - $C_{55}$ EI2
  - 0.7472
  - 0.7717
  - 0.7528
  - $+3.3\%$
  - $+0.8\%$
* - $C_{66}$ EI3
  - 0.7472
  - 0.7322
  - 0.7606
  - $-2.0\%$
  - $+1.8\%$
* - $C_{36}$ (GA3–EI2)
  - $+1.589\!\times\!10^8$
  - —
  - —
  - $-39.7\%$
  - $\mathbf{+4.6\%}$
```

(GJ softens to $-9.0\%$ under the exact constraint — a residual alongside $GA_2$, most likely the
field-$\omega_3$ discretisation and the free-$\omega_3$ end condition, both open refinements.)

## Reproduce

```powershell
# env: prepend the conda env to PATH (Windows DLLs)
cd mitc_rm_segment
python run_taper_indep_study.py     # 8 cases, general + indep-Lagrange vs solid -> the .dat
```

Bundle ([`taper_indep_study/`](https://github.com/bagla0/OpenSG-TW/tree/main/mitc_rm_segment/taper_indep_study)):
`run_taper_indep_study.py` (driver), `taper_indep_results.dat` (full $6\times6$, all cases),
`meshes/{square,circle}_{shell,solid}_<tag>.yaml` (all 16 input meshes). Solid references:
`examples/data/benchmark/taper_{square,study}_solid_{iso,m45}.npz`. Background and the failed
approaches (drop, denominator reframe, frame rotation) are in
[`mitc_rm_segment/GA3_investigation_and_fix.txt`](https://github.com/bagla0/OpenSG-TW/blob/main/mitc_rm_segment/GA3_investigation_and_fix.txt).
