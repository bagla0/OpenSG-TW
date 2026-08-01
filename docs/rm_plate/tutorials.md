# Tutorials — four standalone, fully runnable cases

Each tutorial is a **complete standalone script** in
[`docs/rm_plate_manual/`](https://github.com/bagla0/OpenSG-TW/tree/rm-xsec-dehom/docs/rm_plate_manual)
— run it from anywhere inside the repository with no edits:

```bash
python docs/rm_plate_manual/tutorial_case1.py
```

Every script builds its 1-D SG YAML + mesh figure, homogenizes and **prints
the plate law**, solves its benchmark plate problem, dehomogenizes the full
3-D stress and displacement fields, compares against the exact 3-D
elasticity solution, writes the out-of-plane / in-plane / displacement
figures, and evaluates **both** max-stress and Tsai–Wu failure indices from
the recovered ply-frame stresses.  The outputs and figures below are the
actual products of running the scripts.

**User options (every tutorial):**

```text
--model 0     classical ABD: 6x6 lamination matrix, Kirchhoff kinematics
--model 1     RM shear-refined 8x8 (default)
--FF N11 N22 N12 M11 M22 M12 Q1 Q2
              dehomogenize from YOUR resultants (exact comparison skipped)
```

---

## Tutorial 1 — cross-ply [0/90/0], S = 10

Simply supported strip, $a=1$ m, $h=a/10$, top-face pressure
$q_0\sin(\pi x/a)$, $q_0=10$ kPa; graphite/epoxy 25:1, [0/90/0] bottom-up.

```{image} ../_img/rm_plate/tutorial_case1_sg.png
:width: 55%
:align: center
```

```{literalinclude} ../rm_plate_manual/tutorial_case1.py
:language: python
```

**Output of the default run:**

```{literalinclude} ../rm_plate_manual/tutorial_case1_output.txt
:language: text
```

```{image} ../_img/rm_plate/tutorial_case1_outofplane.png
:width: 78%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case1_inplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case1_disp.png
:width: 98%
:align: center
```

**Option demos.**  `--model 0` (classical ABD, Kirchhoff): the same run
reports U₁ 85 %, U₃ 45 %, σ₁₁ 14 % — exactly what the shear refinement buys
at $S=10$.  `--FF 0 0 0 1013.2 0 0 3183.1 0`: dehomogenization from user
resultants (the strip's own statics values), exact comparison skipped.

---

## Tutorial 2 — sandwich [0/core/0], S = 10

Same strip and load; stiff carbon faces (0.1 h each) over a soft core
(0.8 h).  Only the material/layup block differs from Tutorial 1.

```{image} ../_img/rm_plate/tutorial_case2_sg.png
:width: 55%
:align: center
```

```{literalinclude} ../rm_plate_manual/tutorial_case2.py
:language: python
```

**Output:**

```{literalinclude} ../rm_plate_manual/tutorial_case2_output.txt
:language: text
```

```{image} ../_img/rm_plate/tutorial_case2_outofplane.png
:width: 78%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case2_inplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case2_disp.png
:width: 98%
:align: center
```

---

## Tutorial 3 — antisymmetric angle ply [15/−15], L/h = 4

Deliberately thick: $a=4$ in, $h=1$ in, split face load
$\sigma_{33}=\pm(q_0/2)\sin(\pi x/a)$, $q_0=1$ psi.  The off-axis plies
couple: nonzero $N_{22}, M_{22}, M_{12}, Q_2$, $\sigma_{23}$ and $U_2$.

```{image} ../_img/rm_plate/tutorial_case3_sg.png
:width: 55%
:align: center
```

```{literalinclude} ../rm_plate_manual/tutorial_case3.py
:language: python
```

**Output:**

```{literalinclude} ../rm_plate_manual/tutorial_case3_output.txt
:language: text
```

```{image} ../_img/rm_plate/tutorial_case3_outofplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case3_inplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case3_disp.png
:width: 98%
:align: center
```

---

## Tutorial 4 — symmetric angle ply [30/−30/−30/30], L/h = 4

Same geometry and load as Tutorial 3; the bending–twist coupling makes
$M_{12}$, $Q_2$ and $\sigma_{23}$ large ($\sigma_{23}\sim 0.5\,q_0$).

```{image} ../_img/rm_plate/tutorial_case4_sg.png
:width: 55%
:align: center
```

```{literalinclude} ../rm_plate_manual/tutorial_case4.py
:language: python
```

**Output:**

```{literalinclude} ../rm_plate_manual/tutorial_case4_output.txt
:language: text
```

```{image} ../_img/rm_plate/tutorial_case4_outofplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case4_inplane.png
:width: 98%
:align: center
```
```{image} ../_img/rm_plate/tutorial_case4_disp.png
:width: 98%
:align: center
```

---

## Reading the errors

All four cases fall at second order in thickness: halving $h/a$ divides
every error by ≈4.  The $L/h=4$ numbers of Tutorials 3–4 are the
deliberately thick limit; by $S=50$–64 they reach 0.05–0.2 %.  The failure
indices use **example** allowables — replace the `Xt/Xc/Yt/Yc/S12a/...`
block with your material system.
