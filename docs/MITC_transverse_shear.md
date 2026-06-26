# MITC / Assumed-Strain Transverse Shear in the RM Thin-Walled Solver

**Status:** stage-1 implemented (`opensg_jax/fe_jax/msg_rm_timo.py:assemble_all`, `shear="mitc"` default).
**Living document** — update the *Change log* (bottom) on every transverse-shear change.
**Owner test:** `scripts/rm_research/tw_regression_guardrail.py` (must PASS before any solver change ships).

---

## 1. Why this exists

The RM thin-walled cross-section element (`opensg_jax/fe_jax/msg_rm.py`, `opensg_jax/fe_jax/msg_rm_timo.py`) is a C0 Lagrange
line element, 5 DOF/node `[w1,w2,w3,omega1,omega2]`, with two transverse-shear strains

```
2*gamma13 = omega2                  (+ geometric/curvature terms)   <- algebraic in the DOF
2*gamma23 = n . dw/ds - omega1      (n = wall normal, s = arclength) <- locking-prone
```

`gamma23` pairs a *differentiated* displacement (`dw/ds`, one order lower) against an *undifferentiated*
rotation (`omega1`): in the thin/stiff limit the discrete field cannot make `gamma23 -> 0` pointwise
without spurious constraints = **transverse-shear locking**. The original code used **selective reduced
integration** of the whole G-energy to relax that. That choice has TWO problems we diagnosed:

1. It reduced-integrates the **non-locking** `gamma13 = omega2` term as well (unnecessary).
2. For **soft-core sandwich walls** (G drops ~100x) the reduced 1-point rule **under-integrates** the
   real `gamma23` variation, over-softening the section chordwise shear **GA2** (mh104 composite
   GA2 = -20.8% vs the 2D-solid; see `mh104_center_ref/MEMORY.md`).

MITC / assumed-natural-strain is the standard, full-rank cure for locking that does **not** rely on
under-integration (so no hourglass / spurious modes).

## 2. MITC / assumed-strain — the idea

Instead of computing transverse shear from the (locking-prone) displacement interpolation, **sample the
shear strain at "tying points" where it is locking-free, interpolate an assumed (lower-order) shear field
from those samples, tie it back to the nodal DOFs, then FULLY integrate** the energy. Variationally a
Hu-Washizu mixed form with the assumed-strain parameters statically condensed (Simo & Hughes 1986).
Equivalent "field-consistency" view (Prathap): locking is field-inconsistency; re-interpolating the
constrained strain at the **Barlow points** (optimal sampling points) removes it.

**Tying points = Barlow points = reduced-Gauss points** (Barlow 1976):
- linear element (p=1): single tying point at `xi = 0` (center) -> assumed shear = CONSTANT.
- quadratic element (p=2): two tying points at `xi = +-1/sqrt(3)` -> assumed shear = LINEAR between them.

**Recipe (per element):**
```
ty   = [0]                  (p=1)   |  [-1/sqrt3, +1/sqrt3]   (p=2)
Nas  = [1]                  (p=1)   |  [(1-sqrt3*xi)/2, (1+sqrt3*xi)/2]  (p=2)
Bshear_tying[k] = Bshear(ty[k])          # locking-prone row evaluated at tying points
for each FULL Gauss point xi_g:
    Bbar = sum_k Nas(xi_g)[k] * Bshear_tying[k]     # assumed (re-interpolated) shear operator
    Ks  += w_g * Bbar^T G Bbar * J                  # FULL integration (full rank -> no hourglass)
```

## 3. What is implemented here (stage 1)

`opensg_jax/fe_jax/msg_rm_timo.py::assemble_all(..., shear="mitc")` — **selective** treatment:
- `gamma13 = omega2` (BGq row 0): **FULL** integration (it is algebraic, does not lock; reduced-int
  would leave the omega2 antisymmetric mode under-penalised).
- `gamma23 = n.dw/ds - omega1` (BGq row 1): **assumed-strain** — sampled at the tying point(s),
  re-interpolated with `Nas`, then full-integrated.

`shear` options: `'mitc'` (default), `'reduced'` (legacy), `'full'`. Threaded through
`strip_RM.rm_timoshenko_6x6(..., shear=...)`.

### IMPORTANT caveat (p=1): MITC == reduced on gamma23
For the **2-node linear** element the assumed-CONSTANT `gamma23` (its center value) integrated fully
is *algebraically identical* to 1-point reduced integration (Prathap & Bhashyam 1982; confirmed
numerically: `tw_regression_guardrail.py` shows mitc-vs-reduced drift <= 0.01% on all TW cases, and the
composite GA2 is unchanged: mitc -20.80% vs reduced -20.79%). So at p=1, **stage 1 is a safe, correct
*refactor* (provably anti-locking, passes all guardrails) but does NOT change the answer** — it only
diverges from reduced at **p=2**, where reduced-int is NOT a faithful MITC.

### Why integration is NOT the composite cure
Measured (mh104 composite thin, GA2 %err vs solid): reduced -20.8%, mitc -20.8%, **full -11.2%**,
solid target ~ -0.5% (KL membrane-only gives -0.5%). `full` "helps" only by adding shear stiffness
(the locking direction) and is still far from the solid; and `full == reduced` on every thin TW case
(no locking observed) -- confirmed to 0.00% across the ENTIRE tube_thesis_314 R/h sweep
(t = 0.0715 .. 0.00715, `scripts/rm_research/debug_sweep_lock.py`), so the reduced scheme never actually prevented
locking in any validated case; it only under-integrated the soft core. => The soft-core GA2 error is **not** an integration artifact. It is the wall
transverse-shear **G leaking into the membrane-carried section shear** (the single-RM-strip cannot
represent the sandwich zig-zag). The real fix is **stage 2**: confine the wall G to its O(zeta^2)
correction so a soft core cannot soften the leading membrane GA2 (see opensg-msg-expert read in
`mh104_center_ref/MEMORY.md`). Blind G-floor is rejected (breaks 3D dehomogenization).

## 4. TW regression guardrail (must pass for EVERY change)

`scripts/rm_research/tw_regression_guardrail.py` — single-cell AND multi-cell TW benchmarks vs the FEniCS 2D-solid:
SINGLE box 1-cell, SINGLE [-45] tube, MULTI box 2-cell (thin+thick), MULTI curved tube 2-cell
(thin+thick). PASS iff MITC stays within 1.5% of the validated `reduced` result on every diagonal
AND RM is still <= KL on the shear terms (GA2,GA3). Current status: **ALL PASS**.

## 5. Cautions for soft-core / sandwich walls

MITC cures the **numerical** pathology (locking + reduced-int hourglass) *within* single-director FSDT
kinematics. It does **not** repair the **physical** inadequacy of one director for a soft-core sandwich
(true through-thickness shear is zig-zag, concentrated in the core). Remedies beyond MITC: a proper
through-thickness shear-energy homogenization of the wall G (already done via the shear-flow method in
`opensg_jax/fe_jax/msg_transverse_shear.py`), confining G to O(zeta^2) (stage 2), or — rigorously —
local 2D-solid / zig-zag (Refined Zigzag Theory) for those walls. Do NOT inherit kappa=5/6 for a
sandwich; do NOT blindly floor G.

## 6. Resources extracted (verified citations)

**MITC / assumed natural strain (origin & families)**
- Dvorkin, E.N. & Bathe, K.-J. (1984). "A continuum mechanics based four-node shell element for general
  non-linear analysis." *Engineering Computations* 1(1), 77-88. DOI 10.1108/eb023562. *(origin of MITC)*
- Bathe, K.-J. & Dvorkin, E.N. (1985). "A four-node plate bending element based on Mindlin/Reissner
  plate theory and a mixed interpolation." *IJNME* 21(2), 367-383. DOI 10.1002/nme.1620210213. *(MITC4)*
- Bathe, K.-J. & Dvorkin, E.N. (1986). "A formulation of general shell elements - the use of mixed
  interpolation of tensorial components." *IJNME* 22(3), 697-722. DOI 10.1002/nme.1620220312. *(names MITC)*
- Bucalem, M.L. & Bathe, K.-J. (1993). "Higher-order MITC general shell elements." *IJNME* 36(21),
  3729-3754. DOI 10.1002/nme.1620362109. *(MITC family, higher order)*
- Lee, P.-S. & Bathe, K.-J. (2010). "The quadratic MITC plate and MITC shell elements in plate bending."
  *Adv. Eng. Software* 41(5), 712-728. DOI 10.1016/j.advengsoft.2009.12.011. *(explicit re-interpolation)*
- Simo, J.C. & Hughes, T.J.R. (1986). "On the variational foundations of assumed strain methods."
  *J. Appl. Mech.* 53(1), 51-54. DOI 10.1115/1.3171737. *(Hu-Washizu basis)*

**Tying points / field consistency / reduced-integration equivalence**
- Barlow, J. (1976). "Optimal stress locations in finite element models." *IJNME* 10(2), 243-251.
  DOI 10.1002/nme.1620100202. *(xi=0 linear; xi=+-1/sqrt(3) quadratic)*
- Prathap, G. (1993). *The Finite Element Method in Structural Mechanics* (field-consistent elements).
  Kluwer, ISBN 978-0-7923-2492-8. DOI 10.1007/978-94-017-3319-9.
- Prathap, G. & Bhashyam, G.R. (1982). "Reduced integration and the shear-flexible beam element."
  *IJNME* 18(2), 195-210. DOI 10.1002/nme.1620180205. *(true vs spurious constraints; p=1 reduced==consistent)*
- Prathap, G. (1986). "Field-consistent strain interpolations for the quadratic shear flexible beam
  element." *IJNME* 23(11), 1973-1984. DOI 10.1002/nme.1620231102. *(p=2: reduced != MITC)*
- Hughes, T.J.R. (1987). *The Finite Element Method.* Prentice-Hall. ISBN 978-0-13-317025-2. *(B-bar, SRI)*
- Zienkiewicz, Taylor & Too (1971). "Reduced integration technique..." *IJNME* 3(2), 275-290.
  DOI 10.1002/nme.1620030211.

**1-D / Timoshenko-beam MITC**
- Carrera, E., de Miguel, A.G. & Pagani, A. (2017). "Extension of MITC to higher-order beam models and
  shear locking analysis..." *IJNME* 112(13), 1889-1908. DOI 10.1002/nme.5588. *(directly applicable 1-D MITC beam)*
- Reddy, J.N. (1997). "On locking-free shear deformable beam finite elements." *CMAME* 149(1-4),
  113-132. DOI 10.1016/S0045-7825(97)00075-3. *(interdependent interpolation element)*

**Soft-core / sandwich physics (why MITC is necessary but not sufficient)**
- Pagano, N.J. (1970). "Exact solutions for rectangular bidirectional composites and sandwich plates."
  *J. Compos. Mater.* 4(1), 20-34. DOI 10.1177/002199837000400102. *(3D benchmark; FSDT misses soft core)*
- Altenbach, Eremeyev & Naumenko (2015). "On the use of FSDT for three-layer plates with thin soft core."
  *ZAMM* 95(10), 1004-1011. DOI 10.1002/zamm.201500069.
- Carrera, E. (2003). "Historical review of zig-zag theories..." *Appl. Mech. Rev.* 56(3), 287-308.
  DOI 10.1115/1.1557614.
- Tessler, Di Sciuva & Gherlone (2009). "A refined zigzag beam theory..." *J. Compos. Mater.* 43(9),
  1051-1081. DOI 10.1177/0021998308097730. *(RZT remedy: FSDT cost, no shear-correction factor)*

**MSG/VABS theory anchor (the section-shear ordering)** — see `OpenSG_KnowledgeBase`:
bagla2025asc Eq.17 (energy `[[D,Y],[Y^T,G]]`, G=O(zeta^2)); yu2002timoshenko Eqs 32,37,49-54 (closed-form
`G_T=(Q^T A^{-1}(C-B^T A^{-1}B)A^{-1}Q)^{-1}`); deoyu2020thinwalled Eqs 2,9-10 (zeroth order = in-plane ABD only).

## 6b. Stage-2 (soft-core G-coupling) — attempts & commercial-code cross-check

**Attempt A (FAILED): remove wall G from the zeroth-order EB warping** (opensg-msg-expert recipe:
V0/Deff from a membrane-only Dhh; V1 keeps full Dhh). RESULT: collapses every case (box1 GA2 -97%,
composite -61%). REASON: this RM element has an INDEPENDENT director DOF `omega2` whose ONLY
value-stiffness is the G-energy (`2g13 = omega2`); dropping G from V0 unposes the director -> garbage
warping. The expert's recipe assumed a VABS-style *displacement* warping; this element is not that.
=> **G must stay in V0** (user-confirmed: "V0 is correct with G, keep it in the EB formulation").
Infra kept (assemble_all returns Dhh_mem; msg_solver.assemble_kkt) but V0 uses the full Dhh.

**Attempt B (rejected): Kirchhoff-penalty G in V0** (large G in V0, real G in V1) — not pursued.

**Commercial-code cross-check (Abaqus/Ansys), how they avoid the leak:**
- Abaqus Theory Guide Sec 3.6.8: transverse-shear stiffness K (2x2) from an ENERGY/EQUILIBRIUM method
  (parabolic per-layer shear matching the bending-stress gradient; 5/6 is just the isotropic special
  case). KEY: "the transverse shear flexibility within a layer is NOT COUPLED to the in-plane
  flexibility." So a soft core lowers K but leaves membrane A66 untouched.
- Ansys SHELL181/281: same energy-equivalence; "transverse shear effects are NOT included in the
  normal material constitutive modeling" (separate block). Caveat: accuracy degrades when adjacent-layer
  modulus ratio is high (= the soft-core regime).
- BOTH explicitly steer soft-core sandwiches to SOLID/continuum-shell: Abaqus "sandwich composites have
  very low transverse shear stiffness and should ... use continuum elements"; Ansys SOLSH190 stacked
  layers. => the 2D-SOLID is the correct oracle for the soft-core spar caps; the FSDT single-director
  shell is outside its validity there.
- IMPLICATION: the chordwise section shear is physically MEMBRANE-routed (A66 / shear flow,
  core-independent). Our ~20% GA2 over-softening is a HOMOGENIZATION-COUPLING artifact -- the warping
  routes the soft wall-Gs into the membrane-carried section shear. The principled target: section-shear
  flexibility = (membrane shear-flow compliance) + (wall transverse-shear compliance) in SERIES, with
  wall-Gs a SMALL add-on, NOT a parallel path that bottlenecks GA2. Implementing that requires auditing
  where the omega2 / (n.dw/ds-omega1) director carries load that belongs to membrane shear flow -- open.

## 7. Change log
- 2026-06-23: stage-1 selective assumed-strain (`shear="mitc"`) added; default. Guardrail created and
  PASSES. Finding recorded: p=1 MITC == reduced (safe refactor, no-op for composite); composite soft-core
  GA2 needs stage 2 (G-coupling), not an integration change. Author: investigation in session 80bf9246.
- 2026-06-23: stage-2 attempt A (remove G from EB) FAILED (unposes director omega2) -> reverted, G stays
  in V0 (user-confirmed). Commercial-code (Abaqus 3.6.8 / Ansys SHELL181) cross-check: membrane A and
  transverse-shear K are SEPARATE; soft core lowers K not A66; both recommend SOLID for soft-core
  sandwiches. Conclusion: GA2 over-softening is a homogenization-coupling artifact + a documented FSDT
  single-director limitation; 2D-solid is the oracle. Principled fix (series-compliance separation) open.
