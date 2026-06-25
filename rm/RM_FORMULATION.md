# MSG thin-walled beam — Reissner–Mindlin (RM) shell model

Source: `Opensg_MSG.pdf` §3.3 "Reissner–Mindlin Shell: Beam Model" (pp. 90–105).
This branch (`msg-reissner-mindlin`) builds the RM model **separately** from the
Kirchhoff Hermite code — no Kirchhoff files are modified.

## Why RM (vs the Kirchhoff model on `jax-msg-shell`)

- Kirchhoff shell → 4th-order PDE, needs penalty/Hermite-C1 for slope continuity;
  cannot fully capture torsion of a 3D SG (shell segment) and has ~1% error in
  the **transverse-shear** beam terms (Table 3.1).
- RM adds the two **transverse-shear strains** `2γ13, 2γ23` and **two rotation
  fluctuating DOFs** `ω1, ω2` per node. Curvature strains then have only **first**
  derivatives of the fluctuating functions → **2nd-order PDE, C0 elements, no
  penalty**. Cost: **shear locking** must be handled (selective/reduced
  integration or assumed strain).
- The RM advantage shows at the **outer reference surface** (the OML case we care
  about for C13): RM validates outer-reference thin-walled beams via eq (4.23),
  whereas at the **center reference** for a *straight prismatic* beam the
  transverse shears vanish and RM reduces to Kirchhoff (eq 4.28).

## Kinematics (eq 4.1–4.13)

- Reference-surface position `r = r̄(x1) + ε yη bη`; orthonormal triad `ai`
  (`a3 = a1×a2` normal). Initial curvature `kα = [−kα2, kα1, kα3]`.
- Deformed strain measures: `R,α = Aα + ϵαβ Aβ + 2γα3 A3` — `ϵαβ` in-plane,
  `γα3` transverse (A3 not normal in RM). Symmetry `ϵ12=ϵ21` via `A1·R,2 = A2·R,1`.
- Beam triad `bi` (b1 = beam axis), `Cab = ai·bj`. Deformed `R = R̄ + yηBη + wi Bi`,
  `w=[w1,w2,w3]` displacement fluctuation.
- Rotation fluctuation `ωi`: `CAB = Cab(Δ − ω̃)`; shell rotations
  `ρs_i = Cab(θb_i + ε ωi)`; drilling `ρs_3 = φ3` known from `ϵ12=ϵ21`, so only
  `ω1, ω2` are independent unknown rotations. Constraints `⟨⟨wi⟩⟩ = 0`.

## Prismatic shell strain field (eq 4.23)  — `˙() = ∂()/∂ζ2`, `Rn = x2 ẋ3 − x3 ẋ2`

Beam strains `ε_b = [γ11, κ1(twist), κ2, κ3]`.  Underlined `w'_s` terms are O(ε²)
(dropped at zeroth/EB order).

```
ϵ11      = γ11 + x3κ2 − x2κ3 + w1'
ϵ22      = ẋη ẇη
2ϵ12     = ẇ1 + κ1 Rn + ẋη wη'
κ11      = ẋη κη + ω2'/ẋ2 ... (+ k22, Rn terms)
κ22      = − ω̇1
κ12+κ21  = − κ1 + ω̇2/ẋ2 + (k22/2)(ẋ3/ẋ2)²(ẇ1−κ1Rn) − ... 
2γ13     = κ1[x2(ẋ2+ẋ3²/2ẋ2) + x3 ẋ3/2] − ẇ1 ẋ3/2ẋ2 + ω2/ẋ2 − w2' ẋ3/2 + w3'(ẋ2+ẋ3²/2ẋ2)
2γ23     = (ẇ3 ẋ2 − ẇ2 ẋ3) − ω1
```

## Constitutive (eq 4.24)

```
2Π = [Γs_D; Γs_G]ᵀ [[D, Y],[Yᵀ, G]] [Γs_D; Γs_G]
Γs_D = [ϵ11, ϵ22, 2ϵ12, κ11, κ22, κ12+κ21]      (6, classical plate -> ABD = D)
Γs_G = [2γ13, 2γ23]                              (2, transverse shear -> G)
```
`Y = 0` for orthotropic laminate; `G` uses shear-correction factors.

## Closed-form solved strains (after zeroth-order minimization, **center ref**)

Symmetric isotropic (eq 4.28),  `λ1=0, λ2=−ν`:
```
ϵ11 = γ11 + x3κ2 − x2κ3 ;  ϵ22 = −ν ϵ11 ;  2ϵ12 = 0
κ11 = ẋη κη ;  κ22 = −ν(ẋη κη) ;  κ12+κ21 = −2κ1 ;  2γ13 = 2γ23 = 0
```
Symmetric anisotropic (eq 4.25/4.26) adds `λ1 = −(A22A16+A12A26)/(A22A66−A26²)`,
`λ2 = −(A12A66+A16A26)/(A22A66−A26²)`, and `D12/D22, D26/D22` couplings.

> Note: `ϵ11, ϵ22, κ11, κ22` (→ EA, EI) are captured directly by these closed-form
> strains, but **closed-section torsion (GJ) and the transverse-shear beam terms
> need the full fluctuation solve** (the Bredt shear flow lives in the
> `ζ2|boun` boundary terms of eq 4.26 / the `w1,ω` fields). So eq 4.28 validates
> EA/EI in closed form; GJ + shear require the FE solve.

## Validation targets (from the document)

Table 3.1 — **isotropic** circular tube, **center ref**, R=5 m, h=0.2 m,
E=3.44 GPa, ν=0.3 (×10⁶):

| term | OpenSG | analytical | VABS |
|------|--------|-----------|------|
| C11 (EA)        | 21606  | 21606  | 21622 |
| C22=C33 (shear) | 4153   | 4157   | 4205  |
| C44 (GJ)        | 207650 | 207650 | 207299 |
| C55=C66 (bend)  | 269680 | 269680 | 269935 |

Table 3.2 — **anisotropic** circular tube, **center ref**, R=0.0715 m,
h=0.008682 m, −45° ply; E1=37, E2=E3=9, G12=G13=G23=4 GPa, ν=0.3 (×10⁶):

| term | OpenSG | Yu 2005 | VABS |
|------|--------|---------|------|
| C11 (EA)            | 47.785   | 47.729  | 47.691 |
| C12 (ext–twist)     | −0.93755 | −0.93607| −0.93541 |
| C22 (GJ)            | 0.14896  | 0.14903 | 0.14843 |
| C33=C44 (bend)      | 0.10710  | 0.10728 | 0.10690 |

## Implementation plan

1. [in progress] **Closed-form EA/EI check** (eq 4.28 iso, 4.25 aniso) — integrate
   `∫ Bᵀ D B ds` over the tube contour, compare C11/C55/C66 to the tables.
   `rm/validate_tube_classical.py`.
2. **Full RM FE solve** (`rm/msg_rm.py`): C0 line elements, 5 DOF/node
   `[w1,w2,w3,ω1,ω2]`, strain operators from eq 4.23, ABD `D` + transverse shear
   `G`, fluctuation minimization with the closed-loop torsion constraint;
   selective/reduced integration for shear locking. Validate full 6×6 vs Table 3.1.
3. **Reference study**: center vs **outer** reference (iso then anisotropic tube),
   the RM selling point — compare to the Kirchhoff results on `jax-msg-shell`.
4. (later) Thick-web st15 with RM; LaTeX/PDF report of the anisotropic-tube study.
