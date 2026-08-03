# Analytical derivation of the FSM for a prismatic section, and what it implies for taper

Everything below is closed form. The point is not the prismatic result itself (that is classical) but the
**three places where the derivation uses prismaticity**, because each one tells us exactly what a
non-prismatic segment needs.

---

## 1. Strip kinematics

A strip is a flat laminated plate: longitudinal coordinate `x ∈ [0,L]`, transverse `y ∈ [0,b]`, thickness
`h`, laminate `[[A,B],[B,D]]` in Voigt order `[11,22,12]`. Displacements are the two membrane components
`u` (longitudinal), `v` (transverse) and the out-of-plane `w`.

## 2. The separable ansatz — and why the phases are what they are

    u(x,y) = U(y) cos(k x)
    v(x,y) = V(y) sin(k x)          k = mπ/L
    w(x,y) = W(y) sin(k x)

This phase assignment is **not** a modelling convenience. It makes the simply-supported (SS3) end condition
hold **identically**, not approximately, at `x = 0, L`:

| condition | check |
|---|---|
| `w = 0` | `W sin(kx)` vanishes at `x=0,L` ✓ |
| `M_x = 0` | `κ_x = -w,xx = k²W sin(kx)` vanishes at the ends, so `M_x = 0` **provided `D16 = 0`** ✓ |
| `v = 0` | `V sin(kx)` vanishes ✓ |
| `u` free | `u = U cos(kx)` is unrestrained; `ε_x = -kU sin(kx)` vanishes at the ends ✓ |

Note the parenthetical in row 2 — it is the first appearance of the anisotropic problem, and it is already
visible here, before any algebra.

## 3. Strains, and the trig-type partition

    ε_x   = u,x      = -k U sin(kx)
    ε_y   = v,y      =    V' sin(kx)
    γ_xy  = u,y+v,x  = (U' + kV) cos(kx)
    κ_x   = -w,xx    =  k² W sin(kx)
    κ_y   = -w,yy    =   -W'' sin(kx)
    κ_xy  = -2w,xy   = -2k W' cos(kx)

The six strains split cleanly into two families:

    sin-type:  s = {ε_x, ε_y, κ_x, κ_y}          (code: ss = [0,1,3,4])
    cos-type:  c = {γ_xy, κ_xy}                  (code: cc = [2,5])

**This partition is the whole structure of the method.**

## 4. Strain energy and the three longitudinal integrals

    U = ½ ∫₀^L ∫₀^b {s;c}ᵀ [[E_ss, E_sc],[E_scᵀ, E_cc]] {s;c} dy dx

where `E_ss`, `E_cc`, `E_sc` are the corresponding partitions of `[[A,B],[B,D]]`. `E_sc` collects exactly
the **16/26 terms**: `A16, A26, B16, B26, D16, D26`.

The `x`-integration needs only three integrals:

    ∫₀^L sin(k_m x) sin(k_m' x) dx = (L/2) δ_mm'                                    (I)
    ∫₀^L cos(k_m x) cos(k_m' x) dx = (L/2) δ_mm'                                    (II)
    ∫₀^L sin(k_m x) cos(k_m' x) dx = c_mm' = (2L/π)·m/(m²-m'²)   for m+m' odd, else 0   (III)

Derivation of (III): `sin A cos B = ½[sin(A+B) + sin(A−B)]`, and for `m+m'` odd both `m±m'` are odd so
`cos((m±m')π) = −1`, giving `(L/π)[1/(m+m') + 1/(m−m')] = (2L/π)·m/(m²−m'²)`. This is `cmm` in the code.

## 5. Assembly, and the decoupling theorem

    K^(m,m') = (L/2) δ_mm' [B_sᵀ E_ss B_s + B_cᵀ E_cc B_c]  +  c_mm' [B_sᵀ E_sc B_c] + c_m'm [B_cᵀ E_scᵀ B_s]

**Theorem (prismatic orthotropic decoupling).** If `E_sc = 0` — i.e. the laminate is orthotropic in the
strip axes — then `K` is block-diagonal in `m` by (I) and (II), and each harmonic is an independent
eigenproblem.

This is the entire licence for classical CUFSM: solve one half-wavelength at a time and sweep. It is a
*theorem about orthotropic prismatic members*, not a general property of the method.

## 6. Geometric stiffness

Second-order work of the pre-buckling membrane state:

    W_G = ½ ∫∫ [ N_x w,x² + N_y w,y² + 2 N_xy w,x w,y ] dy dx

with `w,x = kW cos(kx)` (**cos-type**) and `w,y = W' sin(kx)` (**sin-type**). So by (I)–(III):

    N_x  term → diagonal in m
    N_y  term → diagonal in m
    N_xy term → couples m+m' odd, carrying the SAME c_mm' as E_sc

That symmetry is why keeping `E_sc` while dropping `N_xy` is inconsistent — the fix now implemented. (It
turns out numerically negligible, because `N_xy = 0` for a traction-controlled tube and only ~2% of `N_x`
on the blade, but the formulation is now complete.)

## 7. Closed-form check

For a single SS orthotropic plate strip `a × b`, mode `w = W sin(mπx/a) sin(nπy/b)`:

    N_x,cr = π² [ D11 (m/a)⁴ + 2(D12 + 2D66)(m/a)²(n/b)² + D22 (n/b)⁴ ] / (m/a)²

and the similarity `λ_m(L) = λ̂(L/m)` gives the signature curve `λ_cr(L) = min_m λ̂(L/m)`.

---

# 8. Exactly where prismaticity was used — and what taper requires

Three uses, each with a different consequence.

### (a) Orthogonality (I) and (II) — **broken by taper**

With `b = b(x)`, `ABD = ABD(x)`, `N = N(x)`, write any varying property as `f(x)` and define its cosine
moments `F_p = ∫₀^L f(x) cos(pπx/L) dx`. Then, using `sin A sin B = ½[cos(A−B) − cos(A+B)]`,

    ∫₀^L f(x) sin(k_m x) sin(k_m' x) dx = ½ [ F_|m-m'| − F_(m+m') ]

Check: `f = f₀` constant gives `F₀ = f₀L`, `F_p = 0` for `p ≥ 1`, recovering `(L/2) f₀ δ_mm'`. ✓

**For a linear taper** `f(x) = f₀ + f₁ x/L`:

    F_p = −2 f₁ L / (pπ)²   for p odd,     F_p = 0 for p even (p ≥ 1)

Two consequences fall straight out:

1. **The selection rule is `m+m'` odd** — because `|m−m'|` and `m+m'` always share parity, only odd `p`
   survives. This is the same rule the anisotropic `E_sc` obeys, and it is why the two mechanisms appear
   together in the code.
2. **The coupling decays as `1/(m−m')²`.** Relative strengths: `|m−m'| = 1 → 1`, `3 → 1/9`, `5 → 1/25`.

Result (2) is the useful one: **the taper coupling matrix is effectively banded.** Truncating to
`|m−m'| ≤ 5` retains ~97% of it. Instead of a dense `M×M` harmonic block structure we need only a banded
one — which is precisely what makes large `M` affordable, and is a better lever than the sparse-storage
saving discussed earlier (that was ~10×; this is asymptotic).

For a general (non-linear) taper the same formula holds with the actual `F_p`; the decay rate is set by the
smoothness of the property variation. A blade whose properties vary smoothly along the span will have
rapidly decaying `F_p` and therefore a narrow band.

### (b) Fixed strip geometry — **first-order in the taper angle**

The strip frame `(cy, cz)` and width `b` were treated as constants. In the connected solver they are
recomputed at each Gauss point, so the leading effect is captured. What is *not* captured is that on a
tapered wall the meridian is inclined to `x` by `α`, so the assumed `x`-aligned strip axis is off by `α`
(≈14° on our cone, `cos α = 0.970`). This is the "convective" correction — genuinely first order, but only
~3%, and provably zero on the flat walls of a tapered box.

### (c) **The fixed phase assignment — this is the anisotropic gap, and the derivation makes it rigorous**

The most general separable field at wavenumber `k` is

    u = U_c cos(kx) + U_s sin(kx)
    v = V_c cos(kx) + V_s sin(kx)
    w = W_c cos(kx) + W_s sin(kx)

— six transverse functions. Step 2 kept only `{U_c, V_s, W_s}`: **half the basis**.

For an orthotropic prismatic member that truncation is **exact**, because `E_sc = 0` makes the energy
block-diagonal in trig type, so the discarded family cannot lower the eigenvalue. It is a theorem, not an
assumption.

When `E_sc ≠ 0` the two families are coupled and the true critical mode needs both. And the discarded
functions are **not recoverable from the retained ones**: `u`'s basis is `{cos(k_m x)}` only, so a mode with
`sin` content in `u` simply cannot be formed at any `M`. The restriction is a genuine subspace restriction,
not a convergence question — which is exactly why the measured error does **not** shrink as `M` grows
(m45 is converged by `M = 18–24` and still 37% high).

By Rayleigh–Ritz, restricting the trial space can only **raise** the computed eigenvalue. **The theory
therefore predicts the sign of the error: over-prediction for any laminate with `A16/D16 ≠ 0`.** That is
what we measure: 0° → 1.005, 90° → 1.087, all off-axis → 1.16–1.58.

The fix follows directly: **carry all six functions**, i.e. give every DOF both a `sin` and a `cos`
longitudinal component (equivalently, a complex amplitude with an arbitrary phase that may vary around the
contour). This doubles the DOF count and is the standard complex-FSM treatment.

Caveat kept honest: the derivation predicts the **sign and the mechanism**, and explains why refinement in
`M` cannot help. It does **not** predict the observed non-monotonic angle dependence (error peaks near 30°
at 1.577 and falls to 1.157 at 60° while `|A16/A11|` rises). That remains unexplained, and the doubled-basis
implementation is the test: full collapse toward 1.0 proves the mechanism; partial closure means something
else is also active.

---

## Summary — what to do for tapered / non-prismatic segments

| use of prismaticity | consequence | action |
|---|---|---|
| (a) orthogonality (I),(II) | harmonics couple; strength `∝ 1/(m−m')²`, `m+m'` odd | keep the coupling but **band it** — `|m−m'| ≤ 5` retains ~97%, making large `M` cheap |
| (b) fixed strip frame | meridian inclined by `α`; ~3% at 14° | convective terms; low priority, and identically zero on flat walls |
| (c) fixed phase | **subspace restriction**, not convergence; guarantees over-prediction when `A16/D16 ≠ 0` | **doubled/complex basis** — the only fix, and the top priority |

The ordering matters: (c) is worth 16–58% on off-axis laminates, (a) is a pure efficiency win once the
coupling is understood, and (b) is a few percent.
