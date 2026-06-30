# Analytical Validation — source notes

Transcribed from `Latex Conference publications/opensg/Opensg_MSG.pdf`, §"Analytical
Validation: Prismatic Beam" (report eqs 3.29–3.51, Table 3.1). Used to populate the
KL paper's §3 (Analytical Validation) and the isotropic-tube results column.

## Prismatic simplification
For a 2D SG (prismatic beam), the beam–shell rotation reduces to
```
C^ab = [[1, 0,   0  ],
        [0, ẋ2,  ẋ3 ],
        [0, -ẋ3, ẋ2 ]]              (report 3.1)
```
giving the KL prismatic shell strains (report 3.29) — identical to the general KL
strains with one circumferential variable ζ2.

## Euler–Bernoulli (zeroth order)
Zeroth-order energy (report 3.34):
```
2Π0 = Eh/(1-ν²) ⟨ ε11² + ε22² + 2ν ε11 ε22 + (1-ν)/2 (2ε12)²
        + h²/12 [ κ11² + κ22² + 2ν κ11 κ22 + (1-ν)/2 (κ12+κ21)² ] ⟩
```
Solved fluctuating functions for the circle (report 3.35):
```
ŵ1 = -d(ζ2 - πd) κ1
ŵ2 = ν(-γ11 x2 + (κ3/2)(x2² - x3²) - κ2 x2 x3)
ŵ3 = ν(-γ11 x3 + κ3 x2 x3 + (κ2/2)(x2² - x3²))
```
Asymptotically correct zeroth-order strains (report 3.36):
```
ε11 = γ11 + x3κ2 - x2κ3 ;  ε22 = -ν ε11 ;  2ε12 = 0
κ11 = ẋρκρ ;  κ22 = -ν ẋρκρ ;  κ12+κ21 = -2κ1
```
Reduced zeroth-order energy (report 3.38):
```
2Π0 = Eh ( γ11²⟨1⟩ + h²/(6(1+ν)) ⟨1⟩ κ1²
          + κ2² ⟨x3² + h²/12 ẋ2²⟩ + κ3² ⟨x2² + h²/12 ẋ3²⟩ )
```

## Timoshenko-like (second order)
2U = εt^T X εt + 2 εt^T F γ + γ^T G γ   (report 3.47), with (3.48, 3.2):
```
G = (Q^T A^-1 C A^-1 Q)^-1 ;  F = B^T A^-1 Q G ;  X = A + F G^-1 F^T
```
For the prismatic circular section: **X = A, F = 0** (no shear–extension/bending
coupling), and the transverse-shear block is diagonal (report 3.50–3.51):
```
G11 = Eh ⟨x2² + h²/12 ẋ3²⟩² / ( 2 d² (1+ν) ⟨x3²⟩ )
G22 = Eh ⟨x3² + h²/12 ẋ2²⟩² / ( 2 d² (1+ν) ⟨x2²⟩ )
```
EB 4×4 (report 3.46):
```
A = Eh diag( ⟨1⟩,  h²/(6(1+ν)) ⟨1⟩,  ⟨x3²+h²/12 ẋ2²⟩,  ⟨x2²+h²/12 ẋ3²⟩ )
            [ EA ,        GJ ,             EI2 ,              EI3 ]   (order γ11,κ1,κ2,κ3)
```
Circle of radius d=R: ⟨1⟩=2πR, ⟨x2²⟩=⟨x3²⟩=πR³, ⟨ẋ2²⟩=⟨ẋ3²⟩=πR.

## Validation case + Table 3.1 (isotropic circular tube)
Case-1: R = 5 m, h = 0.2 m, E = 3.44 GPa, ν = 0.3, **center reference**.
MSG-Solid mesh: 1536 2D elems / 1600 nodes; MSG-TW: 64 1D line elements.

| Stiffness (×10^6) | OpenSG (FEniCSx, OpenSG-1.0) | Analytical | MSG-Solid (VABS) |
|-------------------|------------------------------|------------|------------------|
| C11               | 21606                        | 21606      | 21622            |
| C22 = C33         | 4153                         | 4157       | 4205             |
| C44               | 207650                       | 207650     | 207299           |
| C55 = C66         | 269680                       | 269680     | 269935           |

~1% in transverse shear (C22/C33) is the KL classical-shell limitation (no RM
transverse-shear DOF); negligible against the dominant terms.

## Anisotropic circular tube (Euler–Bernoulli / classical only) — report Table 3.2
Validated against Yu et al. (2005) closed-form (NOT a full Timoshenko analytical):
| Stiffness (×10^6) | MSG-TW (center ref) | Yu et al. 2005 | MSG-Solid (VABS) |
|-------------------|---------------------|----------------|------------------|
| C11               | 47.785              | 47.729         | 47.691           |
| C12               | -0.93755            | -0.93607       | -0.93541         |
| C22               | 0.14896             | 0.14903        | 0.14843          |
| C33 = C44         | 0.10710             | 0.10728        | 0.10690          |

## TODO for this paper
- JAX gradient-Kirchhoff (OpenSG-2.0) KL column for the isotropic tube — RUN to fill.
- Confirm FEniCSx-KL vs JAX-KL agree to the printed sig figs (headline comparison).
