"""timo_ring_ident.py -- closed-form identification checks for the analytical ring chain:
(1) does the beam GA depend on the wall shear block Gs?  (2) GA form GA = pi R Gh (1+h^2/6R^2)
across thicknesses; (3) anisotropic EA/GA against the lambda-condensed wall moduli.
"""
import numpy as np
from analytical_timo_ring import ring_chain, wall_iso, wall_ply45

R = 0.0715
E, nu = 70e9, 0.3
G = E / (2 * (1 + nu))

print("== (1) Gs dependence, iso h=R/2 ==")
A3, D3, G2 = wall_iso(E, nu, R / 2)
for f in (0.5, 1.0, 5.0):
    d, _ = ring_chain(A3, D3, G2 * f, R)
    print("  Gs x%.1f : GA = %.6e   GJ = %.6e" % (f, d[1], d[3]))

print("== (2) GA/(pi R G h) vs 1 + h^2/(6R^2) ==")
for hf in (0.5, 0.25, 0.1, 0.05):
    h = R * hf
    A3, D3, G2 = wall_iso(E, nu, h)
    d, _ = ring_chain(A3, D3, G2, R)
    base = np.pi * R * G * h
    print("  h/R=%.2f : ratio %.6f   model %.6f   GJ/(2 pi R^3 G h) %.6f   EA/(2 pi R E h) %.6f"
          % (hf, d[1] / base, 1 + hf ** 2 / 6, d[3] / (2 * np.pi * R ** 3 * G * h),
             d[0] / (2 * np.pi * R * E * h)))

print("== (3) [-45] condensed moduli ==")
for hf in (0.5, 0.1):
    h = R * hf
    A3, D3, G2 = wall_ply45(h)
    d, _ = ring_chain(A3, D3, G2, R)
    Ax = np.linalg.inv(np.linalg.inv(A3)[:1, :1])[0, 0]        # condensed axial modulus*h
    # condensed in-plane shear modulus*h (for the shear flow): condense e11,e22 at fixed g12
    Gx = np.linalg.inv(np.linalg.inv(A3)[2:, 2:])[0, 0]
    print("  h/R=%.2f : EA/(2 pi R Ax) = %.6f   GA/(pi R Gx (1+h^2/6R^2)) = %.6f   GJ/(2 pi R^3 Gx) = %.6f"
          % (hf, d[0] / (2 * np.pi * R * Ax), d[1] / (np.pi * R * Gx * (1 + hf ** 2 / 6)),
             d[3] / (2 * np.pi * R ** 3 * Gx / h * h)))
