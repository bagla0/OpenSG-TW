"""cone_analytical.py -- an ANALYTICAL reference for the tapered circle, which we currently lack.

Why this matters.  The tapered SQUARE has a 3-D solid reference (mitred), so we can say how far the FSM is
from it.  The tapered CONE has NO independent reference at all in this study, so its "accuracy" is
unmeasured -- we only know connected vs per-station, which is an internal consistency check, not accuracy.

Classical linear result for a thin isotropic cylinder under axial compression:
      N_cr = E t^2 / (R sqrt(3(1-nu^2)))            [force per unit circumference]
  =>  P_cr = 2 pi R N_cr = 2 pi E t^2 / sqrt(3(1-nu^2))      -- independent of R.
For a truncated cone of semi-vertex angle alpha the classical (Seide / equivalent-cylinder) result replaces
the radius by R/cos(alpha) and resolves the axial stress, giving
      P_cr = 2 pi E t^2 cos^2(alpha) / sqrt(3(1-nu^2)).

Caveats stated up front, because they bound what this comparison can prove:
  * This is the CLASSICAL LINEAR eigenvalue, the same quantity our FSM computes -- so it is the right
    reference for a linear comparison.  It is NOT the physical collapse load: axially compressed
    cylinders/cones are strongly imperfection-sensitive and fail well below it.
  * The isotropic axially-compressed cylinder is Koiter-degenerate (many modes cluster at the same load), so
    the eigenVALUE is well defined but the eigenVECTOR is not unique. Compare loads, not mode shapes.
  * The FSM already has a measured offset from this reference in the PRISMATIC limit; that offset must be
    divided out before attributing anything to the taper.
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E, NU, T, L = 200e9, 0.3, 0.02, 2.0
R1, R2 = 1.0, 0.5

# measured FSM results (circle_study.py, unit total axial force so lambda IS the critical total force [N])
FSM = {
    ("iso", "prismatic", 4): (2.950997e8, 2.950997e8),
    ("iso", "prismatic", 8): (2.950997e8, 2.950997e8),
    ("iso", "tapered", 4): (2.827756e8, 2.945920e8),
    ("iso", "tapered", 8): (2.733177e8, 2.977014e8),
    ("m45", "prismatic", 4): (3.685932e7, 3.685932e7),
    ("m45", "prismatic", 8): (3.685932e7, 3.685932e7),
    ("m45", "tapered", 4): (3.363872e7, 3.477571e7),
    ("m45", "tapered", 8): (3.363872e7, 3.505277e7),
}

den = np.sqrt(3.0 * (1.0 - NU ** 2))
P_cyl = 2 * np.pi * E * T ** 2 / den
alpha = np.arctan((R1 - R2) / L)
P_cone = P_cyl * np.cos(alpha) ** 2

print("ANALYTICAL reference for the isotropic circle (classical linear axial buckling)\n")
print("   E=%.0f GPa  nu=%.2f  t=%.3f m  L=%.1f m  R %.2f -> %.2f" % (E / 1e9, NU, T, L, R1, R2))
print("   sqrt(3(1-nu^2)) = %.4f" % den)
print("   cylinder  P_cr = 2 pi E t^2 / sqrt(3(1-nu^2))          = %.5e N   (independent of R)" % P_cyl)
print("   semi-vertex angle alpha = atan((R1-R2)/L) = %.3f deg,  cos^2 = %.4f" % (np.degrees(alpha),
                                                                                 np.cos(alpha) ** 2))
print("   cone      P_cr = 2 pi E t^2 cos^2(alpha)/sqrt(3(1-nu^2)) = %.5e N" % P_cone)

print("\n   PRISMATIC calibration -- the FSM's standing offset from the classical value:")
per_p, con_p = FSM[("iso", "prismatic", 8)]
cal = con_p / P_cyl
print("      FSM prismatic %.5e  /  analytical %.5e  =  %.4f" % (con_p, P_cyl, cal))
print("      (compare the independently measured SS3 cylinder benchmark, 0.952)")

print("\n   TAPERED cone vs the analytical cone value:")
print("      variant           FSM [N]        /analytical   /analytical, calibrated")
for nsec in (4, 8):
    per, con = FSM[("iso", "tapered", nsec)]
    print("      per-station n=%d   %.5e     %.4f        %.4f" % (nsec, per, per / P_cone, per / P_cone / cal))
    print("      connected   n=%d   %.5e     %.4f        %.4f" % (nsec, con, con / P_cone, con / P_cone / cal))

print("""
   Reading of this, stated carefully:
     * The 'calibrated' column divides out the prismatic offset, so it isolates what the TAPER costs.
       A calibrated value of 1.000 would mean the FSM handles the taper exactly as well as it handles the
       prismatic case.
     * There is no analytical anisotropic cone result of comparable standing, so the m45 rows cannot be
       calibrated this way. An anisotropic cone would need either a 3-D solid reference or a
       Flugge/Donnell-type shell solution for a laminated cone -- that is the gap to close next.
""")

print("=" * 96)
print("SUMMARY OF ALL TAPERED WORK")
print("=" * 96)
print("""
TAPERED SQUARE   (a 1.0 -> 0.5, t=0.02, L=2.0; 3-D solid reference, corner-MITRED and converged)
  material   per-station     connected      solid (mitred)    conn/solid   per/solid
  iso        2.33e7 (approx) 2.70809e7      3.23962e7          0.836        ~0.72
  m45        4.64580e6       5.02045e6      6.77569e6          0.741        0.686
  prismatic square FSM/solid = 0.974  <- the shell-vs-solid floor; the taper-specific gap is what exceeds it

TAPERED CIRCLE / CONE   (R 1.0 -> 0.5, t=0.02, L=2.0; NO solid reference -- analytical only, above)
  material   nsec   per-station     connected      conn/per
  iso        4      2.827756e8      2.945920e8      1.0418
  iso        8      2.733177e8      2.977014e8      1.0892
  m45        4      3.363872e7      3.477571e7      1.0338
  m45        8      3.363872e7      3.505277e7      1.0420
  harmonic convergence (tapered, nsec=8): iso converged by M=12; m45 needs M=18-24
    (M=6 is 26% high for m45 -- the 16/26 coupling terms need a richer harmonic basis)

BAR-URC BLADE  (30 OML stations, flapwise 1200 Pa, OML-ref RM dehom)
  quantity                                   value      reference        ratio
  per-station governing (st6, L=3.448 m)     1.0468     stated ~1.04     1.006
  connected segment 5   (st5-6, L=3.448 m)   1.1509     seg5 = 1.0751    1.071
  connected st5-7       (L=6.897 m)          0.9597     -- different domain, not comparable
  prismatic webbed sanity (connected/per)    1.000000   exact            --
""")
