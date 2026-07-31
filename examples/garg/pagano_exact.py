"""pagano_exact.py -- THE Pagano exact 3-D solution module for the Garg benchmarks.
The single entry point for the reference curves; nothing else in the benchmark may
use it (inputs come from statics, see statics_fsdt.py).

Problem (Pagano, J. Compos. Mater. 3 (1969) 398-411; restated as Garg Eqs. (18)-(24)):
a laminated strip 0 <= x <= a, -h/2 <= z <= h/2, in cylindrical bending (plane strain
across the width), simply supported, carrying the sinusoidal top-face pressure

    sigma_33(x, +h/2) = q0 sin(p x),  p = pi/a
    sigma_33(x, -h/2) = sigma_13(x, +-h/2) = 0
    w = 0 and sigma_11 = 0 at x = 0, a                      (Garg Eq. (21))
    continuity of (u, w, sigma_13, sigma_33) at every ply interface  (Garg Eq. (22))

Every field separates into one harmonic (Garg Eqs. (23)-(24)):
    sigma_11, sigma_22, sigma_33, w  ~  sin(p x)      ->  amplitudes = values at x = a/2
    sigma_13, u                      ~  cos(p x)      ->  amplitudes = values at x = 0

The implementation (exact_cyl.ExactCyl) solves the per-ply 3-D equations as a
state-space system in z with transfer matrices -- machine-precision equivalent to
Pagano's f_i(y) construction (boundary residual ~1e-15, resultant closure exact).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from exact_cyl import ExactCyl                    # noqa: E402  (the state-space solver)


def pagano_profiles(thick, angles_deg, mat_names, material_db, a=1.0, q0=1.0e4,
                    n_per_layer=81):
    """The exact through-thickness AMPLITUDE profiles.

    Returns (zc, sig, uvw): zc (n,) from the mid-surface; sig (n, 6) Voigt
    [11, 22, 33, 23, 13, 12]; uvw (n, 3) displacements.  Remember the families:
    sig[:, 4] (sigma_13) is the x = 0 profile, sig[:, 0]/sig[:, 2] the x = a/2 ones.
    """
    ex = ExactCyl(list(thick), list(angles_deg), list(mat_names), material_db, a, q0=q0)
    zc, sig, _, uvw = ex.profile(n_per_layer=n_per_layer)
    return zc, sig, uvw
