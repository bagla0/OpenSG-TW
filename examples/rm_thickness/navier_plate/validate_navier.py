"""validate_navier.py -- verification of the double-sine (Pagano-1970) machinery.

  1  Exact solver: traction BCs, sigma_z closure, and the CYLINDRICAL LIMIT -- with
     b/a -> inf the double-sine solution must collapse onto the already-validated
     cylindrical-bending solver of ../exact_cyl.py.
  2  Literature anchors: the central deflection of the [0/90/0] plate in the
     Mendonca-Ruviaro normalisation, wbar(0) ~ 2.00 (a/H=4) and ~ 0.435 (a/H=100)
     [Pagano 1970 / standard benchmark values], and the FSDT k=5/6 values their
     figures show (~1.78 and ~0.4337).
  3  Consistency: the analytic-FSDT recovery closes sigma_z on q11 BEFORE its scaling
     step (their Theorem 1), and the MSG sigma_z closes without any scaling step.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from jaxcfg import jnp                 # noqa: E402
from materials import MATDB            # noqa: E402
from exact_cyl import ExactCyl         # noqa: E402
from exact_navier import ExactNavier   # noqa: E402
import navier_models as NM             # noqa: E402

OK, BAD = "  ok ", " FAIL"


def _r(name, val, tol, fmt="%.3e"):
    tag = OK if val <= tol else BAD
    print(f"{tag}  {name:<58} {fmt % val}  (tol {fmt % tol})")
    return val <= tol


def test_exact():
    print("\n1. exact double-sine solver")
    good = True
    for case, aspect in (('sym', 4), ('sym', 100), ('asym', 4), ('sandwich', 4)):
        H = 1.0 / aspect
        thick, ang, mats = NM.CASES[case](H)
        ex = ExactNavier(thick, ang, mats, NM.MATDB_MR, 1.0, 1.0, q11=1e4)
        good &= _r(f"{case:<8} a/H={aspect:>3}  traction-BC residual",
                   ex.bc_residual(), 1e-11)
        good &= _r(f"{case:<8} a/H={aspect:>3}  sigma_z closure |.-1|",
                   ex.sigz_closure(), 1e-8)

    # cylindrical limit: b -> inf must reproduce ../exact_cyl.py
    thick = [1 / 3] * 3; ang = [0., 90., 0.]; mats = ['pagano'] * 3
    S = 10
    cyl = ExactCyl(thick, ang, mats, MATDB, S * 1.0)
    nav = ExactNavier(thick, ang, mats, MATDB, S * 1.0, 1e5 * S, q11=1.0)
    zc1, s1, _, _ = cyl.profile(n_per_layer=31)
    zc2, s2, _, _ = nav.profile(n_per_layer=31)
    for name, i in (('sigma11', 0), ('sigma13', 4), ('sigma33', 2)):
        good &= _r(f"cylindrical limit (b/a=1e5)  {name}",
                   NM.relerr(s2[:, i], s1[:, i]), 2e-4)
    return good


def test_anchors():
    print("\n2. literature anchors, [0/90/0] central deflection wbar(z=0)")
    good = True
    targets = {4: (2.00, 1.78), 100: (0.435, 0.4337)}   # (3D approx, FSDT k=5/6)
    for aspect, (t3d, tfs) in targets.items():
        r = NM.run_case('sym', aspect)
        i0 = int(np.argmin(np.abs(r['zc'])))
        w3d = r['exact']['w'][i0] * r['nrm']['w']
        wfs = r['fsdt']['d'][2] * r['nrm']['w']
        wmsg = r['msg']['d'][2] * r['nrm']['w']
        print(f"       a/H={aspect:>3}:  3D {w3d:.4f} (lit ~{t3d})   "
              f"FSDT {wfs:.4f} (fig ~{tfs})   OpenSG-RM {wmsg:.4f}")
        good &= _r(f"a/H={aspect:>3}  |wbar_3D - lit|/lit", abs(w3d - t3d) / t3d, 0.02)
        good &= _r(f"a/H={aspect:>3}  |wbar_FSDT - fig|/fig", abs(wfs - tfs) / tfs, 0.02)
    return good


def test_consistency():
    print("\n3. recovery consistency")
    good = True
    for case, aspect in (('sym', 4), ('asym', 4), ('sandwich', 4), ('sym', 100)):
        r = NM.run_case(case, aspect)
        good &= _r(f"{case:<8} a/H={aspect:>3}  FSDT sigma_z pre-scaling |./q11-1|",
                   abs(r['fsdt']['sz_top_raw'] - 1.0), 5e-3)
        good &= _r(f"{case:<8} a/H={aspect:>3}  MSG  sigma_z closure (no scaling)",
                   r['msg']['sz_top_closure'], 5e-3)
        # both transverse-shear recoveries must vanish on both faces
        for m in ('fsdt', 'msg'):
            tmax = np.max(np.abs(r[m]['txz']))
            good &= _r(f"{case:<8} a/H={aspect:>3}  {m:<4} tau_xz faces",
                       max(abs(r[m]['txz'][0]), abs(r[m]['txz'][-1])) / tmax, 2e-2)
    return good


if __name__ == '__main__':
    ok = True
    ok &= test_exact()
    ok &= test_anchors()
    ok &= test_consistency()
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)
