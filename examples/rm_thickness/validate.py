"""validate.py -- verification suite.  Run this before trusting any result in the paper.

  1  MSG-SG (JAX) vs the production OpenSG ``compute_ABD_matrix`` -- the ABD must agree
     to round-off, which pins the JAX port to the shipped pipeline.
  2  Isotropic nu = 0 -> G_msg = (5/6) G h EXACTLY, with U* -> 0.  This is the closed-form
     Reissner value and the sharpest single check on the RM projection.
  3  Exact elasticity: traction BCs, stress resultants, the isotropic parabola, and
     O(S^-2) convergence of sigma11 to CLT.
  4  sigma33 top-face closure: the equilibrium-integrated sigma33 must land on q0.
"""
import os
import sys

import numpy as np

from jaxcfg import jnp
import sg_plate as SG
import models as M
from materials import MATDB
from exact_cyl import ExactCyl

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OK = "  ok "
BAD = " FAIL"


def _r(name, val, tol, fmt="%.3e"):
    tag = OK if val <= tol else BAD
    print(f"{tag}  {name:<56} {fmt % val}  (tol {fmt % tol})")
    return val <= tol


def test_abd_vs_opensg():
    print("\n1. ABD vs production opensg_jax.fe_jax.msg_materials.compute_ABD_matrix")
    from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix
    good = True
    cases = [([1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3),
             ([0.25] * 4, [0., 45., -45., 90.], ['as4'] * 4),
             ([0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face'])]
    for thick, ang, mats in cases:
        h = sum(thick)
        sg = SG.build(thick, ang, mats, MATDB, n_per_layer=4, elem_order=3, z_ref=h / 2)
        # compute_ABD_matrix defaults to the BOTTOM face; ask for the same mid-surface
        # reference the SG here uses, otherwise the B block differs by (h/2)A.
        A_ref = compute_ABD_matrix(thick, ang, mats, MATDB, n_per_layer=4,
                                   elem_order=3, z_ref=h / 2)[0]
        A_ref = np.asarray(A_ref)
        rel = float(np.max(np.abs(np.asarray(sg['A6']) - A_ref))
                    / np.max(np.abs(A_ref)))
        good &= _r(f"[{'/'.join(str(a) for a in ang)}]  max|dA|/max|A|", rel, 1e-10)
    return good


def test_iso_shear():
    print("\n2. isotropic G_msg  (nu = 0 -> exactly 5/6 G h; nu = 0.3 -> Hutchinson-ish)")
    h = 0.01
    good = True
    sg = SG.build([h], [0.], ['iso0'], MATDB, n_per_layer=4, elem_order=3, z_ref=h / 2)
    k = float(sg['G_msg'][0, 0]) / (35e9 * h)
    good &= _r("nu = 0   |k - 5/6|", abs(k - 5.0 / 6.0), 1e-9)
    good &= _r("nu = 0   U*_rel", float(sg['Ustar_rel']), 1e-12)
    sg2 = SG.build([h], [0.], ['iso'], MATDB, n_per_layer=8, elem_order=3, z_ref=h / 2)
    k2 = float(sg2['G_msg'][0, 0]) / ((70e9 / 2.6) * h)
    print(f"       nu = 0.3 : k = {k2:.6f}   (Hutchinson 5/(6-nu) = {5/(6-0.3):.6f})")
    return good


def test_exact():
    print("\n3. exact elasticity")
    thick = [1 / 3] * 3
    ang = [0., 90., 0.]
    mats = ['pagano'] * 3
    good = True
    for S in (4, 10, 100):
        ex = ExactCyl(thick, ang, mats, MATDB, S * 1.0)
        N11, M11, Q1 = ex.resultants()
        good &= _r(f"S = {S:>3}  traction-BC residual", ex.bc_residual(), 1e-11)
        good &= _r(f"S = {S:>3}  |M11 / (q0/p^2) - 1|",
                   abs(M11 / (1.0 / ex.p ** 2) - 1.0), 1e-8)
        good &= _r(f"S = {S:>3}  |Q1 / (q0/p) - 1|", abs(Q1 / (1.0 / ex.p) - 1.0), 1e-8)
        good &= _r(f"S = {S:>3}  |N11| / (q0 h)", abs(N11), 1e-9)

    ex = ExactCyl([1.0], [0.], ['iso'], MATDB, 10.0)
    zc, sig, _, _ = ex.profile(n_per_layer=41)
    par = 1.5 * (1.0 / ex.p) * (1 - 4 * zc ** 2)
    good &= _r("isotropic S = 10, sigma13 vs 1.5 Q/h parabola",
               float(np.max(np.abs(sig[:, 4] - par)) / np.max(np.abs(par))), 2e-3)

    print("   sigma11 -> CLT as S grows (expect a clean 1/S^2 slope):")
    prev = None
    for S in (25, 50, 100, 200):
        ex = ExactCyl(thick, ang, mats, MATDB, S * 1.0)
        sg = SG.build(thick, ang, mats, MATDB, n_per_layer=6, elem_order=3)
        E6 = M.plate_strains(sg['A6'], ex.p)
        zc, sig, _, _ = ex.profile(n_per_layer=41)
        lay = np.repeat(np.arange(3), 41)
        clt = np.asarray(M._clt_inplane(sg, E6, zc, lay)[:, 0])
        rel = float(np.linalg.norm(sig[:, 0] - clt) / np.linalg.norm(clt))
        rate = "" if prev is None else f"   ratio {prev / rel:6.2f} (expect 4.00)"
        print(f"       S = {S:>4} : {rel:.3e}{rate}")
        prev = rel
    return good


def test_sigma33_closure():
    print("\n4. sigma33 top-face closure (equilibrium integration must land on q0)")
    good = True
    for name, thick, ang, mats in [("[0/90/0] ", [1 / 3] * 3, [0., 90., 0.],
                                    ['pagano'] * 3),
                                   ("sandwich ", [0.1, 0.8, 0.1], [0.] * 3,
                                    ['face', 'core', 'face'])]:
        for S in (5, 20):
            r = M.run(thick, ang, mats, MATDB, S, npl_sg=8)
            good &= _r(f"{name} S = {S:>3}  |sigma33(top)/q0 - 1|",
                       abs(float(r['msg']['s33'][-1]) - 1.0), 5e-3)
    return good


if __name__ == '__main__':
    ok = True
    ok &= test_abd_vs_opensg()
    ok &= test_iso_shear()
    ok &= test_exact()
    ok &= test_sigma33_closure()
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)
