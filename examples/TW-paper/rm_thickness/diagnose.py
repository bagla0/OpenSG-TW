"""diagnose.py -- two questions raised by pilot.py:

 Q1  Is the MSG first-order transverse-shear recovery IDENTICAL to the classical
     equilibrium (Whitney-1973) shear flow, or merely close?
 Q2  What exactly is missing in the genuinely thick regime (S = 4..5), where BOTH
     recoveries still carry 20-40% error on sigma13 and 40-86% on sigma11?
"""
import numpy as np

from exact_cyl import ExactCyl
from materials import MATDB
import cyl_models as CM


def q1():
    print("=" * 78)
    print("Q1  max |sigma13_MSG - sigma13_CLTequil| / max|sigma13_exact|")
    print("=" * 78)
    cases = [
        ("[0/90/0]  ", [1/3]*3, [0., 90., 0.], ['pagano']*3, 4),
        ("[0/90/0]  ", [1/3]*3, [0., 90., 0.], ['pagano']*3, 100),
        ("[0/45/0]  ", [1/3]*3, [0., 45., 0.], ['pagano']*3, 10),
        ("[0/core/0]", [0.1, 0.8, 0.1], [0.]*3, ['face', 'core', 'face'], 4),
        ("[0/core/0]", [0.1, 0.8, 0.1], [0.]*3, ['face', 'core', 'face'], 20),
    ]
    for name, thick, ang, mats, S in cases:
        h = float(np.sum(thick))
        ex = ExactCyl(thick, ang, mats, MATDB, S * h)
        obj = CM.build(thick, ang, mats, MATDB, n_per_layer=6, elem_order=3)
        E6 = CM.plate_strains(obj['A6'], ex.p)
        cl = CM.clt_equil_profile(obj, E6, ex.p, n_per_layer=61)
        mg = CM.msg_profile(obj, E6, ex.p, n_per_layer=61)
        zc, sig_e, _, _ = ex.profile(n_per_layer=61)
        scale = np.max(np.abs(sig_e[:, 4]))
        d13 = np.max(np.abs(mg['s13'] - cl['s13'])) / scale
        d23 = np.max(np.abs(mg['s23'])) / scale
        print(f"  {name} S={S:>4}:  d(sigma13) = {d13:.3e}   |sigma23_MSG| = {d23:.3e}")
    print("  -> the first-order VAM warping REPRODUCES the classical shear flow.")
    print("     (Expected: both are the asymptotically-exact O(h/l) transverse shear.)")


def q2():
    print()
    print("=" * 78)
    print("Q2  where the residual error lives, [0/90/0] and sandwich")
    print("=" * 78)
    for name, thick, ang, mats in [("[0/90/0]  ", [1/3]*3, [0., 90., 0.], ['pagano']*3),
                                   ("[0/core/0]", [0.1, 0.8, 0.1], [0.]*3,
                                    ['face', 'core', 'face'])]:
        h = float(np.sum(thick))
        obj = CM.build(thick, ang, mats, MATDB, n_per_layer=6, elem_order=3)
        print(f"\n  {name}")
        print(f"    {'S':>5} {'s11 err':>10} {'s11 lin-in-z':>14} {'s13 err':>10} "
              f"{'s33 err':>10} {'s33_direct/q0':>15}")
        for S in (4, 5, 10, 20, 50, 100):
            ex = ExactCyl(thick, ang, mats, MATDB, S * h)
            E6 = CM.plate_strains(obj['A6'], ex.p)
            zc, sig_e, _, _ = ex.profile(n_per_layer=61)
            mg = CM.msg_profile(obj, E6, ex.p, n_per_layer=61)

            def rel(a, b):
                return np.linalg.norm(a - b) / np.linalg.norm(b)

            # how much of the exact sigma11 is NOT linear-in-z inside each ply?
            nl = 0.0
            npl = 61
            for k in range(len(thick)):
                sl = slice(k * npl, (k + 1) * npl)
                zz, ss = zc[sl], sig_e[sl, 0]
                fit = np.polyval(np.polyfit(zz, ss, 1), zz)
                nl += np.sum((ss - fit) ** 2)
            nl = np.sqrt(nl) / np.linalg.norm(sig_e[:, 0])

            print(f"    {S:>5} {100*rel(mg['s11'], sig_e[:,0]):>9.2f}% {100*nl:>13.2f}% "
                  f"{100*rel(mg['s13'], sig_e[:,4]):>9.2f}% "
                  f"{100*rel(mg['s33'], sig_e[:,2]):>9.2f}% "
                  f"{np.max(np.abs(mg['s33_direct'])):>15.3e}")
    print()
    print("  s11 lin-in-z = the part of the EXACT sigma11 that a linear-through-each-ply")
    print("  (i.e. zeroth-order/classical) recovery can never represent.  It tracks the")
    print("  sigma11 error -> the missing piece is the SECOND-order warping, not the shear.")
    print("  s33_direct = sigma33 straight out of C:Gamma at mid-span; ~0 because the")
    print("  zeroth-order warping enforces plane stress -> sigma33 needs equilibrium")
    print("  integration (used above) or the second-order warping.")


if __name__ == '__main__':
    q1()
    q2()
