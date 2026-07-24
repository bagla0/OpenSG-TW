"""diagnose.py -- the two questions the headline table raises, answered with numbers.

 Q1  Is the MSG first-order transverse-shear recovery IDENTICAL to the classical
     equilibrium (Whitney-1973) shear flow, or merely close?
 Q2  In the genuinely thick regime (S = 4..5) both recoveries still carry 20-40% error on
     sigma13 and 40-86% on sigma11.  WHERE does that error live?

Q2 is answered by fitting a straight line to the exact sigma11 INSIDE each ply and
measuring the residual.  If the exact distribution were strongly curved within a ply, a
zeroth-order (linear-in-z) recovery could never represent it and the fix would be more
through-thickness resolution.  It is not: for the sandwich only 0.26% of the exact
sigma11 is non-linear within a ply while the total error is 86%, so the plies are still
essentially linear and it is the PARTITIONING of M11 between them that is wrong -- the
zig-zag, i.e. the second-order warping.
"""
import numpy as np

from jaxcfg import jnp          # noqa: F401  (x64 first)
from exact_cyl import ExactCyl
from materials import MATDB
import models as M

CASES = {
    'crossply': ([1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3, 6),
    'sandwich': ([0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face'], 8),
}


def q1():
    print("=" * 76)
    print("Q1  max |sigma13_MSG - sigma13_CLTequil| / max|sigma13_exact|")
    print("=" * 76)
    cases = [
        ("[0/90/0]  ", [1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3, 4, 6),
        ("[0/90/0]  ", [1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3, 100, 6),
        ("[0/45/0]  ", [1 / 3] * 3, [0., 45., 0.], ['pagano'] * 3, 10, 6),
        ("[0/core/0]", [0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face'], 4, 8),
        ("[0/core/0]", [0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face'], 20, 8),
    ]
    for name, thick, ang, mats, S, npl in cases:
        r = M.run(thick, ang, mats, MATDB, S, npl_sg=npl)
        scale = np.max(np.abs(r['exact'][:, 4]))
        d13 = np.max(np.abs(r['msg']['s13'] - r['clt']['s13'])) / scale
        d23 = np.max(np.abs(r['msg']['s23'])) / scale
        print(f"  {name} S = {S:>4}:  d(sigma13) = {d13:.3e}"
              f"   |sigma23_MSG| = {d23:.3e}")
    print("  -> the first-order VAM warping REPRODUCES the classical shear flow to")
    print("     round-off.  Both are the asymptotically exact O(h/L) transverse shear.")


def q2():
    print()
    print("=" * 76)
    print("Q2  where the residual error lives")
    print("=" * 76)
    for tag, (thick, ang, mats, npl) in CASES.items():
        print(f"\n  {tag}")
        print(f"    {'S':>5} {'s11 err':>9} {'s11 nonlin-in-ply':>19} {'s13 err':>9}"
              f" {'s33 err':>9} {'s33 direct':>12}")
        for S in (4, 5, 10, 20, 50, 100):
            r = M.run(thick, ang, mats, MATDB, S, n_per_layer_out=61, npl_sg=npl)
            ex = r['exact']
            zc = r['zc']

            nl = 0.0
            for k in range(len(thick)):
                sl = slice(k * 61, (k + 1) * 61)
                zz, ss = zc[sl], ex[sl, 0]
                nl += np.sum((ss - np.polyval(np.polyfit(zz, ss, 1), zz)) ** 2)
            nl = np.sqrt(nl) / np.linalg.norm(ex[:, 0])

            print(f"    {S:>5} {100 * M.relerr(r['msg']['s11'], ex[:, 0]):>8.2f}%"
                  f" {100 * nl:>18.2f}%"
                  f" {100 * M.relerr(r['msg']['s13'], ex[:, 4]):>8.2f}%"
                  f" {100 * M.relerr(r['msg']['s33'], ex[:, 2]):>8.2f}%"
                  f" {np.max(np.abs(r['msg']['s33_direct'])):>12.2e}")
    print()
    print("  'nonlin-in-ply' = the share of the EXACT sigma11 that a linear-through-each-")
    print("  ply recovery can never represent.  It is far SMALLER than the total error,")
    print("  so the plies are still essentially linear -- what is wrong is how the moment")
    print("  is split BETWEEN them.  That is the second-order warping (Yu 2002 / 2005).")
    print("  's33 direct' = sigma33 straight out of C:Gamma at mid-span; ~0 because the")
    print("  zeroth-order warping enforces plane stress, which is why sigma33 comes from")
    print("  the equilibrium integration instead.")


def q3():
    print()
    print("=" * 76)
    print("Q3  OPEN -- angle-ply sigma23, MSG vs the classical route vs exact")
    print("=" * 76)
    print(f"  {'layup':<11} {'S':>4} {'peak exact':>12} {'peak MSG':>10}"
          f" {'MSG err':>9} {'CLTeq err':>10}")
    for th in (15., 30., 45., 60.):
        for S in (10, 20):
            r = M.run([1 / 3] * 3, [0., th, 0.], ['pagano'] * 3, MATDB, S, npl_sg=6)
            ex = r['exact']
            i = int(np.argmax(np.abs(ex[:, 3])))
            print(f"  {'[0/%g/0]' % th:<11} {S:>4} {ex[i, 3]:>12.4f}"
                  f" {r['msg']['s23'][i]:>10.4f}"
                  f" {100 * M.relerr(r['msg']['s23'], ex[:, 3]):>8.2f}%"
                  f" {100 * M.relerr(r['clt']['s23'], ex[:, 3]):>9.2f}%")
    print("  -> under-predicted ~2.4x, and NOT converging in S.  MSG and the classical")
    print("     route agree with each other, so this is a theory-order gap, not a bug in")
    print("     either implementation.  Must be settled before claiming the aniso case.")


if __name__ == '__main__':
    q1()
    q2()
    q3()
