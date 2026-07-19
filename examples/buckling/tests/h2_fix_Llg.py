"""h2_fix_Llg.py -- ROOT CAUSE + FIX test for the folded-shell over-stiffening.

Element kinematics in shell_buckling._B_at:
    kappa_xx = beta_x,x            (Bb[0,3::5] = dN/dx)
    gamma_xz = w,x + beta_x        (Bs[0,3::5] = N)
=> the local rotational DOFs are the MINDLIN SECTION ROTATIONS beta=(beta_x,beta_y) with
   u(z) = z*beta_x , v(z) = z*beta_y.
A rotation PSEUDO-VECTOR theta produces  u = theta x (0,0,z) = (+z*theta_y, -z*theta_x, 0), so
    beta_x = +theta.e2_?   ->   beta_x = theta_y_local = theta.e2 ,   beta_y = -theta.e1 .

shell_buckling._L_lg instead writes
    L[5a+3:5a+5, 6a+3:6a+6] = T[:2]      ->   beta_x = theta.e1 ,  beta_y = theta.e2
which is the true map composed with a 90-deg rotation IN THE ELEMENT'S OWN TANGENT PLANE,
i.e. the nodal 'rotation' DOF it actually stores is  r = -e3 x theta.
Because r depends on the ELEMENT NORMAL e3, a single nodal r cannot represent the same physical
theta for two elements with different e3  ->  exact for a FLAT plate (one e3), O(dihedral) error on a
smooth curved shell, and a HARD SPURIOUS CONSTRAINT at a 90-deg FOLD (box corner / blade web junction).

This script reruns every benchmark with the CORRECTED _L_lg (monkeypatched, production code untouched).
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUCK)
import shell_buckling as sb

E, NU = 3.0e10, 0.3
_L_lg_orig = sb._L_lg


def _L_lg_fixed(T):
    """beta_x = +theta.e2 , beta_y = -theta.e1  (Mindlin section rotation from the rotation pseudo-vector)."""
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


def loft(sec_pts, sec_elems, xs):
    Ntot = len(sec_pts); NS = len(xs)
    pts = np.zeros((NS * Ntot, 3))
    for p in range(NS):
        pts[p * Ntot:(p + 1) * Ntot, 0] = xs[p]
        pts[p * Ntot:(p + 1) * Ntot, 1:] = sec_pts
    quads = []
    for p in range(NS - 1):
        b0, b1 = p * Ntot, (p + 1) * Ntot
        for (a, b) in sec_elems:
            quads.append([b0 + b, b1 + b, b1 + a, b0 + a])
    return pts, np.array(quads)


def box_case(NPS=10, NXB=40, Lb=20.0, a_=0.5, hb=0.01, Pb=1.0e4):
    side = np.linspace(-a_, a_, NPS + 1)
    loop = ([(y, -a_) for y in side[:-1]] + [(a_, z) for z in side[:-1]] +
            [(y, a_) for y in side[::-1][:-1]] + [(-a_, z) for z in side[::-1][:-1]])
    secB = np.array(loop); Ns = len(secB)
    secB_e = [(i, (i + 1) % Ns) for i in range(Ns)]
    xsB = np.linspace(0, Lb, NXB + 1)
    nodesB, quadsB = loft(secB, secB_e, xsB)
    neB = len(quadsB)
    ABDb, Gsb = sb._iso_ABD(E, NU, hb)
    ABD_eB = np.repeat(ABDb[None], neB, 0); Gs_eB = np.repeat(Gsb[None], neB, 0)
    rootB = np.unique([6 * n + k for n in range(Ns) for k in range(6)])
    fB = np.zeros(6 * len(nodesB))
    tb = NXB * Ns
    for j in range(Ns):
        fB[6 * (tb + j) + 2] = Pb / Ns
    return dict(nodes=nodesB, quads=quadsB, ABD=ABD_eB, Gs=Gs_eB, root=rootB, f=fB,
                sec=secB, sec_e=secB_e, xs=xsB, Ns=Ns, Iyy=16.0 * a_ ** 3 / 3.0, h=hb, L=Lb, P=Pb)


def box_report(tag, cs):
    u = sb.solve_static(cs["nodes"], cs["quads"], cs["ABD"], cs["Gs"], cs["f"], cs["root"])
    Nfe = sb.element_membrane_N(cs["nodes"], cs["quads"], cs["ABD"], u)
    itip = int(np.argmax(cs["nodes"][:, 0]))
    EBd = cs["P"] * cs["L"] ** 3 / (3 * E * cs["h"] * cs["Iyy"])
    print("  [%s] tip z-disp = %.6e   EB = %.6e   FE/EB = %.4f" % (tag, u[6 * itip + 2], EBd, u[6 * itip + 2] / EBd))
    Ns = cs["Ns"]; secB = cs["sec"]; sec_e = cs["sec_e"]; xsB = cs["xs"]
    print("        plane      x       M_y(FE)      M_y(applied)     ratio")
    for p in [2, 5, 10, 20, 30, 37]:
        xm = 0.5 * (xsB[p] + xsB[p + 1]); Mapp = -cs["P"] * (cs["L"] - xm); M = 0.0
        for j in range(Ns):
            aa, bb_ = sec_e[j]; pa, pb = secB[aa], secB[bb_]
            ds = np.linalg.norm(pb - pa); zm = 0.5 * (pa[1] + pb[1])
            M += -Nfe[p * Ns + j, 0] * zm * ds
        print("        %5d %7.3f  %+.5e   %+.5e   %7.4f" % (p, xm, M, Mapp, M / Mapp))
    # N11 vs beam theory at p=20
    xm = 0.5 * (xsB[20] + xsB[21]); My = -cs["P"] * (cs["L"] - xm)
    rr = []
    for j in range(Ns):
        zm = 0.5 * (secB[sec_e[j][0], 1] + secB[sec_e[j][1], 1])
        ex = -My * zm / cs["Iyy"]
        if abs(ex) > 1e-6 * abs(My / cs["Iyy"]):
            rr.append(Nfe[20 * Ns + j, 0] / ex)
    print("        N11_FE/N11_beam at p=20:  mean=%.4f  min=%.4f  max=%.4f" % (np.mean(rr), np.min(rr), np.max(rr)))
    return u, Nfe


print("=" * 100)
print("BOX BEAM: original _L_lg  vs  corrected _L_lg")
print("=" * 100)
cs = box_case()
sb._L_lg = _L_lg_orig
box_report("ORIG", cs)
print()
sb._L_lg = _L_lg_fixed
box_report("FIXED", cs)

print()
print("=" * 100)
print("MESH-REFINEMENT of the ORIGINAL bug (is it a constraint, i.e. does it NOT converge?)")
print("=" * 100)
for NPS in [5, 10, 20, 40]:
    c2 = box_case(NPS=NPS, NXB=40)
    itip = int(np.argmax(c2["nodes"][:, 0]))
    EBd = c2["P"] * c2["L"] ** 3 / (3 * E * c2["h"] * c2["Iyy"])
    out = []
    for tag, fn in [("ORIG", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
        sb._L_lg = fn
        u = sb.solve_static(c2["nodes"], c2["quads"], c2["ABD"], c2["Gs"], c2["f"], c2["root"])
        out.append(u[6 * itip + 2] / EBd)
    print("  nodes/side=%3d (Ns=%3d):  tip/EB  ORIG=%.4f   FIXED=%.4f" % (NPS, c2["Ns"], out[0], out[1]))

print()
print("=" * 100)
print("NO-REGRESSION: flat plate + cylinder validations with the corrected _L_lg")
print("=" * 100)
for tag, fn in [("ORIG", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
    sb._L_lg = fn
    print("--- %s ---" % tag)
    fe, an = sb.validate_plate(nx=24)
    print("    plate ratio  = %.4f" % (fe / an))
    fe, an = sb.validate_cylinder(mesh=(120, 60), bc="SS", verbose=False)
    print("    SS cyl ratio = %.4f" % (fe / an))
sb._L_lg = _L_lg_orig
