"""h2_drill_moment.py -- SECOND defect: the drilling penalty destroys moment equilibrium.

sum-of-moments of (K u) about any point is zero  <=>  K annihilates rigid-body ROTATION.
shell_buckling adds   Kn = kdr * outer(e3,e3)   to the ABSOLUTE nodal rotation block, i.e. it penalizes
theta.e3 itself -- including the RIGID-BODY part.  So K v_rot != 0, the element is not moment-objective,
and the root reaction moment does NOT balance the applied moment (measured: 0.855 with _L_lg fixed).
On a blade this is severe because every VERTICAL wall (shear webs, LE, TE) has e3 ~ chordwise, so
theta.e3 IS the flapwise bending rotation -> the 'drilling' spring fights beam bending directly.

Sweep _KDR_SCALE with the corrected _L_lg and watch the moment ratio -> -1.
"""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUCK)
import shell_buckling as sb

E, NU = 3.0e10, 0.3
_L_lg_orig = sb._L_lg


def _L_lg_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


# ---------------------------------------------------------------- box beam KDR sweep
def loft(sec_pts, sec_elems, xs):
    Ntot = len(sec_pts); NS = len(xs)
    pts = np.zeros((NS * Ntot, 3))
    for p in range(NS):
        pts[p * Ntot:(p + 1) * Ntot, 0] = xs[p]; pts[p * Ntot:(p + 1) * Ntot, 1:] = sec_pts
    quads = [[p * Ntot + b, (p + 1) * Ntot + b, (p + 1) * Ntot + a, p * Ntot + a]
             for p in range(NS - 1) for (a, b) in sec_elems]
    return pts, np.array(quads)


NPS, NXB, Lb, a_, hb, Pb = 10, 40, 20.0, 0.5, 0.01, 1.0e4
side = np.linspace(-a_, a_, NPS + 1)
loop = ([(y, -a_) for y in side[:-1]] + [(a_, z) for z in side[:-1]] +
        [(y, a_) for y in side[::-1][:-1]] + [(-a_, z) for z in side[::-1][:-1]])
secB = np.array(loop); Ns = len(secB)
secB_e = [(i, (i + 1) % Ns) for i in range(Ns)]
xsB = np.linspace(0, Lb, NXB + 1)
nodesB, quadsB = loft(secB, secB_e, xsB)
ABDb, Gsb = sb._iso_ABD(E, NU, hb)
ABD_eB = np.repeat(ABDb[None], len(quadsB), 0); Gs_eB = np.repeat(Gsb[None], len(quadsB), 0)
rootB = np.unique([6 * n + k for n in range(Ns) for k in range(6)])
fB = np.zeros(6 * len(nodesB))
for j in range(Ns):
    fB[6 * (NXB * Ns + j) + 2] = Pb / Ns
Iyy = 16.0 * a_ ** 3 / 3.0
EBd = Pb * Lb ** 3 / (3 * E * hb * Iyy)

print("=" * 100)
print("BOX BEAM: drilling-penalty sweep with corrected _L_lg   (M_y ratio should -> -1.000)")
print("=" * 100)
sb._L_lg = _L_lg_fixed
print("  %10s | %8s | %s" % ("KDR_SCALE", "tip/EB", "M_y(-oint N11 z ds)/M_applied at planes 2/10/20/30/37"))
for kdr in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 0.0]:
    sb._KDR_SCALE = kdr
    try:
        u = sb.solve_static(nodesB, quadsB, ABD_eB, Gs_eB, fB, rootB)
    except Exception as ex:
        print("  %10.0e | SINGULAR (%s)" % (kdr, type(ex).__name__)); continue
    Nfe = sb.element_membrane_N(nodesB, quadsB, ABD_eB, u)
    row = []
    for p in [2, 10, 20, 30, 37]:
        xm = 0.5 * (xsB[p] + xsB[p + 1]); Mapp = -Pb * (Lb - xm); M = 0.0
        for j in range(Ns):
            aa, bb_ = secB_e[j]; pa, pb = secB[aa], secB[bb_]
            M += -Nfe[p * Ns + j, 0] * 0.5 * (pa[1] + pb[1]) * np.linalg.norm(pb - pa)
        row.append(M / Mapp)
    itip = int(np.argmax(nodesB[:, 0]))
    print("  %10.0e | %8.4f | %s" % (kdr, u[6 * itip + 2] / EBd, "  ".join("%7.4f" % v for v in row)))
sb._KDR_SCALE = 1e-3

# ---------------------------------------------------------------- blade KDR sweep
print()
print("=" * 100)
print("BLADE: drilling-penalty sweep with corrected _L_lg")
print("=" * 100)
import blade_iso as bi
import blade_buckling as bb
NSE = bi.NSE
sb._L_lg = _L_lg_fixed
bl = bi.build()
nodes, quads = bl["nodes"], bl["quads"]
ABD_e, Gs_e, root = bl["ABD_e"], bl["Gs_e"], bl["root"]
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
itip = int(np.argmax(nodes[:, 0]))
print("  %10s | %8s | %10s | %s" % ("KDR_SCALE", "tip[m]", "reacMy/FF", "membrane-moment ratio at sta 5/15/25/35/45"))
for kdr in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 0.0]:
    sb._KDR_SCALE = kdr
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    try:
        u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
    except Exception as ex:
        print("  %10.0e | SINGULAR (%s)" % (kdr, type(ex).__name__)); continue
    R = (K @ u) - f
    Rx = R.reshape(-1, 6)
    Mre = (np.cross(nodes, Rx[:, :3]).sum(0) + Rx[:, 3:6].sum(0))[1]
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    row = []
    for i in [5, 15, 25, 35, 45]:
        p = min(i * bb.MPER, bl["NS"] - 2); P = bl["Pk"][i]; M = 0.0
        for se in range(NSE):
            a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
            M += -Nf[p * NSE + se, 0] * 0.5 * (P[a, 1] + P[b, 1]) * np.linalg.norm(P[b] - P[a])
        row.append(M / FF[i][4])
    print("  %10.0e | %8.3f | %10.4f | %s"
          % (kdr, u[6 * itip + 2], -Mre / FF[0][4], "  ".join("%7.4f" % v for v in row)))
sb._KDR_SCALE = 1e-3
sb._L_lg = _L_lg_orig
