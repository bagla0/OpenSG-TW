"""
verify_strains_paper.py -- STRAIN-BY-STRAIN verification of quad_ops_general
against the VERBATIM RM-paper expressions (Overleaf opensg-rm-timo main.tex):
  membrane   e11s, e22s, 2e12s          (Shell Strains, lines 310-316)
  shear      2e13s, 2e23s               (Shell Strains, lines 317-326)
  curvature  k11s, k22s, k12s+k21s      (Appendix, lines 903-906)
  drilling   omega_3, Lambda_alpha      (Appendix, lines 851, 882-891)

Method (no analytic guessing):
  * a linearly tapered CONE patch at (theta0, z0); the exact surface frame and
    ALL geometric coefficient derivatives (.)_{;alpha} -- x_{i;beta;alpha},
    y_{i;alpha}, C33_{;alpha}, Rn_{;alpha} -- by CENTRAL FINITE DIFFERENCE of
    the analytic frame/coordinates along the hoop (zeta_2) and generator
    (zeta_1) arcs;
  * smooth analytic test fields w_i, omega_beta, w'_i, omega'_beta (independent,
    as in the Gamma_l structure) and constant beam strains eb;
  * PAPER side: every row evaluated verbatim with those exact derivatives;
    the documented drops are applied EXPLICITLY and their size is REPORTED:
      DROP-1  w_{i|1|alpha}, w_{i|2|alpha}   (2nd SG derivative, C0-RM)
      DROP-2  kappa'_11 term of Lambda       (needs eb'; untestable here, eb=const)
  * CODE side: quad_ops_general on a tiny quad of the same cone, rows dotted
    with the nodal field samples; k22 passed as the exact hoop normal curvature.
  * chain rule (paper convention): the TOTAL zeta_alpha derivative of a
    fluctuation splits micro + macro:  d(w_i)/dzeta_alpha -> w_{i|alpha}
    + x_{1;alpha} w'_i, which is why every w_{i|alpha} coefficient re-appears
    times x_{1;alpha} on w'_i.  The same applies to omega_beta and to w' itself
    (whose macro part w'' is DROP-2-adjacent and excluded by both sides).

Run:  python verify_strains_paper.py
"""
import os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
from segment_element_general import quad_ops_general, NDOF, C33_EPS

# ------------------------------------------------------------------ geometry
R0, m = 1.0, -0.15                 # cone R(z) = R0 + m z   (aR=0.7 over L=2)
z0, th0 = 1.0, 0.7                 # test point (away from y3~0 poles)
AX, CROSS = 2, (0, 1)              # beam axis = z ; b1=z, b2=x, b3=y


def Rz(z):
    return R0 + m * z


def point(th, z):
    return np.array([Rz(z) * math.cos(th), Rz(z) * math.sin(th), z])


def frame(th):
    """exact cone surface frame (axially invariant): a1=generator, a2=hoop CCW,
    n = a1 x a2 (inward for CCW hoop)."""
    a2 = np.array([-math.sin(th), math.cos(th), 0.0])
    g = np.array([m * math.cos(th), m * math.sin(th), 1.0])
    a1 = g / np.linalg.norm(g)
    n = np.cross(a1, a2)
    return a1, a2, n


def beam_comp(v):
    """global vector -> beam components [b1=axis, b2=cross0, b3=cross1]."""
    return np.array([v[AX], v[CROSS[0]], v[CROSS[1]]])


# d/dzeta_2 = (1/R) d/dtheta at fixed z ; d/dzeta_1 = d/dz / sqrt(1+m^2) along generator
S1F = 1.0 / math.sqrt(1.0 + m * m)
H = 1e-6


def dz2(f, th=th0, z=z0):
    return (f(th + H, z) - f(th - H, z)) / (2 * H) / Rz(z)


def dz1(f, th=th0, z=z0):
    return (f(th, z + H) - f(th, z - H)) / (2 * H) * S1F


# geometric quantities as functions of (th, z) in beam components
def geo(th, z):
    a1, a2, n = frame(th)
    P = point(th, z)
    xi1 = beam_comp(a1); xi2 = beam_comp(a2); yv = beam_comp(n)
    Pb = beam_comp(P)
    return dict(xi1=xi1, xi2=xi2, yv=yv, x2=Pb[1], x3=Pb[2])


G0 = geo(th0, z0)
xi1, xi2, yv = G0["xi1"], G0["xi2"], G0["yv"]
x2, x3 = G0["x2"], G0["x3"]
C33 = yv[2]                                      # y3 (healthy at th0=0.7)
Rn1 = x2 * xi1[2] - x3 * xi1[1]
Rn2 = x2 * xi2[2] - x3 * xi2[1]

# exact coefficient derivatives (FD of the analytic frame/coords)
def gd(key, alpha):
    f = lambda th, z: geo(th, z)[key]
    return dz1(f) if alpha == 1 else dz2(f)


xi1_1, xi1_2 = gd("xi1", 1), gd("xi1", 2)        # (x_{i;1})_{;alpha}
xi2_1, xi2_2 = gd("xi2", 1), gd("xi2", 2)        # (x_{i;2})_{;alpha}
yv_1, yv_2 = gd("yv", 1), gd("yv", 2)            # (y_i)_{;alpha}
x2_1, x2_2 = gd("x2", 1), gd("x2", 2)            # coordinate derivatives
x3_1, x3_2 = gd("x3", 1), gd("x3", 2)
C33_1, C33_2 = yv_1[2], yv_2[2]
Rn1_a = [x2_1 * xi1[2] + x2 * xi1_1[2] - x3_1 * xi1[1] - x3 * xi1_1[1],
         x2_2 * xi1[2] + x2 * xi1_2[2] - x3_2 * xi1[1] - x3 * xi1_2[1]]
Rn2_a = [x2_1 * xi2[2] + x2 * xi2_1[2] - x3_1 * xi2[1] - x3 * xi2_1[1],
         x2_2 * xi2[2] + x2 * xi2_2[2] - x3_2 * xi2[1] - x3 * xi2_2[1]]
k22_exact = float(yv_2 @ xi2)                    # code convention: d e3/ds . e2
kg_exact = float(xi1_2 @ xi2)                    # geodesic: d a1/ds . a2

# ------------------------------------------------------------------ fields
RNG = np.random.default_rng(7)
CW = RNG.normal(size=(3, 5)); CO = RNG.normal(size=(2, 5))
CWp = RNG.normal(size=(3, 5)); COp = RNG.normal(size=(2, 5))
EB = np.array([0.11, -0.23, 0.17, 0.09])         # [g11, k1, k2, k3] constant


def sfield(C, th, z):
    """smooth scalar field family: rows -> [sin2th, costh, z-lin, z^2, th*z]"""
    b = np.array([math.sin(2 * th), math.cos(th), (z - z0), (z - z0) ** 2, (th - th0) * (z - z0)])
    return C @ b


def w_f(th, z):   return sfield(CW, th, z)
def om_f(th, z):  return sfield(CO, th, z)
def wp_f(th, z):  return sfield(CWp, th, z)
def omp_f(th, z): return sfield(COp, th, z)


# micro (SG-surface) derivatives of the fields at P0, exact by FD
w0 = w_f(th0, z0); om0 = om_f(th0, z0); wp0 = wp_f(th0, z0); omp0 = omp_f(th0, z0)
w_1, w_2 = dz1(w_f), dz2(w_f)                    # w_{i|1}, w_{i|2}
om_1, om_2 = dz1(om_f), dz2(om_f)
wp_1, wp_2 = dz1(wp_f), dz2(wp_f)                # w'_{i|alpha}
# second SG derivatives (for sizing DROP-1)
w_11 = dz1(lambda th, z: dz1(w_f, th, z))
w_12 = dz2(lambda th, z: dz1(w_f, th, z))
w_21 = dz1(lambda th, z: dz2(w_f, th, z))
w_22 = dz2(lambda th, z: dz2(w_f, th, z))

g11, k1, k2, k3 = EB
x11, x12 = xi1[0], xi2[0]
y1, y2 = yv[0], yv[1]
x21, x31 = xi1[1], xi1[2]
x22, x32 = xi2[1], xi2[2]

# ------------------------------------------------------------------ PAPER rows
def paper_membrane():
    e11 = (x11 ** 2 * g11 + x11 * (-x21 * x3 + x31 * x2) * k1 + x11 ** 2 * x3 * k2 - x11 ** 2 * x2 * k3
           + float(xi1 @ w_1) + x11 * float(xi1 @ wp0))
    e22 = (x12 ** 2 * g11 + x12 * (-x22 * x3 + x32 * x2) * k1 + x12 ** 2 * x3 * k2 - x12 ** 2 * x2 * k3
           + float(xi2 @ w_2) + x12 * float(xi2 @ wp0))
    e12 = (2 * x11 * x12 * g11
           + (x2 * (x11 * x32 + x12 * x31) - x3 * (x22 * x11 + x12 * x21)) * k1
           + 2 * x11 * x12 * x3 * k2 - 2 * x11 * x12 * x2 * k3
           + float(xi2 @ w_1) + float(xi1 @ w_2)
           + float((x11 * xi2 + x12 * xi1) @ wp0))          # paper line 316 TYPO corrected
    return e11, e22, e12


def paper_shear(inv33):
    h33 = 0.5 * inv33
    k113 = (x2 * ((x32 * x32 * x11 - x32 * x31 * x12) * h33 + C33 * x11)
            - x3 * ((x32 * x22 * x11 - x32 * x21 * x12) * h33 + y2 * x11))
    k123 = (x2 * ((x31 * x31 * x12 - x31 * x32 * x11) * h33 + C33 * x12)
            - x3 * ((x31 * x21 * x12 - x31 * x22 * x11) * h33 + y2 * x12))
    a13 = x32 * xi2 * h33 + yv                    # w_{i|1} coeff of 2e13
    b13 = -x32 * xi1 * h33                        # w_{i|2} coeff
    a23 = x31 * xi1 * h33 + yv                    # w_{i|2} coeff of 2e23
    b23 = -x31 * xi2 * h33                        # w_{i|1} coeff
    om13 = np.array([x12 - x32 * y1 * inv33, x22 - x32 * y2 * inv33])
    om23 = np.array([-x11 + x31 * y1 * inv33, -x21 + x31 * y2 * inv33])
    e13 = (x11 * y1 * g11 + k113 * k1 + x11 * y1 * x3 * k2 - x11 * y1 * x2 * k3
           + float(a13 @ w_1) + float(b13 @ w_2) + float(om13 @ om0)
           + float((x11 * a13 + x12 * b13) @ wp0))          # chain: macro part of BOTH derivs
    e23 = (x12 * y1 * g11 + k123 * k1 + x12 * y1 * x3 * k2 - x12 * y1 * x2 * k3
           + float(a23 @ w_2) + float(b23 @ w_1) + float(om23 @ om0)
           + float((x12 * a23 + x11 * b23) @ wp0))
    return e13, e23


def paper_lambda(alpha, with_drop1=False):
    """Lambda_alpha, Eq. omega3alpha (lines 882-891), exact coefficients."""
    ia = alpha - 1
    x1a = (x11, x12)[ia]
    xi1_a = (xi1_1, xi1_2)[ia]; xi2_a = (xi2_1, xi2_2)[ia]
    yv_a = (yv_1, yv_2)[ia]; C33_a = (C33_1, C33_2)[ia]
    Rn1a = Rn1_a[ia]; Rn2a = Rn2_a[ia]
    wA = (w_1, w_2)[ia]                            # w_{i|alpha}
    wpA = (wp_1, wp_2)[ia]                         # w'_{i|alpha}
    omA = (om_1, om_2)[ia]                         # omega_{beta|alpha}
    tw = 2 * C33
    # kappa_11 coefficient-derivative group
    t1 = k1 * ((xi1_a[0] * Rn2 + x11 * Rn2a - xi2_a[0] * Rn1 - x12 * Rn1a) / tw
               - (x11 * Rn2 - x12 * Rn1) * (2 * C33_a) / (tw * tw))
    # kappa'_11 term: eb constant -> zero here (DROP-2, untestable)
    t3 = float(wpA @ (x11 * xi2 - x12 * xi1)) / tw
    t4 = (float(xi2 @ wp_1) - float(xi1 @ wp_2)) / tw * x1a
    t5 = float(wp0 @ ((xi1_a[0] * xi2 + x11 * xi2_a - xi2_a[0] * xi1 - x12 * xi1_a) / tw
                      - (x11 * xi2 - x12 * xi1) * (2 * C33_a) / (tw * tw)))
    t6 = 0.0
    if with_drop1:
        wA1 = (w_11, w_21)[ia]; wA2 = (w_12, w_22)[ia]   # w_{i|1|alpha}, w_{i|2|alpha}
        t6 = (float(xi2 @ wA1) - float(xi1 @ wA2)) / tw
    t7 = (float(w_1 @ (xi2_a / tw - xi2 * (2 * C33_a) / (tw * tw)))
          - float(w_2 @ (xi1_a / tw - xi1 * (2 * C33_a) / (tw * tw))))
    t8 = (-float((yv[:2] / C33) @ (omA + x1a * omp0))
          - float(om0 @ ((yv_a[:2] * C33 - C33_a * yv[:2]) / (C33 * C33))))
    return t1 + t3 + t4 + t5 + t6 + t7 + t8


def paper_curvature(with_drop1=False):
    L1 = paper_lambda(1, with_drop1); L2 = paper_lambda(2, with_drop1)
    kv = np.array([k1, k2, k3])
    k11s = (x11 * float(xi2 @ kv) + float(xi2[:2] @ (x11 * omp0 + om_1)) + x32 * L1)
    k22s = -(x12 * float(xi1 @ kv) + float(xi1[:2] @ (x12 * omp0 + om_2)) + x31 * L2)
    k12s = (float((x12 * xi2 - x11 * xi1) @ kv)
            + float((x12 * xi2[:2] - x11 * xi1[:2]) @ omp0)
            + float(xi2[:2] @ om_2) - float(xi1[:2] @ om_1)
            + x32 * L2 - x31 * L1)
    return k11s, k22s, k12s


# ------------------------------------------------------------------ CODE rows
dth, dzq = 2e-3, 2e-3
thm, thp = th0 - dth / 2, th0 + dth / 2
zm, zp = z0 - dzq / 2, z0 + dzq / 2
# winding [(z-,th-),(z-,th+),(z+,th+),(z+,th-)] -> xi ~ hoop, eta ~ axial
Xq = np.array([point(thm, zm), point(thp, zm), point(thp, zp), point(thm, zp)])
a1c, a2c, nc = frame(th0)
BDe, BDh, BDl, BGe, BGh, BGl, dA = quad_ops_general(
    Xq, a1c, a2c, nc, 0.0, 0.0, k22_exact, cross=CROSS, ax=AX, kg=kg_exact)
# Darboux-model residuals vs exact FD (surface torsion etc. -- must be ~0 on a cone)
print("Darboux check: |xi1_2 - kg*xi2| = %.2e   |xi2_2 + k22*yv + kg*xi1| = %.2e   |yv_2 - k22*xi2| = %.2e"
      % (np.abs(xi1_2 - kg_exact * xi2).max(),
         np.abs(xi2_2 + k22_exact * yv + kg_exact * xi1).max(),
         np.abs(yv_2 - k22_exact * xi2).max()))

vh = np.zeros(4 * NDOF); vl = np.zeros(4 * NDOF)
for a, (tt, zz) in enumerate([(thm, zm), (thp, zm), (thp, zp), (thm, zp)]):
    vh[NDOF * a:NDOF * a + 3] = w_f(tt, zz)
    vh[NDOF * a + 3:NDOF * a + 5] = om_f(tt, zz)
    vl[NDOF * a:NDOF * a + 3] = wp_f(tt, zz)
    vl[NDOF * a + 3:NDOF * a + 5] = omp_f(tt, zz)

code = [float(BDh[r] @ vh + BDl[r] @ vl + BDe[r] @ EB) for r in range(6)] + \
       [float(BGh[r] @ vh + BGl[r] @ vl + BGe[r] @ EB) for r in range(2)]

# ------------------------------------------------------------------ report
inv_exact = 1.0 / C33
inv_tik = C33 / (C33 * C33 + C33_EPS ** 2)
pm = paper_membrane()
ps_e = paper_shear(inv_exact)
ps_t = paper_shear(inv_tik)
pc = paper_curvature(False)
pc_full = paper_curvature(True)

print("tapered cone: m=%.3f  th0=%.2f  z0=%.2f   C33=y3=%.4f  k22=%.4f  kg(geodesic)=%.4f"
      % (m, th0, z0, C33, k22_exact, kg_exact))
print("%-8s %14s %14s %14s   %s" % ("row", "code", "paper", "|diff|", "note"))
names = ["e11", "e22", "2e12", "k11s", "k22s", "k12+21", "2e13", "2e23"]
paper_vals = list(pm) + list(pc) + list(ps_t)
notes = ["", "", "paper-typo corrected", "", "", "", "Tikhonov 1/C33", "Tikhonov 1/C33"]
for r in range(8):
    pv = paper_vals[[0, 1, 2, 3, 4, 5, 6, 7][r] if r < 6 else r]
    print("%-8s %14.8f %14.8f %14.2e   %s" % (names[r], code[r], pv, abs(code[r] - pv), notes[r]))
print("\nshear rows vs EXACT 1/C33 (regularization deviation at healthy C33=%.3f):" % C33)
for r, pv in ((6, ps_e[0]), (7, ps_e[1])):
    print("  %-6s |code-exact| = %.3e" % (names[r], abs(code[r] - pv)))
print("\nsize of DROP-1 (2nd SG derivative w_{i|j|a} group) in the curvature rows:")
for nm, a, b in (("k11s", pc[0], pc_full[0]), ("k22s", pc[1], pc_full[1]), ("k12+21", pc[2], pc_full[2])):
    print("  %-6s |dropped| = %.3e  (row scale %.3e)" % (nm, abs(b - a), abs(a) + 1e-30))
