"""h2_closedform.py -- VALIDATE element_membrane_N (+ solve_static) against CLOSED-FORM beam theory.

Two self-built benchmarks with an exact answer, using the SAME element/frame/assembly code path
(and the SAME quad ordering convention) as blade_iso.build():

  A) FLAT STRIP, in-plane bending.  Rectangle in the x-y plane, length L (x), height 2c (y),
     thickness h.  Clamped at x=0, tip shear P in +y.  PURE MEMBRANE:  N11(x,y) = -P(L-x) y / Iz,
     Iz = (2c)^3/12 (per unit thickness, i.e. N = sigma*h with I_area = h(2c)^3/12).

  B) SQUARE BOX BEAM cantilever, side 2a, wall thickness h, length L, tip load P in +z.
     Closed thin-walled section, 4 folds -> the blade topology in miniature.
     Statics (exact, independent of I):   M_y(x) = -oint N11 z ds  must equal  -P (L - x).
     Beam theory:                          N11 = -M_y z h / I_yy  (checked too).

Everything is built with the identical loft/quad convention as blade_iso:
    quads.append([b0+b, b1+b, b1+a, b0+a])   ->  _elem_frame v1 = X[1]-X[0] = SPAN  -> e1 = span.
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUCK)
import shell_buckling as sb

E, NU = 3.0e10, 0.3
np.set_printoptions(precision=4, suppress=False, linewidth=150)


def loft(sec_pts, sec_elems, xs):
    """sec_pts (Ntot,2)=(y,z); sec_elems (NSE,2); xs (NS,) span stations -> nodes,quads (blade_iso convention)."""
    Ntot = len(sec_pts); NS = len(xs)
    pts = np.zeros((NS * Ntot, 3))
    for p in range(NS):
        pts[p * Ntot:(p + 1) * Ntot, 0] = xs[p]
        pts[p * Ntot:(p + 1) * Ntot, 1:] = sec_pts
    quads = []
    for p in range(NS - 1):
        b0, b1 = p * Ntot, (p + 1) * Ntot
        for (a, b) in sec_elems:
            quads.append([b0 + b, b1 + b, b1 + a, b0 + a])       # identical to blade_iso.build
    return pts, np.array(quads)


# =====================================================================================
print("=" * 100)
print("A) FLAT STRIP, in-plane bending  (pure membrane, exact N11 = -P(L-x) y / Iz)")
print("=" * 100)
L, c, h = 10.0, 0.5, 0.02
NX, NY = 40, 16                                    # span elems, height elems
ys = np.linspace(-c, c, NY + 1)
sec = np.column_stack([ys, np.zeros(NY + 1)])      # (y, z=0) -> strip lies in the x-y plane
sec_e = [(j, j + 1) for j in range(NY)]
xs = np.linspace(0, L, NX + 1)
nodes, quads = loft(sec, sec_e, xs)
Ntot = len(sec); ne = len(quads)
ABD, Gs = sb._iso_ABD(E, NU, h)
ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)

# element frame check
T0, _ = sb._elem_frame(nodes, quads[0])
print("  elem0 frame  e1=%s  e2=%s  e3=%s   (e1 must be the SPAN [1,0,0])"
      % (np.array2string(T0[0], precision=4), np.array2string(T0[1], precision=4),
         np.array2string(T0[2], precision=4)))

P = 1.0e4
fixed = []
for n in range(len(nodes)):                        # kill out-of-plane w, and rot about x,y (keep drilling rz)
    fixed += [6 * n + 2, 6 * n + 3, 6 * n + 4]
for n in range(Ntot):                              # clamp root plane
    fixed += [6 * n + k for k in range(6)]
fixed = np.unique(fixed)
f = np.zeros(6 * len(nodes))
tipbase = NX * Ntot
# parabolic (exact) shear flow at the tip so no local end effect: q(y) ~ (c^2 - y^2)
wq = (c ** 2 - ys ** 2); wq = wq / wq.sum()
for j in range(Ntot):
    f[6 * (tipbase + j) + 1] = P * wq[j]
u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, fixed)
Nfe = sb.element_membrane_N(nodes, quads, ABD_e, u)

Iz = (2 * c) ** 3 / 12.0                            # per unit thickness -> N11 = sigma*h
print("  tip y-disp = %.6e m   (Euler-Bernoulli P L^3/(3 E I_area) = %.6e)"
      % (u[6 * (tipbase + NY // 2) + 1], P * L ** 3 / (3 * E * h * Iz)))
print("\n  %6s %10s | %13s %13s %8s | %13s %13s" %
      ("elem", "y", "N11_FE", "N11_exact", "ratio", "N22_FE", "N12_FE"))
for p in [5, 15, 25, 35]:
    xmid = 0.5 * (xs[p] + xs[p + 1])
    for j in [0, NY // 4, NY // 2, 3 * NY // 4, NY - 1]:
        e = p * NY + j
        ymid = 0.5 * (ys[sec_e[j][0]] + ys[sec_e[j][1]])
        ex = -P * (L - xmid) * ymid / Iz
        r = Nfe[e, 0] / ex if abs(ex) > 1e-12 else np.nan
        print("  %6d %10.4f | %+13.5e %+13.5e %8.4f | %+13.5e %+13.5e"
              % (e, ymid, Nfe[e, 0], ex, r, Nfe[e, 1], Nfe[e, 2]))
# moment integral
print("\n  station-moment check  Mz = int N11 y dy   vs  -P(L-x):")
for p in [2, 10, 20, 30, 38]:
    xmid = 0.5 * (xs[p] + xs[p + 1])
    M = 0.0
    for j in range(NY):
        a, b = sec_e[j]
        ymid = 0.5 * (ys[a] + ys[b]); ds = abs(ys[b] - ys[a])
        M += Nfe[p * NY + j, 0] * ymid * ds
    print("    x=%6.3f   int N11 y dy = %+.5e   -P(L-x) = %+.5e   ratio = %.4f"
          % (xmid, M, -P * (L - xmid), M / (-P * (L - xmid))))

# =====================================================================================
print()
print("=" * 100)
print("B) SQUARE BOX BEAM cantilever (closed section, 4 folds -- blade topology in miniature)")
print("=" * 100)
Lb, a_, hb = 20.0, 0.5, 0.01
NPS = 10                                            # nodes per side  -> Ns = 4*NPS section nodes
NXB = 40
side = np.linspace(-a_, a_, NPS + 1)
loop = []
loop += [(y, -a_) for y in side[:-1]]               # bottom  z=-a
loop += [(a_, z) for z in side[:-1]]                # right   y=+a
loop += [(y, a_) for y in side[::-1][:-1]]          # top     z=+a
loop += [(-a_, z) for z in side[::-1][:-1]]         # left    y=-a
secB = np.array(loop); Ns = len(secB)
secB_e = [(i, (i + 1) % Ns) for i in range(Ns)]
xsB = np.linspace(0, Lb, NXB + 1)
nodesB, quadsB = loft(secB, secB_e, xsB)
neB = len(quadsB)
ABDb, Gsb = sb._iso_ABD(E, NU, hb)
ABD_eB = np.repeat(ABDb[None], neB, 0); Gs_eB = np.repeat(Gsb[None], neB, 0)
print("  box mesh: %d nodes  %d quads   (Ns=%d section nodes, NX=%d)" % (len(nodesB), neB, Ns, NXB))

Pb = 1.0e4
rootB = np.unique([6 * n + k for n in range(Ns) for k in range(6)])
fB = np.zeros(6 * len(nodesB))
tb = NXB * Ns
for j in range(Ns):
    fB[6 * (tb + j) + 2] = Pb / Ns                  # uniform tip ring load in +z
uB = sb.solve_static(nodesB, quadsB, ABD_eB, Gs_eB, fB, rootB)
NfeB = sb.element_membrane_N(nodesB, quadsB, ABD_eB, uB)
itip = int(np.argmax(nodesB[:, 0]))
Iyy = 16.0 * a_ ** 3 / 3.0                          # per unit thickness (I_area = hb * Iyy)
print("  tip z-disp = %.6e m   (EB  P L^3/(3 E I_area) = %.6e)"
      % (uB[6 * itip + 2], Pb * Lb ** 3 / (3 * E * hb * Iyy)))

print("\n  element frames at span-plane p=20 (should be e1 = span = [1,0,0]):")
for j in [0, NPS // 2, NPS, 2 * NPS, 3 * NPS]:
    Tj, _ = sb._elem_frame(nodesB, quadsB[20 * Ns + j])
    print("    se=%3d  mid(y,z)=(%+.3f,%+.3f)  e1=%s e2=%s e3=%s"
          % (j, *(0.5 * (secB[secB_e[j][0]] + secB[secB_e[j][1]])),
             np.array2string(Tj[0], precision=3), np.array2string(Tj[1], precision=3),
             np.array2string(Tj[2], precision=3)))

print("\n  N11 distribution around the section at span-plane p=20 (x=%.2f):" % (0.5 * (xsB[20] + xsB[21])))
xm = 0.5 * (xsB[20] + xsB[21]); My = -Pb * (Lb - xm)
print("  %5s %8s %8s | %13s %13s %8s | %13s %13s" %
      ("se", "y", "z", "N11_FE", "N11_beam", "ratio", "N22_FE", "N12_FE"))
for j in range(0, Ns, max(1, Ns // 16)):
    e = 20 * Ns + j
    ym, zm = 0.5 * (secB[secB_e[j][0]] + secB[secB_e[j][1]])
    exb = -My * zm * hb / (hb * Iyy)
    print("  %5d %8.3f %8.3f | %+13.5e %+13.5e %8.4f | %+13.5e %+13.5e"
          % (j, ym, zm, NfeB[e, 0], exb, NfeB[e, 0] / exb if abs(exb) > 1e-12 else np.nan,
             NfeB[e, 1], NfeB[e, 2]))

print("\n  SECTION-EQUILIBRIUM (the blade test):  M_y = -oint N11 z ds   vs  -P(L-x)")
print("  %5s %9s | %14s %14s %8s | %14s %14s" % ("plane", "x", "M_y(FE)", "M_y(applied)", "ratio", "Mz=+oint N11 y ds", "F1=oint N11 ds"))
for p in [2, 5, 10, 20, 30, 37]:
    xm = 0.5 * (xsB[p] + xsB[p + 1]); Mapp = -Pb * (Lb - xm)
    M = 0.0; Mz = 0.0; F1 = 0.0
    for j in range(Ns):
        aa, bb_ = secB_e[j]
        pa, pb = secB[aa], secB[bb_]
        ds = np.linalg.norm(pb - pa); ym, zm = 0.5 * (pa + pb)
        n11 = NfeB[p * Ns + j, 0]
        M += -n11 * zm * ds; Mz += n11 * ym * ds; F1 += n11 * ds
    print("  %5d %9.4f | %+14.5e %+14.5e %8.4f | %+14.5e %+14.5e" % (p, xm, M, Mapp, M / Mapp, Mz, F1))

# also: all three components' moment integral, to see if the bending went into N22 or N12
print("\n  moment integral computed from EACH component (p=20): which one carries M_y?")
for k, nm in enumerate(["N11", "N22", "N12"]):
    M = 0.0
    for j in range(Ns):
        aa, bb_ = secB_e[j]
        pa, pb = secB[aa], secB[bb_]
        ds = np.linalg.norm(pb - pa); zm = 0.5 * (pa[1] + pb[1])
        M += -NfeB[20 * Ns + j, k] * zm * ds
    xm = 0.5 * (xsB[20] + xsB[21])
    print("    -oint %s z ds = %+.5e   / M_applied = %8.4f" % (nm, M, M / (-Pb * (Lb - xm))))
