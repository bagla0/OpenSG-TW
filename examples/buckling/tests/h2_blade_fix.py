"""h2_blade_fix.py -- (1) rigid-body-rotation patch test of the shell element on a FOLDED mesh,
(2) the actual dbg_blade_equil moment table recomputed with the corrected _L_lg."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUCK)
import shell_buckling as sb

_L_lg_orig = sb._L_lg


def _L_lg_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


# ---------------------------------------------------------------- (1) rigid-body patch test
print("=" * 100)
print("(1) RIGID-BODY ROTATION PATCH TEST   u = theta x r , rot dof = theta   -> strain energy must be 0")
print("=" * 100)
E, NU, h = 3.0e10, 0.3, 0.01
a_ = 0.5
side = np.linspace(-a_, a_, 6)
loop = ([(y, -a_) for y in side[:-1]] + [(a_, z) for z in side[:-1]] +
        [(y, a_) for y in side[::-1][:-1]] + [(-a_, z) for z in side[::-1][:-1]])
sec = np.array(loop); Ns = len(sec)
xs = np.linspace(0, 4.0, 9)
pts = np.zeros((len(xs) * Ns, 3))
for p in range(len(xs)):
    pts[p * Ns:(p + 1) * Ns, 0] = xs[p]; pts[p * Ns:(p + 1) * Ns, 1:] = sec
quads = np.array([[p * Ns + (j + 1) % Ns, (p + 1) * Ns + (j + 1) % Ns, (p + 1) * Ns + j, p * Ns + j]
                  for p in range(len(xs) - 1) for j in range(Ns)])
ABD, Gs = sb._iso_ABD(E, NU, h)
ABD_e = np.repeat(ABD[None], len(quads), 0); Gs_e = np.repeat(Gs[None], len(quads), 0)

# also a single FLAT plate patch, for contrast
nxp = 4
XX, YY = np.meshgrid(np.linspace(0, 1, nxp + 1), np.linspace(0, 1, nxp + 1), indexing="ij")
pptsf = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
idf = lambda i, j: i * (nxp + 1) + j
qf = np.array([[idf(i, j), idf(i + 1, j), idf(i + 1, j + 1), idf(i, j + 1)]
               for i in range(nxp) for j in range(nxp)])
ABD_f = np.repeat(ABD[None], len(qf), 0); Gs_f = np.repeat(Gs[None], len(qf), 0)

for meshname, P_, Q_, A_, G_ in [("FLAT plate (single e3)", pptsf, qf, ABD_f, Gs_f),
                                 ("BOX tube  (4 folds)   ", pts, quads, ABD_e, Gs_e)]:
    print("  %s   ref-energy scale = ||K||*|u|^2" % meshname)
    for tag, fn in [("ORIG ", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
        sb._L_lg = fn
        K = sb.assemble_K(P_, Q_, A_, G_)
        row = []
        for axis in range(3):
            th = np.zeros(3); th[axis] = 1e-4
            u = np.zeros(6 * len(P_))
            for n in range(len(P_)):
                u[6 * n:6 * n + 3] = np.cross(th, P_[n]); u[6 * n + 3:6 * n + 6] = th
            Uen = 0.5 * u @ (K @ u)
            scale = 0.5 * np.abs(K).max() * (u @ u)
            row.append(Uen / scale)
        print("    [%s]  U_rigid/scale about (x,y,z) = %.3e  %.3e  %.3e" % (tag, *row))

# ---------------------------------------------------------------- (2) the blade
print()
print("=" * 100)
print("(2) BLADE section-equilibrium table (dbg_blade_equil) with ORIG vs FIXED _L_lg")
print("=" * 100)
sb._L_lg = _L_lg_orig
import blade_iso as bi
import blade_buckling as bb
NSE = bi.NSE; MPER = bb.MPER
t0 = time.time(); bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
print("  mesh %d nodes %d quads  (%.0fs)" % (len(nodes), len(quads), time.time() - t0))
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
itip = int(np.argmax(nodes[:, 0]))


def moment_from_N(Ne, p, i, comp=0):
    P = bl["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, comp] * zmid * ds
    return M


res = {}
for tag, fn in [("ORIG", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
    sb._L_lg = fn
    t1 = time.time()
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root)
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    res[tag] = (u, Nf)
    print("  [%s] tip flap disp = %.4f m   (%.0fs)" % (tag, u[6 * itip + 2], time.time() - t1))

print("\n   sta   FF_My(applied)    M_y(FE ORIG)   ratio    M_y(FE FIXED)   ratio")
for i in [5, 15, 25, 35, 45]:
    p = min(i * MPER, bl["NS"] - 2)
    ff = FF[i][4]
    mo = moment_from_N(res["ORIG"][1], p, i); mf = moment_from_N(res["FIXED"][1], p, i)
    print("   %2d   %+.4e     %+.4e  %7.3f   %+.4e  %7.3f" % (i, ff, mo, mo / ff, mf, mf / ff))

print("\n   per-component moment ratio with FIXED (which component carries M_y?)  station 25:")
i = 25; p = min(i * MPER, bl["NS"] - 2)
for k, nm in enumerate(["N11(span)", "N22(arc)", "N12(shear)"]):
    print("     %-11s  -oint N z ds / FF_My = %+8.4f" % (nm, moment_from_N(res["FIXED"][1], p, i, k) / FF[i][4]))

np.savez(os.path.join(BUCK, "data", "h2_blade_fix.npz"),
         N_orig=res["ORIG"][1], N_fixed=res["FIXED"][1], u_orig=res["ORIG"][0], u_fixed=res["FIXED"][0])
sb._L_lg = _L_lg_orig
