"""h1_static_sanity.py -- H1: is the FE static solution u itself garbage?

(a) residual ||K u - f|| / ||f|| on the free DOF
(b) global equilibrium of the reactions at the clamped root (force AND moment)
(c) displacement field: spanwise profile, plane-section rotation vs slope
(d) conditioning of Kff
No production code is modified.
"""
import os, sys, time
import numpy as np
import scipy.sparse as spr
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

np.set_printoptions(linewidth=200)
NSE = bi.NSE; MPER = bb.MPER; NSTA = bb.NSTA; Ntot = bb.Ntot

t0 = time.time()
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]
ndof = 6 * len(nodes)
print("mesh %d nodes %d quads ndof=%d NS=%d Ntot=%d  (%.1fs)" % (len(nodes), len(quads), ndof, NS, Ntot, time.time() - t0))

t0 = time.time()
K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
print("K assembled (%.1fs)  nnz=%d" % (time.time() - t0, K.nnz))
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
fixed = np.asarray(root, int)
free = np.setdiff1d(np.arange(ndof), fixed)

t0 = time.time()
u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
print("static solved (%.1fs)" % (time.time() - t0))

# ---------------- (a) residual ----------------
Ku = K @ u
res = (Ku - f)[free]
print("\n=== (a) RESIDUAL ===")
print("  ||K u - f||_2 (free) / ||f||_2 (free) = %.3e" % (np.linalg.norm(res) / np.linalg.norm(f[free])))
print("  ||K u - f||_inf (free)                = %.3e   (||f||_inf free = %.3e)"
      % (np.abs(res).max(), np.abs(f[free]).max()))

# ---------------- (b) reactions / global equilibrium ----------------
R = (Ku - f)                       # nonzero only on fixed dof (up to solver residual)
Rn = R.reshape(-1, 6)
fx = f.reshape(-1, 6)[:, :3]
rootn = np.arange(Ntot)            # plane-0 nodes = clamped ring
Fr = Rn[rootn, :3].sum(0)
Cr = Rn[rootn, 3:].sum(0)          # reaction couples
Mr = np.cross(nodes[rootn], Rn[rootn, :3]).sum(0) + Cr
Fapp = fx.sum(0)
Mapp = np.cross(nodes, fx).sum(0)
print("\n=== (b) GLOBAL EQUILIBRIUM (reactions at clamped root) ===")
print("  applied  F = %s" % np.array2string(Fapp, precision=4))
print("  reaction F = %s   -> F_react/-F_app = %s"
      % (np.array2string(Fr, precision=4), np.array2string(-Fr / np.where(np.abs(Fapp) > 1, Fapp, np.nan), precision=4)))
print("  applied  M(about origin) = %s" % np.array2string(Mapp, precision=4))
print("  reaction M(about origin) = %s" % np.array2string(Mr, precision=4))
print("    -> M_react/-M_app      = %s"
      % np.array2string(-Mr / np.where(np.abs(Mapp) > 1, Mapp, np.nan), precision=4))
print("  FF[0] (beam, about origin) = %s" % np.array2string(FF[0], precision=4))
print("  reaction couple part Cr    = %s   (fraction of M_react: %s)"
      % (np.array2string(Cr, precision=4),
         np.array2string(Cr / np.where(np.abs(Mr) > 1, Mr, np.nan), precision=3)))
# how much moment is absorbed by grounded (non-rigid-invariant) springs anywhere in the mesh
imbF = Fapp + Fr
imbM = Mapp + Mr
print("  IMBALANCE  F = %s   (rel %.3e)" % (np.array2string(imbF, precision=4),
                                            np.linalg.norm(imbF) / (np.linalg.norm(Fapp) + 1e-30)))
print("  IMBALANCE  M = %s   (rel %.3e)" % (np.array2string(imbM, precision=4),
                                            np.linalg.norm(imbM) / (np.linalg.norm(Mapp) + 1e-30)))
print("  (a grounded rotational spring absorbs MOMENT but no force -> F balances, M does not)")

# ---------------- (c) displacement field ----------------
U = u.reshape(-1, 6)
print("\n=== (c) DISPLACEMENT FIELD ===")
itip = int(np.argmax(nodes[:, 0]))
print("  tip node flap u_z = %.4f m ; u_y = %.4f ; u_x = %.4f" % (U[itip, 2], U[itip, 1], U[itip, 0]))
print("\n  layer   X[m]    mean_uz    max|uz|    mean_ux   std_ux   planefit c(=du_x/dz)  -dw/dx(FD)   c/(-dw/dx)   fit_resid")
prof_uz = np.zeros(NS); prof_c = np.zeros(NS)
rows = []
for p in range(NS):
    sl = slice(p * Ntot, (p + 1) * Ntot)
    Xp = nodes[sl][0, 0]
    up = U[sl]
    y = nodes[sl][:, 1]; z = nodes[sl][:, 2]
    A = np.column_stack([np.ones(Ntot), y, z])
    coef, *_ = np.linalg.lstsq(A, up[:, 0], rcond=None)
    fit = A @ coef
    resid = np.linalg.norm(up[:, 0] - fit) / (np.linalg.norm(up[:, 0]) + 1e-30)
    prof_uz[p] = up[:, 2].mean(); prof_c[p] = coef[2]
    rows.append((p, Xp, up[:, 2].mean(), np.abs(up[:, 2]).max(), up[:, 0].mean(), up[:, 0].std(), coef[2], resid))
dwdx = np.gradient(prof_uz, nodes[::Ntot, 0])
for (p, Xp, mz, xz, mx, sx, c, resid) in rows:
    if p % 5 == 0 or p == NS - 1:
        r = c / (-dwdx[p]) if abs(dwdx[p]) > 1e-12 else np.nan
        print("   %3d  %7.2f  %+.4e  %.4e  %+.3e %.3e   %+.4e        %+.4e   %7.3f    %.3f"
              % (p, Xp, mz, xz, mx, sx, c, -dwdx[p], r, resid))
print("  monotonic mean_uz in span? %s   (mean_uz[0]=%.3e  [-1]=%.3e)"
      % (bool(np.all(np.diff(prof_uz) >= -1e-12)), prof_uz[0], prof_uz[-1]))
# how much of the wall's axial displacement is the plane-section (beam) part
print("  plane-section share of u_x: layer 25 resid=%.3f  layer 50 resid=%.3f  layer 100 resid=%.3f"
      % (rows[25][7], rows[50][7], rows[100][7]))

# rotation dof: mean theta about chord (y) per layer, compare to plane-fit c
print("\n  layer     X      mean_rot_y   plane c   ratio(rot_y/c)    mean_rot_x   mean_rot_z")
for p in range(0, NS, 10):
    sl = slice(p * Ntot, (p + 1) * Ntot)
    up = U[sl]
    ry = up[:, 4].mean(); rx = up[:, 3].mean(); rz = up[:, 5].mean()
    print("   %3d  %7.2f   %+.4e  %+.4e   %8.3f       %+.3e  %+.3e"
          % (p, nodes[p * Ntot, 0], ry, prof_c[p], ry / prof_c[p] if abs(prof_c[p]) > 1e-30 else np.nan, rx, rz))

# ---------------- moment check reproduced here for reference ----------------
Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)


def moment_from_N(Ne, p, i):
    P = bl["Pk"][i]
    M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


print("\n=== moment ratio (reproduce the bug) ===")
for i in [5, 15, 25, 35, 45]:
    p = min(i * MPER, NS - 2)
    print("   sta %2d  M_FE/FF = %+.4f" % (i, moment_from_N(Nf, p, i) / FF[i][4]))

# ---------------- (d) conditioning ----------------
print("\n=== (d) CONDITIONING of Kff ===")
Kff = K[free][:, free].tocsc()
d = Kff.diagonal()
print("  diag: min=%.3e  max=%.3e  ratio=%.3e   n_nonpos=%d" % (d.min(), d.max(), d.max() / d.min(), int((d <= 0).sum())))
# translational vs rotational diag scale
dof_kind = np.arange(ndof)[free] % 6
for lbl, sel in [("trans u", dof_kind < 3), ("rot   th", dof_kind >= 3)]:
    print("  %s diag: min=%.3e median=%.3e max=%.3e" % (lbl, d[sel].min(), np.median(d[sel]), d[sel].max()))
try:
    from scipy.sparse.linalg import eigsh
    t0 = time.time()
    w = eigsh(Kff, k=4, sigma=0.0, which="LM", return_eigenvectors=False)
    wmx = eigsh(Kff, k=1, which="LM", return_eigenvectors=False)
    print("  smallest 4 eigenvalues of Kff: %s  (%.0fs)" % (np.array2string(np.sort(w), precision=4), time.time() - t0))
    print("  largest eigenvalue: %.4e  -> cond ~ %.3e" % (wmx[0], wmx[0] / np.sort(w)[0]))
except Exception as e:
    print("  eigsh failed: %s" % e)
