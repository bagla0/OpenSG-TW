"""h1b_moment_sink.py -- follow-up to h1_static_sanity.

h1 showed: residual 9e-7 (solve fine), FORCE balances to 5e-10, but MOMENT imbalance = 63%.
A grounded rotational spring absorbs moment and no force -> prime suspect is the drilling penalty
Kn = kdr*outer(T2,T2) added to the NODAL 3x3 rotational diagonal block (a spring to GROUND, not a
relative/Hughes-Brezzi drilling constraint).  This script:

  1. validates element_membrane_N against a finite-difference axial strain
  2. measures kdr vs the element's physical bending rotational diagonal
  3. computes the moment absorbed by the drilling springs directly, sum kdr*(T2 T2^T) theta_node,
     and compares it to the 63% imbalance
  4. sweeps _KDR_SCALE and reports residual / tip flap / moment imbalance / M_FE(sta)/FF
"""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

np.set_printoptions(linewidth=220)
NSE = bi.NSE; MPER = bb.MPER; Ntot = bb.Ntot

bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]; ndof = 6 * len(nodes)
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
fx = f.reshape(-1, 6)[:, :3]
Fapp = fx.sum(0); Mapp = np.cross(nodes, fx).sum(0)
fixed = np.asarray(root, int); free = np.setdiff1d(np.arange(ndof), fixed)
flat = sb._flat_node_mask(nodes, quads)
print("mesh %d nodes %d quads; folds %d" % (len(nodes), len(quads), int((~flat).sum())))


def moment_from_N(Ne, p, i):
    P = bl["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


def axial_force_from_N(Ne, p, i):
    P = bl["Pk"][i]; F = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        F += Ne[p * NSE + se, 0] * np.linalg.norm(P[b] - P[a])
    return F


# ---------------- 2. drilling-penalty magnitude ----------------
print("\n=== 2. drilling spring magnitude vs physical rotational stiffness ===")
print("  elem     kdr(=%.0e*mean|diag_w|)   mean|diag_w|   bend-rot diag(Ke[3,3])   kdr/bendrot" % sb._KDR_SCALE)
for e in [0, 50, NSE * 25, NSE * 25 + 60, NSE * 49]:
    e = min(e, len(quads) - 1)
    T, xyl = sb._elem_frame(nodes, quads[e])
    Ke, _ = sb.element_K_KG(xyl, ABD_e[e], Gs_e[e], np.zeros(3))
    dw = np.mean(np.abs(np.diag(Ke)[2::5])); kdr = sb._KDR_SCALE * dw
    brot = np.mean(np.abs(np.diag(Ke)[3::5]))
    print("  %6d   %.4e            %.4e     %.4e             %8.2f" % (e, kdr, dw, brot, kdr / brot))


def run(kdr_scale, label, verbose=True):
    sb._KDR_SCALE = kdr_scale
    t0 = time.time()
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
    Ku = K @ u
    resr = np.linalg.norm((Ku - f)[free]) / np.linalg.norm(f[free])
    Rn = (Ku - f).reshape(-1, 6)
    rootn = np.arange(Ntot)
    Fr = Rn[rootn, :3].sum(0)
    Mr = np.cross(nodes[rootn], Rn[rootn, :3]).sum(0) + Rn[rootn, 3:].sum(0)
    imbM = Mapp + Mr
    itip = int(np.argmax(nodes[:, 0]))
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    rats = [moment_from_N(Nf, min(i * MPER, NS - 2), i) / FF[i][4] for i in [5, 15, 25, 35, 45]]
    if verbose:
        print("  %-10s res=%.2e  tip_uz=%.4f  Mimb_y/Mapp_y=%+.4f  Mreact_y/-Mapp_y=%+.4f  "
              "M_FE/FF[5,15,25,35,45]=%s   (%.0fs)"
              % (label, resr, u[6 * itip + 2], imbM[1] / Mapp[1], -Mr[1] / Mapp[1],
                 np.array2string(np.array(rats), precision=3), time.time() - t0))
    return u, Nf, K, imbM, Mr


# ---------------- default run ----------------
sb._KDR_SCALE = 1e-3
u, Nf, K, imbM, Mr = run(1e-3, "kdr=1e-3", verbose=False)
U = u.reshape(-1, 6)
print("\n=== baseline (kdr scale 1e-3) ===")
print("  Mapp=%s  Mreact=%s  imbalance=%s" % (np.array2string(Mapp, precision=4),
                                              np.array2string(Mr, precision=4), np.array2string(imbM, precision=4)))

# ---------------- 1. validate element_membrane_N with FD strain ----------------
print("\n=== 1. element_membrane_N vs finite-difference axial strain ===")
print("   elem   eps11(Bm)     eps11(FD du1/dx)   ratio     N11(code)   A11*eps_FD")
p = 50
for se in [0, 15, 30, 45, 60, 90]:
    e = p * NSE + se
    q = quads[e]; T, xyl = sb._elem_frame(nodes, q)
    L = sb._L_lg(T); ul = L @ np.concatenate([u[6 * n:6 * n + 6] for n in q])
    eps = np.zeros(3); ar = 0.0
    for xi in sb.G2:
        for eta in sb.G2:
            Bm, _, _, _, dJ, _, _ = sb._B_at(xyl, xi, eta)
            eps += (Bm @ ul) * dJ; ar += dJ
    eps /= ar
    # FD: local-e1 displacement difference between the two span planes / span length
    d0 = 0.5 * ((U[q[0], :3] + U[q[3], :3]) @ T[0])      # plane p (nodes 0,3)
    d1 = 0.5 * ((U[q[1], :3] + U[q[2], :3]) @ T[0])      # plane p+1 (nodes 1,2)
    Lx = 0.5 * (np.linalg.norm(nodes[q[1]] - nodes[q[0]]) + np.linalg.norm(nodes[q[2]] - nodes[q[3]]))
    epsfd = (d1 - d0) / Lx
    A11 = ABD_e[e][0, 0]
    print("  %6d  %+.4e   %+.4e   %7.3f   %+.4e  %+.4e"
          % (e, eps[0], epsfd, eps[0] / epsfd if epsfd else np.nan, Nf[e, 0], A11 * epsfd))

# ---------------- 3. moment absorbed by the drilling springs ----------------
print("\n=== 3. moment absorbed by the grounded drilling springs ===")
sb._KDR_SCALE = 1e-3
Mdrill = np.zeros(3); Edrill = 0.0
for e, q in enumerate(quads):
    T, xyl = sb._elem_frame(nodes, q)
    Ke, _ = sb.element_K_KG(xyl, ABD_e[e], Gs_e[e], np.zeros(3))
    kdr = sb._KDR_SCALE * np.mean(np.abs(np.diag(Ke)[2::5]))
    Kn = kdr * np.outer(T[2], T[2])
    for a in range(4):
        n = q[a]
        if sb._COPLANAR_ONLY and not flat[n]:
            continue
        th = U[n, 3:]
        c = Kn @ th
        Mdrill += c
        Edrill += 0.5 * th @ c
print("  sum of drilling-spring couples      = %s" % np.array2string(Mdrill, precision=4))
print("  moment IMBALANCE (Mapp + Mreact)    = %s" % np.array2string(imbM, precision=4))
print("  ratio (spring couples / imbalance)  = %s"
      % np.array2string(Mdrill / np.where(np.abs(imbM) > 1, imbM, np.nan), precision=4))
print("  drilling strain energy = %.4e J ; total ext work 0.5*f.u = %.4e J  -> %.1f%% of the work"
      % (Edrill, 0.5 * f @ u, 100 * Edrill / (0.5 * f @ u)))

# ---------------- cap-node rotation check (is theta the fiber rotation?) ----------------
print("\n=== cap-node rotation DOF vs the true section slope -dw/dx ===")
sl = slice(50 * Ntot, 51 * Ntot)
zz = nodes[sl][:, 2]
itop = 50 * Ntot + int(np.argmax(zz)); ibot = 50 * Ntot + int(np.argmin(zz))
ile = 50 * Ntot + int(np.argmax(nodes[sl][:, 1])); ite = 50 * Ntot + int(np.argmin(nodes[sl][:, 1]))
uz = np.array([U[p * Ntot:(p + 1) * Ntot, 2].mean() for p in range(NS)])
dwdx = np.gradient(uz, nodes[::Ntot, 0])[50]
print("  layer 50: -dw/dx = %+.4e   (expected fiber rotation magnitude)" % (-dwdx))
for lbl, n in [("top cap", itop), ("bot cap", ibot), ("LE", ile), ("TE", ite)]:
    print("   %-8s pos(y,z)=(%+7.3f,%+7.3f)  th=(%+.3e,%+.3e,%+.3e)  |th|/|dw/dx|=%.3f"
          % (lbl, nodes[n, 1], nodes[n, 2], U[n, 3], U[n, 4], U[n, 5],
             np.linalg.norm(U[n, 3:]) / abs(dwdx)))

# ---------------- 4. kdr sweep ----------------
print("\n=== 4. drilling-penalty sweep (the decisive test) ===")
for s in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 0.0]:
    try:
        run(s, "kdr=%.0e" % s)
    except Exception as ex:
        print("  kdr=%.0e FAILED: %s" % (s, ex))
sb._KDR_SCALE = 1e-3
