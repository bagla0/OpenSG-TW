"""h1c_rigid_rotation.py -- the moment sink is NOT the drilling spring (kdr=0 changes nothing).
K must annihilate rigid-body ROTATION.  Test it, and test the suspected cause: _L_lg maps the global
rotation vector to the local Mindlin fiber rotations as (beta_x,beta_y) = (e1.th, e2.th), but the
element's own kinematics (gamma_xz = w,x + beta_x, kappa_xx = beta_x,x  <=>  u = z*beta_x) require
    beta_x = +e2.th ,  beta_y = -e1.th
(displacement of a fiber point: th x (z e3) = z(th2 e1 - th1 e2)).
The two differ by a 90 deg rotation about e3 -> spurious transverse SHEAR under rigid rotation.
"""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

np.set_printoptions(linewidth=220)
NSE = bi.NSE; MPER = bb.MPER; Ntot = bb.Ntot
_L_orig = sb._L_lg


def _L_fixed(T):
    """beta_x = e2.theta, beta_y = -e1.theta  (Mindlin fiber rotation from the global rotation vector)."""
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


# ---------- A. single-element rigid-body test ----------
print("=== A. element rigid-body test: u = th x x , rot dof = th  -> energy must be 0 ===")
rng = np.random.default_rng(0)
ABD, Gs = sb._iso_ABD(3.0e10, 0.3, 0.05)
for trial, X in enumerate([
        np.array([[0., 0., 0.], [1.4, 0., 0.], [1.4, .3, .05], [0., .3, .05]]),          # blade-like strip
        np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),              # flat unit square
        rng.normal(size=(4, 3)) * 0.3 + np.array([[0, 0, 0], [1, 0, 0], [1, 1, .0], [0, 1, 0]])]):
    X[3] = X[0] + (X[2] - X[1])                                                          # keep it planar
    nodes4 = X; q = np.arange(4)
    T, xyl = sb._elem_frame(nodes4, q)
    Ke, _ = sb.element_K_KG(xyl, ABD, Gs, np.zeros(3))
    for lbl, Lf in [("code   _L_lg", _L_orig), ("fixed  _L_lg", _L_fixed)]:
        L = Lf(T)
        Kg = L.T @ Ke @ L
        # rigid translations
        et = []
        for d in range(3):
            ug = np.zeros(24); ug[d::6] = 1.0
            et.append(ug @ Kg @ ug)
        # rigid rotations about the 3 global axes
        er = []
        for d in range(3):
            th = np.zeros(3); th[d] = 1.0
            ug = np.zeros(24)
            for a in range(4):
                ug[6 * a:6 * a + 3] = np.cross(th, nodes4[a]); ug[6 * a + 3:6 * a + 6] = th
            er.append(ug @ Kg @ ug)
        scale = np.abs(Kg).max()
        print("  trial %d  %s : rigid-TRANS energy %s   rigid-ROT energy/|K|max = %s"
              % (trial, lbl, np.array2string(np.array(et), precision=2),
                 np.array2string(np.array(er) / scale, precision=4)))

# ---------- B. full blade with the fixed transform ----------
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]; ndof = 6 * len(nodes)
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
fx = f.reshape(-1, 6)[:, :3]
Fapp = fx.sum(0); Mapp = np.cross(nodes, fx).sum(0)
fixed = np.asarray(root, int); free = np.setdiff1d(np.arange(ndof), fixed)


def moment_from_N(Ne, p, i):
    P = bl["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


print("\n=== B. full blade static, code vs fixed _L_lg ===")
for lbl, Lf in [("code ", _L_orig), ("FIXED", _L_fixed)]:
    sb._L_lg = Lf
    t0 = time.time()
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
    Ku = K @ u
    resr = np.linalg.norm((Ku - f)[free]) / np.linalg.norm(f[free])
    Rn = (Ku - f).reshape(-1, 6); rootn = np.arange(Ntot)
    Fr = Rn[rootn, :3].sum(0)
    Mr = np.cross(nodes[rootn], Rn[rootn, :3]).sum(0) + Rn[rootn, 3:].sum(0)
    itip = int(np.argmax(nodes[:, 0]))
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    rats = [moment_from_N(Nf, min(i * MPER, NS - 2), i) / FF[i][4] for i in [5, 15, 25, 35, 45]]
    print("  %s res=%.2e  tip_uz=%.4f  F_react/-F_app=%.5f  M_react_y/-M_app_y=%+.4f"
          % (lbl, resr, u[6 * itip + 2], -Fr[2] / Fapp[2], -Mr[1] / Mapp[1]))
    print("        M_react=%s   M_app=%s" % (np.array2string(Mr, precision=4), np.array2string(Mapp, precision=4)))
    print("        M_FE/FF at sta [5,15,25,35,45] = %s   (%.0fs)"
          % (np.array2string(np.array(rats), precision=4), time.time() - t0))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1c_Nf_%s.npy" % lbl.strip()), Nf)

# ---------- C. do the published validations still hold with the fix? ----------
print("\n=== C. re-validate plate + cylinder with the FIXED transform ===")
for lbl, Lf in [("code ", _L_orig), ("FIXED", _L_fixed)]:
    sb._L_lg = Lf
    fe, an = sb.validate_plate(nx=24)
    print("  %s plate  ratio = %.4f" % (lbl, fe / an))
    fe, an = sb.validate_cylinder(mesh=(120, 60), bc="SS", verbose=False)
    print("  %s cyl SS ratio = %.4f" % (lbl, fe / an))
sb._L_lg = _L_orig
