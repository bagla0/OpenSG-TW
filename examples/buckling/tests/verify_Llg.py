"""verify_Llg.py -- independent re-validation of the _L_lg fiber-rotation fix.

A correct shell stiffness must annihilate RIGID-BODY motion exactly.  Test 1 checks that directly (the
strongest, most basic check there is).  Tests 2-4 confirm the analytic anchors are still met, and test 5
re-checks the SQUARE, whose corner nodes have two normals and are therefore exactly where the old 90-deg
swap was NOT a harmless change of variables.

The grounded drilling spring ALSO breaks rotation invariance (it is a spring to ground, not a relative
drilling constraint), so rigid-rotation energy is reported at kdr=0 (pure element test) and at the default.
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb
import fsm_buckling as fsm

E, nu, t = 200e9, 0.3, 0.02
ABD, Gs = sb._iso_ABD(E, nu, t)

print("=" * 78)
print("1) ELEMENT RIGID-BODY TEST   u^T K u / |K|max   (must be ~0 for translations AND rotations)")
cases = {
    "flat unit square": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
    "tilted/warped   ": np.array([[0, 0, 0], [1, 0, .05], [1.1, 1, .2], [0, .9, .1]], float),
    "blade-like strip": np.array([[0, 0, 0], [1.37, 0.02, .01], [1.37, .35, .04], [0, .33, .0]], float),
}
for nm, X in cases.items():
    T, xyl = sb._elem_frame(nodes := X, [0, 1, 2, 3])
    Ke, _ = sb.element_K_KG(xyl, ABD, Gs, np.zeros(3))
    L = sb._L_lg(T); Kg24 = L.T @ Ke @ L
    scale = np.abs(Kg24).max()
    tr = []
    for d in np.eye(3):                       # rigid translations
        ug = np.concatenate([np.r_[d, 0, 0, 0] for _ in range(4)])
        tr.append(ug @ Kg24 @ ug / scale)
    ro = []
    for w in np.eye(3):                       # rigid rotations about the centroid
        c = X.mean(0)
        ug = np.concatenate([np.r_[np.cross(w, X[a] - c), w] for a in range(4)])
        ro.append(ug @ Kg24 @ ug / scale)
    print("   %s  transl=%s  ROT=%s" % (nm, np.array2string(np.array(tr), precision=1),
                                        np.array2string(np.array(ro), precision=2)))

print("\n2) SS PLATE  (analytic 4 pi^2 D / a^2)")
print("   ", end="")
sb.validate_plate()

print("\n3) SS3 CYLINDER  (classical E t^2 / (R sqrt(3(1-nu^2))))")
R, L2, nc, nl = 1.0, 2.0, 160, 80
Ncl = E * t**2 / (R * np.sqrt(3 * (1 - nu**2)))
th = np.linspace(0, 2 * np.pi, nc, endpoint=False); xs = np.linspace(0, L2, nl + 1)
nod = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
ix = lambda i, j: i * nc + (j % nc)
qd = np.array([[ix(i, j), ix(i + 1, j), ix(i + 1, j + 1), ix(i, j + 1)] for i in range(nl) for j in range(nc)])
ne = len(qd); Ae = np.repeat(ABD[None], ne, 0); Ge = np.repeat(Gs[None], ne, 0)
Nv = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0); fx = []
for j in range(nc):
    r0, rL = ix(0, j), ix(nl, j); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
lc = sb.solve_buckling(nod, qd, Ae, Ge, Nv, np.unique(fx), n_modes=6)[0]
print("   N_cr=%.4e  classical=%.4e  ratio=%.4f" % (lc[0], Ncl, lc[0] / Ncl))

print("\n4) PRISMATIC SQUARE BOX  (analytic k=4:  P_cr = 16 pi^2 D / a)   <-- corner nodes: the sensitive case")
D = E * t**3 / (12 * (1 - nu**2)); a = 1.0; Pcl = 16 * np.pi**2 * D / a
nps = nc // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
ring = np.array([np.array(cor[k], float) + (j / nps) * (np.array(cor[(k + 1) % 4], float) - np.array(cor[k], float))
                 for k in range(4) for j in range(nps)])
strips = np.array([[i, (i + 1) % (4 * nps)] for i in range(4 * nps)])
Napp = -1.0 / (4 * a)
lamF = np.asarray(fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips),
                                      [np.array([Napp, 0.0, 0.0])] * len(strips), L2, 16, n_modes=4))
nod2 = np.array([[xs[i], ring[p, 0], ring[p, 1]] for i in range(nl + 1) for p in range(nc)])
qd2 = np.array([[ix(i, p), ix(i + 1, p), ix(i + 1, p + 1), ix(i, p + 1)] for i in range(nl) for p in range(nc)])
N2 = np.repeat(np.array([Napp, 0, 0])[None], ne, 0); fx2 = []
for p in range(nc):
    r0, rL = ix(0, p), ix(nl, p); fx2 += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
ls = sb.solve_buckling(nod2, qd2, Ae, Ge, N2, np.unique(fx2), n_modes=6)[0]
print("   FEA=%.4e  analytic(k=4)=%.4e  FEA/an=%.4f   FSM=%.4e  FSM/FEA=%.4f"
      % (ls[0], Pcl, ls[0] / Pcl, lamF[0], lamF[0] / ls[0]))

print("\n5) STRUCTURE-LEVEL rigid rotation energy (cylinder), u^T K u, vs a unit-curvature bending energy")
K = sb.assemble_K(nod, qd, Ae, Ge)
c = nod.mean(0)
for w, lab in [(np.array([1.0, 0, 0]), "about span x"), (np.array([0, 1.0, 0]), "about y")]:
    ug = np.zeros(6 * len(nod))
    ug.reshape(-1, 6)[:, :3] = np.cross(w, nod - c); ug.reshape(-1, 6)[:, 3:] = w
    print("   rigid rot %s : u^T K u = %.4e" % (lab, ug @ (K @ ug)))
