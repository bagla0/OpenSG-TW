"""modes_export.py -- first two local-buckling mode shapes for JAX-Shell FEA and FSM-RM (OpenSG),
axially-compressed cylinder, isotropic + m45.  REFINED shell mesh (nc=240).  Outputs into the paper
folder: mode-shape PNGs, eigenvalue .dat, mode npz, ring yaml.  Pre-buckling N = uniform axial (the
shell membrane resultant that K_G consumes; = int sigma dz of the dehomogenized stress)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

OUT = os.path.abspath(os.path.join(BUCK, "..", "TW-paper", "fsm_buckling"))
R, t, L = 1.0, 0.02, 2.0
NC, NL, M = 240, 120, 16                                      # REFINED shell mesh
ABD_iso = fsm.iso_abd(200e9, 0.3, t); Gs_iso = (5. / 6.) * (200e9 / 2.6) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)


def cyl_fea_modes(ABD, Gs):
    th = np.linspace(0, 2 * np.pi, NC, endpoint=False); xs = np.linspace(0, L, NL + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(NL + 1) for j in range(NC)])
    idx = lambda i, j: i * NC + (j % NC)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(NL) for j in range(NC)])
    ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0)
    fx = []
    for j in range(NC):
        r0, rL = idx(0, j), idx(NL, j)
        fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
    loads, modes = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fx), n_modes=4)
    return loads, modes, nodes, th, xs


def fsm_mode_3d(Vm, th, nx=90):
    """reconstruct the FSM 3-D deflection over the cylinder from a modal amplitude Vm (M, nn, 4)."""
    xs = np.linspace(0, L, nx); U = np.zeros((nx, NC)); Dy = np.zeros((nx, NC)); Dz = np.zeros((nx, NC))
    for m in range(1, M + 1):
        k = m * np.pi / L
        U += np.outer(np.cos(k * xs), Vm[m - 1, :, 0])
        Dy += np.outer(np.sin(k * xs), Vm[m - 1, :, 1]); Dz += np.outer(np.sin(k * xs), Vm[m - 1, :, 2])
    ur = Dy * np.cos(th)[None, :] + Dz * np.sin(th)[None, :]
    return xs, U, Dy, Dz, ur


def panel(ax, X, Y, Z, C, title):
    Xc = np.column_stack([X, X[:, :1]]); Yc = np.column_stack([Y, Y[:, :1]])
    Zc = np.column_stack([Z, Z[:, :1]]); Cc = np.column_stack([C, C[:, :1]])
    v = np.max(np.abs(Cc)) + 1e-30
    ax.plot_surface(Xc, Yc, Zc, facecolors=plt.cm.RdBu(plt.Normalize(-v, v)(Cc)),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((2, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=-62); ax.set_title(title, fontsize=9)


dat = ["# cylinder axial local buckling eigenvalues (N_cr, N/m).  REFINED shell nc=%d." % NC,
       "# case   method       mode1        mode2"]
for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
    loads, modes, nodes, th, xs = cyl_fea_modes(ABD, Gs)
    ring, strips = fsm.cyl_ring(R, NC); N_s = [np.array([-1.0, 0, 0])] * len(strips)
    lam, V = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M, n_modes=4, return_vecs=True)
    fig = plt.figure(figsize=(9, 6.5))
    rr = np.hypot(nodes[:, 1], nodes[:, 2]) + 1e-30
    for c in range(2):                                       # FEA modes 1,2 (top row)
        ax = fig.add_subplot(2, 2, c + 1, projection="3d")
        u = modes[:, :3, c]; s = 0.32 / (np.abs(u).max() + 1e-30)
        X = (nodes[:, 0] + s * u[:, 0]).reshape(NL + 1, NC)
        Y = (nodes[:, 1] + s * u[:, 1]).reshape(NL + 1, NC); Z = (nodes[:, 2] + s * u[:, 2]).reshape(NL + 1, NC)
        Cr = ((u[:, 1] * nodes[:, 1] + u[:, 2] * nodes[:, 2]) / rr).reshape(NL + 1, NC)
        panel(ax, X, Y, Z, Cr, "JAX-Shell FEA  mode %d\n$N_{cr}=%.3e$" % (c + 1, loads[c]))
    xs2, U, Dy, Dz, ur = None, None, None, None, None
    for c in range(2):                                       # FSM modes 1,2 (bottom row)
        ax = fig.add_subplot(2, 2, c + 3, projection="3d")
        xs2, U, Dy, Dz, ur = fsm_mode_3d(V[:, :, :, c], th)
        s = 0.32 / (np.abs(ur).max() + 1e-30)
        X = (xs2[:, None] + s * U); Y = (R * np.cos(th)[None, :] + s * Dy); Z = (R * np.sin(th)[None, :] + s * Dz)
        panel(ax, X, Y, Z, ur, "FSM-RM (OpenSG)  mode %d\n$N_{cr}=%.3e$" % (c + 1, lam[c]))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "png", "cyl_%s_modes.png" % tag), dpi=130, bbox_inches="tight")
    np.savez(os.path.join(OUT, "data", "cyl_%s_modes.npz" % tag), fea_loads=loads, fea_modes=modes,
             fsm_lam=lam, nodes=nodes)
    dat.append("  %-5s  JAX-Shell    %.5e  %.5e" % (tag, loads[0], loads[1]))
    dat.append("  %-5s  FSM-RM       %.5e  %.5e" % (tag, lam[0], lam[1]))
    print("%s: FEA[%.4e,%.4e]  FSM[%.4e,%.4e] -> png/cyl_%s_modes.png" % (tag, loads[0], loads[1], lam[0], lam[1], tag))
open(os.path.join(OUT, "dat", "cyl_eigenvalues.dat"), "w").write("\n".join(dat) + "\n")
print("wrote dat/cyl_eigenvalues.dat")
