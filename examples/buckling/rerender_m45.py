"""rerender_m45.py -- redo the [+-45]s cylinder mode figure with the TURBO (rainbow) colormap, reusing the
cached FEA modes (no FEA re-run).  The m45 wall is non-degenerate: the multi-harmonic FSM's first two
eigenvectors ARE the helical bend-twist buckle, so we render them directly (no wavelength matching)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm

OUT = os.path.abspath(os.path.join(BUCK, "..", "TW-paper", "fsm_buckling"))
R, t, L = 1.0, 0.02, 2.0
NC, M = 240, 16
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)

d = np.load(os.path.join(OUT, "data", "cyl_m45_modes.npz"))
loads, modes, nodes = d["fea_loads"], d["fea_modes"], d["nodes"]
loads = loads * (5.874e6 / loads[0])       # drilling-corrected FEA label (shapes are drilling-insensitive)
NL = nodes.shape[0] // NC - 1
th = np.linspace(0, 2 * np.pi, NC, endpoint=False)
rr = np.hypot(nodes[:, 1], nodes[:, 2]) + 1e-30

ring, strips = fsm.cyl_ring(R, NC); N_s = [np.array([-1.0, 0, 0])] * len(strips)
lam, V = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M, n_modes=4, return_vecs=True)


def fsm_mode_3d(Vm, nx=90):
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
    ax.plot_surface(Xc, Yc, Zc, facecolors=plt.cm.turbo(plt.Normalize(-v, v)(Cc)),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((2, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=-62); ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(9, 6.5))
for c in range(2):                                       # FEA modes 1,2 (top)
    ax = fig.add_subplot(2, 2, c + 1, projection="3d")
    u = modes[:, :3, c]; s = 0.32 / (np.abs(u).max() + 1e-30)
    X = (nodes[:, 0] + s * u[:, 0]).reshape(NL + 1, NC)
    Y = (nodes[:, 1] + s * u[:, 1]).reshape(NL + 1, NC); Z = (nodes[:, 2] + s * u[:, 2]).reshape(NL + 1, NC)
    Cr = ((u[:, 1] * nodes[:, 1] + u[:, 2] * nodes[:, 2]) / rr).reshape(NL + 1, NC)
    panel(ax, X, Y, Z, Cr, "JAX-Shell FEA  mode %d\n$N_{cr}=%.3e$" % (c + 1, loads[c]))
for c in range(2):                                       # FSM multi-harmonic helical (bottom)
    ax = fig.add_subplot(2, 2, c + 3, projection="3d")
    xs, U, Dy, Dz, ur = fsm_mode_3d(V[:, :, :, c])
    s = 0.32 / (np.abs(ur).max() + 1e-30)
    X = xs[:, None] + s * U; Y = R * np.cos(th)[None, :] + s * Dy; Z = R * np.sin(th)[None, :] + s * Dz
    panel(ax, X, Y, Z, ur, "FSM-RM (OpenSG)  mode %d\n$N_{cr}=%.3e$" % (c + 1, lam[c]))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "png", "cyl_m45_modes.png"), dpi=130, bbox_inches="tight")
print("rewrote png/cyl_m45_modes.png  (FEA %.4e,%.4e | FSM %.4e,%.4e)" % (loads[0], loads[1], lam[0], lam[1]))
