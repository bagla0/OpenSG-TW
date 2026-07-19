"""rerender_iso.py -- redo ONLY the isotropic cylinder mode figure.  The iso cylinder buckling is
Koiter-degenerate: a whole family of (m,n) modes buckle at essentially the SAME load, so even the two
FEA modes differ in wavenumber and the FSM's raw first eigenvector need not coincide with them.  For a
faithful figure we reconstruct each FSM panel (single-harmonic) at the axial half-wavelength of the FEA
panel above it, giving matching short-wave diamonds at the same critical load.  The reported FSM
buckling LOAD in the tables is the multi-harmonic converged value; FEA modes are loaded from the cached
npz (no FEA re-run)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm

OUT = os.path.abspath(os.path.join(BUCK, "..", "TW-paper", "fsm_buckling"))
R, t, L = 1.0, 0.02, 2.0
ABD = fsm.iso_abd(200e9, 0.3, t)

d = np.load(os.path.join(OUT, "data", "cyl_iso_modes.npz"))
loads, modes, nodes = d["fea_loads"], d["fea_modes"], d["nodes"]
loads = loads * (4.838e7 / loads[0])       # label with the drilling-corrected FEA (nc-converged 0.999*classical);
                                           # mode SHAPES are drilling-insensitive, so the cached nc=240 modes stand
NC = 240; NL = nodes.shape[0] // NC - 1
th = np.linspace(0, 2 * np.pi, NC, endpoint=False)
rr = np.hypot(nodes[:, 1], nodes[:, 2]) + 1e-30
ring, strips = fsm.cyl_ring(R, NC)
ABD_s = [ABD] * len(strips); N_s = [np.array([-1.0, 0, 0])] * len(strips)


def fea_mode_shape(c):
    ur = ((modes[:, 1, c] * nodes[:, 1] + modes[:, 2, c] * nodes[:, 2]) / rr).reshape(NL + 1, NC)
    col = ur[:, int(np.argmax(np.abs(ur).max(0)))]
    m_ax = max(2, int(np.sum(np.abs(np.diff(np.sign(col))) > 0)))    # axial half-waves
    n_circ = int(np.argmax(np.abs(np.fft.rfft(ur[NL // 2, :]))[1:])) + 1   # circumferential order
    return m_ax, n_circ


def fsm_ncirc(Vk):                                                  # circumferential order of an FSM eigenvector
    p = Vk[:, 1] * np.cos(th) + Vk[:, 2] * np.sin(th)
    return int(np.argmax(np.abs(np.fft.rfft(p))[1:])) + 1


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
xs = np.linspace(0, L, 160)
a_scan = np.linspace(0.30, 0.95, 40)                     # moderate wavelengths -> FSM first mode is a clean diamond
for c in range(2):                                       # FSM diamond of matching circumferential order (degenerate family)
    m_ax, n_fea = fea_mode_shape(c)
    ncs = np.array([fsm_ncirc(fsm.solve_fsm(ring, strips, ABD_s, N_s, a)[1][:, :, 0]) for a in a_scan])
    a = a_scan[int(np.argmin(np.abs(ncs - n_fea)))]; k = np.pi / a
    lam_a, V = fsm.solve_fsm(ring, strips, ABD_s, N_s, a)
    ax = fig.add_subplot(2, 2, c + 3, projection="3d")
    U = np.outer(np.cos(k * xs), V[:, 0, 0])
    Dy = np.outer(np.sin(k * xs), V[:, 1, 0]); Dz = np.outer(np.sin(k * xs), V[:, 2, 0])
    ur = Dy * np.cos(th)[None, :] + Dz * np.sin(th)[None, :]
    s = 0.32 / (np.abs(ur).max() + 1e-30)
    X = xs[:, None] + s * U; Y = R * np.cos(th)[None, :] + s * Dy; Z = R * np.sin(th)[None, :] + s * Dz
    panel(ax, X, Y, Z, ur, "FSM-RM (OpenSG)  mode %d\n$N_{cr}=%.3e$" % (c + 1, lam_a[0]))
    print("mode%d  FEA (m=%d,n=%d) -> FSM a=%.3f n=%d  N_cr=%.4e" % (
        c + 1, m_ax, n_fea, a, fsm_ncirc(V[:, :, 0]), lam_a[0]))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "png", "cyl_iso_modes.png"), dpi=130, bbox_inches="tight")
print("rewrote png/cyl_iso_modes.png")
