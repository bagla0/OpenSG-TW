"""render_cyl_modes.py -- cylinder buckling mode shapes: JAX-FEA vs RM-OpenSG, side by side.
Also the dehom membrane-N distribution around the ring.  Writes cyl_modes.png, cyl_N.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCK = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BUCK, "data"); FIG = os.path.join(BUCK, "fig")
d = np.load(os.path.join(DATA, "cyl_bench.npz"))
nodes = d["nodes"]; mf = d["modes_fea"]; mr = d["modes_rm"]
lf = d["loads_fea"]
lr = d["loads_rm"] * abs(d["Nring"][:, 0].mean())    # RM eigenvalue -> physical N_cr (x |N| imposed)
NC, NL = 160, 80
Ncl = 4.8418e7
rr = np.hypot(nodes[:, 1], nodes[:, 2]) + 1e-30


def radial(modes, m):
    u = modes[:, :3, m]
    return (u[:, 1] * nodes[:, 1] + u[:, 2] * nodes[:, 2]) / rr


NMODE = 4
fig = plt.figure(figsize=(10, 2.7 * NMODE))
for m in range(NMODE):
    for c, (modes, loads, name) in enumerate([(mf, lf, "JAX-FEA"), (mr, lr, "RM-OpenSG")]):
        ax = fig.add_subplot(NMODE, 2, 2 * m + c + 1, projection="3d")
        ur = radial(modes, m)
        s = 0.35 / (np.max(np.abs(ur)) + 1e-30)
        u = modes[:, :3, m] * s
        X = (nodes[:, 0] + u[:, 0]).reshape(NL + 1, NC)
        Y = (nodes[:, 1] + u[:, 1]).reshape(NL + 1, NC)
        Z = (nodes[:, 2] + u[:, 2]).reshape(NL + 1, NC)
        C = ur.reshape(NL + 1, NC)
        Xc = np.column_stack([X, X[:, :1]]); Yc = np.column_stack([Y, Y[:, :1]])
        Zc = np.column_stack([Z, Z[:, :1]]); Cc = np.column_stack([C, C[:, :1]])
        norm = plt.Normalize(-np.max(np.abs(Cc)), np.max(np.abs(Cc)))
        ax.plot_surface(Xc, Yc, Zc, facecolors=plt.cm.RdBu(norm(Cc)),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_title("%s  mode %d\n$N=%.3e$ N/m ($%.3f\\,N_{cl}$)" % (name, m + 1, loads[m], loads[m] / Ncl),
                     fontsize=9)
        ax.set_box_aspect((2, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=18, azim=-60)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cyl_modes.png"), dpi=130, bbox_inches="tight")
print("wrote cyl_modes.png")

# membrane N around the ring (dehom)
ang = d["ang"]; Nring = d["Nring"]
o = np.argsort(ang)
fig2, ax2 = plt.subplots(figsize=(7, 3.2))
ax2.plot(np.degrees(ang[o]), Nring[o, 0], "-", label="$N_{xx}$ (axial)")
ax2.plot(np.degrees(ang[o]), Nring[o, 1], "--", label="$N_{yy}$ (hoop)")
ax2.plot(np.degrees(ang[o]), Nring[o, 2], ":", label="$N_{xy}$ (shear)")
ax2.set_xlabel("circumferential angle (deg)"); ax2.set_ylabel("membrane $N$ (N/m)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(FIG, "cyl_N.png"), dpi=130, bbox_inches="tight")
print("wrote cyl_N.png")
