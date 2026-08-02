"""make_curves.py -- the Fig.-5-style HISTORY CURVES for Example 5:
center deflection w(t) and the top-surface mid-point stress
sigma_x(a/2, b/2, h/2, t), at Nayak's time step (dt = 50 us), for
    OpenSG-RM   (per-time-step dehomogenization from the S4 resultants)
    Abaqus 3-D  (the solid benchmark)
for both pulses (step, blast).  Everything comes from the EXISTING job
.dat files in Abaqus_results/ -- no new Abaqus runs.

THE PROCESS (what generates each curve):
  w(t):    the *NODE PRINT of the center node, printed at every 50-us
           increment in both jobs, read straight from the .dat (the solid
           value sign-flipped to the +z convention); plotted as w/0.0254 m
           (Nayak's scaling).
  sx(t) OpenSG-RM: at EVERY increment, the printed PATCHC SF/SM are
           patch-fitted about (a/2, b/2) -> E6 = S6 R6 and its gradients
           -> msgrm_strain_at_depth at z = h/2 with the double-sine load
           ladder at that instant -> sigma_11 / q0.  This is the
           dehomogenization applied 400 times, once per time step.
  sx(t) solid: the COLC top-layer element-centroid S11 at every 10th
           increment (the deck's print cadence), global frame, flipped,
           / q0.  NOTE its z is the element centroid (h/2 - 0.95 mm, one
           half-ply below the surface); the RM curve is also evaluated
           there for the like-for-like comparison, and at the exact
           surface z = h/2 in the .dat.

Outputs -> curves/ :
  w_center_<pulse>.png/.dat      sx_top_center_<pulse>.png/.dat

Run:  python examples/opensg-rm_dynamic/ex5/make_curves.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse the validated machinery of the recovery post-processor
from recover_dyn import (step_times, node_history, patch_series, fit_patch,
                         load_ladder, solid_columns, r, S6, thick, H, XSTA,
                         Q0, CBLAST, read_elprint_tables, P)
from opensg_jax.fe_jax.msg_rm_plate import msgrm_strain_at_depth

OUT = os.path.join(HERE, "curves")
os.makedirs(OUT, exist_ok=True)
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2)
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4,
              mfc="none", mew=1.1)
STY_FS = dict(ls="-.", marker="^", color="#2ca02c", lw=1.3, ms=5,
              mfc="none", mew=1.1)
TPLY = thick[0]
Z_SURF = H / 2 - 1e-9                  # the exact top surface
Z_CTR = H / 2 - TPLY / 2               # the solid top-element centroid


def one_plot(fname, curves, xlabel, ylabel, note=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for x, y, sty, lab in curves:
        ax.plot(x, y, label=lab, **sty)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                color="0.35")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(kind):
    frm = os.path.join(HERE, "Abaqus_results", "sandwich_RM_%s.dat" % kind)
    fso = os.path.join(HERE, "Abaqus_results",
                       "sandwich_SOLID_%s.dat" % kind)
    # ---- w(t) ----------------------------------------------------------
    t_rm = step_times(frm)
    uc = node_history(frm, "NCEN")
    t_rm = t_rm[len(t_rm) - len(uc):]
    w_rm = uc[:, 2] / 0.0254
    t_so_all = step_times(fso)
    uc3 = node_history(fso, "NCEN3D")
    t_so = t_so_all[len(t_so_all) - len(uc3):]
    w_so = -uc3[:, 2] / 0.0254
    curves = [(1e3 * np.concatenate([[0], t_so]),
               np.concatenate([[0], w_so]),
               dict(STY_SOL, markevery=(0, 32)), "Abaqus 3-D solid"),
              (1e3 * np.concatenate([[0], t_rm]),
               np.concatenate([[0], w_rm]),
               dict(STY_RM, markevery=(10, 30)), "OpenSG-RM")]
    # the conventional Abaqus FSDT (Whitney-1973-type composite S4)
    ffs = os.path.join(HERE, "Abaqus_results",
                       "sandwich_FSDT_%s.dat" % kind)
    w_fs = None
    if os.path.isfile(ffs):
        t_fs = step_times(ffs)
        uf = node_history(ffs, "NCEN")
        t_fs = t_fs[len(t_fs) - len(uf):]
        w_fs = uf[:, 2] / 0.0254
        curves.append((1e3 * np.concatenate([[0], t_fs]),
                       np.concatenate([[0], w_fs]),
                       dict(STY_FS, markevery=(20, 30)),
                       "Abaqus FSDT\n(composite S4)"))
    one_plot("w_center_%s.png" % kind, curves, "time [ms]",
             r"center deflection $w/0.0254$ m")
    # aligned .dat (everything interpolated onto the 50-us grid)
    w_so_i = np.interp(t_rm, t_so, w_so)
    cols = [1e3 * t_rm, w_rm, w_so_i]
    hdr = ("t[ms]  w/0.0254_OpenSG-RM  w/0.0254_Abaqus3D"
           "  w/0.0254_AbaqusFSDT (%s pulse, dt = 50 us)" % kind)
    if w_fs is not None:
        cols.append(np.interp(t_rm, t_fs, w_fs))
    np.savetxt(os.path.join(OUT, "w_center_%s.dat" % kind),
               np.column_stack(cols), header=hdr)
    # ---- sigma_x at the top mid point, PER TIME STEP -------------------
    tables = read_elprint_tables(frm)
    sf, xy = patch_series(tables, "PATCHC")
    fits = fit_patch(sf, xy, XSTA["C"])
    n = min(len(t_rm), fits.shape[0])

    def ft(t):
        return 1.0 if kind == "step" else np.exp(-CBLAST * t)

    z6 = np.zeros(6)
    sx_s = np.empty(n)                 # at the exact surface z = h/2
    sx_c = np.empty(n)                 # at the solid centroid depth
    for k in range(n):
        F = fits[k]
        E6 = S6 @ F[[0, 1, 2, 5, 6, 7], 0]
        dE1 = S6 @ F[[0, 1, 2, 5, 6, 7], 1]
        dE2 = S6 @ F[[0, 1, 2, 5, 6, 7], 2]
        qt6 = load_ladder(*XSTA["C"], Q0 * ft(t_rm[k]))
        for z, dest in ((Z_SURF, sx_s), (Z_CTR, sx_c)):
            dest[k] = msgrm_strain_at_depth(
                r, z, E6, dE1, dE2, -P * P * E6, z6, -P * P * E6,
                qt6=qt6)[1][0]
    sx_s /= Q0
    sx_c /= Q0
    # solid: COLC top-layer centroid S11 at the dump instants
    sol = solid_columns(fso)
    zc_s, gs = sol["COLC"]
    td = list(t_so_all[9::10])
    if gs.shape[0] == len(td) + 1:
        td.append(t_so_all[-1])
    td = np.array(td[:gs.shape[0]])
    sx_solid = gs[:, -1, 0] / Q0
    one_plot("sx_top_center_%s.png" % kind,
             [(1e3 * td, sx_solid, STY_SOL,
               "Abaqus 3-D solid\n(element centroid,"
               " $z = h/2 - t_{ply}/2$)"),
              (1e3 * t_rm[:n], sx_c, dict(STY_RM, markevery=(10, 30)),
               "OpenSG-RM dehom\n(same depth, every 50 $\\mu$s)")],
             "time [ms]",
             r"$\bar\sigma_x = \sigma_x(a/2,\,b/2,\,z)/q_0$",
             note="the solid prints S only every 10th increment; the RM"
                  " curve exists at every increment")
    np.savetxt(os.path.join(OUT, "sx_top_center_%s.dat" % kind),
               np.column_stack([1e3 * t_rm[:n], sx_s, sx_c,
                                np.interp(t_rm[:n], td, sx_solid)]),
               header="t[ms]  sx/q0_RM_surface(z=h/2)  "
                      "sx/q0_RM_centroid(z=h/2-tply/2)  "
                      "sx/q0_Abaqus3D_centroid(interp between dumps) "
                      "(%s pulse)" % kind)
    ipk = int(np.argmax(np.abs(w_rm)))
    print("%6s: peak w/0.0254 RM %.4f | solid %.4f;  peak sx/q0 RM(ctr)"
          " %.3f | solid %.3f"
          % (kind, w_rm[ipk], w_so.max() if kind == "step"
             else w_so[np.argmax(np.abs(w_so))],
             sx_c[np.argmax(np.abs(sx_c))],
             sx_solid[np.argmax(np.abs(sx_solid))]))


if __name__ == "__main__":
    for kind in ("step", "blast"):
        run(kind)
    print("curves written to curves/")
