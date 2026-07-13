"""dehom_r020_figs.py -- RM shell two-step dehom vs VABS .SM for IEA r=0.2, on the mid LP
spar-cap through-thickness path + the upper-surface (LP) circumferential path.  Same FF as
st15.  Reads the PreVABS/VABS files from examples/data/2d_yaml/ and the 1-D shell there.
Run AFTER `vabs iea_r020.sg 10` has produced iea_r020.sg.SM.
  -> figures/dehom_r020_capcentre.png, dehom_r020_circumferential.png
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
# RM side: the OML-referenced shell (0.5 mm inset ~ OML, same frame as the .sg).  The freshly
# converted iea_r020_shell.yaml is mid-surface (OpenSG_io fraction=0.5, 44 mm inset) -> wrong depth.
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
SM = os.path.join(D2, "iea_r020.sg.SM")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    # VABS 4.1 .SM: 2 header lines ("Stresses ... load case #", "1"), then
    # y2 y3 s11 s12 s13 s22 s23 s33 (material frame) -> reorder to [S11,S22,S33,S23,S13,S12]
    d = np.loadtxt(path, skiprows=2)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


if not os.path.exists(SM):
    sys.exit("VABS .SM not found yet: %s\n  run:  cd %s ; vabs iea_r020.sg 10" % (SM, D2))

sm_xy, sm_s = load_sm(SM)
tree = cKDTree(sm_xy)
bundle = dehom_rm.build_rm_bundle(SHELL, ref="oml")   # RM ring (C0, MITC-g23)


def make(fn, xlabel, title, out, around=False):
    coords = np.loadtxt(os.path.join(D2, fn))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    xs = z * (1.0 if around else 1e3)
    S = np.asarray(dehom_rm.stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")["stress"])
    V = sm_s[tree.query(coords)[1]]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for k, c in enumerate(COMP):
        a = ax.flat[k]; oop = c in ("S33", "S13", "S23")
        a.plot(xs, S[:, k] / 1e6, "r--o", ms=3.5, label="RM shell (two-step)")
        a.plot(xs, V[:, k] / 1e6, "g-^", ms=4, label="VABS (.SM)")
        a.set_title(r"$\sigma_{%s}$" % c[1:],
                    color=("darkred" if oop else "black"), fontsize=10)
        a.set_xlabel(xlabel); a.set_ylabel("%s (MPa)" % c)
        a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    ip = np.linalg.norm(S[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / np.linalg.norm(V[:, [0, 1, 5]]) * 100
    print("wrote", os.path.basename(out), "| in-plane ||.|| = %.1f%%" % ip)


make("solid.lp_sparcap_left_thickness_r020.coords",
     "through-thickness (mm, OML$\\to$IML)",
     "IEA r=0.2 LP spar-cap (LEFT) through-thickness: RM shell vs VABS",
     os.path.join(FIG, "dehom_r020_capleft.png"))
make("solid.circumferential_r020.coords", "upper-surface arc length (m, LE$\\to$TE)",
     "IEA r=0.2 upper-surface (LP) circumferential: RM shell vs VABS",
     os.path.join(FIG, "dehom_r020_circumferential.png"), around=True)
