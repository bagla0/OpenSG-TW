"""dehom_st15_figs.py -- RM shell two-step dehom vs VABS .SM on the VALID paths
(cap-centre through-thickness + circumferential around the section), 6 stress
components each.  -> figures/dehom_st15_capcentre.png, dehom_st15_circumferential.png
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points

DEH = os.path.join(HERE, "..", "..", "..", "examples", "data", "dehom_st15")
SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


sm_xy, sm_s = load_sm(os.path.join(DEH, "bar_urc-15-t-0.in.SM"))
tree = cKDTree(sm_xy)
bundle = solve_tw_from_yaml(SHELL15, frac=0.0)


def make(fn, xlabel, title, out, around=False):
    coords = np.loadtxt(os.path.join(DEH, fn))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    xs = z * (1.0 if around else 1e3)
    S = np.asarray(stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")["stress"])
    V = sm_s[tree.query(coords)[1]]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for k, c in enumerate(COMP):
        a = ax.flat[k]; oop = c in ("S33", "S13", "S23")
        a.plot(xs, S[:, k] / 1e6, "r--o", ms=3.5, label="RM shell (two-step)")
        a.plot(xs, V[:, k] / 1e6, "g-^", ms=4, label="VABS (.SM)")
        a.set_title(r"$\sigma_{%s}$" % c[1:] + ("  [out-of-plane]" if oop else ""),
                    color=("darkred" if oop else "black"), fontsize=10)
        a.set_xlabel(xlabel); a.set_ylabel("%s (MPa)" % c)
        a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    ip = np.linalg.norm(S[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / np.linalg.norm(V[:, [0, 1, 5]]) * 100
    print("wrote", os.path.basename(out), "| in-plane ||.|| = %.1f%%" % ip)


make("solid.lp_sparcap_center_thickness_015.coords",
     "through-thickness (mm, OML$\\to$IML)",
     "Station-15 LP spar-cap CENTRE through-thickness: RM shell vs VABS",
     os.path.join(FIG, "dehom_st15_capcentre.png"))
make("solid.circumferential_015.coords", "circumferential arc length (m)",
     "Station-15 circumferential path (around the section): RM shell vs VABS",
     os.path.join(FIG, "dehom_st15_circumferential.png"), around=True)
