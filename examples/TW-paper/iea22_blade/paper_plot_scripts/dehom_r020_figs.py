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


def make(fn, out, kind):
    coords = np.loadtxt(os.path.join(D2, fn))[:, :2]
    # RIGOROUS sampling -- no lateral penalty, no artificial change.  The cap path is built FROM the
    # VABS gauss points, so every point is exact in BOTH VABS and RM; drop any point that does not
    # coincide with a gauss point to 1e-5 (per-point, robust).  The circumferential path lies on the
    # OML surface where plain nearest is exact enough.
    dist, gi = tree.query(coords)
    if kind == "cap":
        ok = dist <= 1e-5
        if not ok.all():
            print("  dropped %d/%d cap points with no exact VABS gauss (dist>1e-5)"
                  % (int((~ok).sum()), len(coords)))
        coords, gi = coords[ok], gi[ok]
        npl, xs = 4, (coords[:, 1] - coords[0, 1]) / (coords[-1, 1] - coords[0, 1])   # finer SG; y3 0..1
    else:
        npl, xs = 2, (coords[:, 0] - coords[0, 0]) / (coords[-1, 0] - coords[0, 0])   # y2 0..1
    S = np.asarray(dehom_rm.stress_at_points(bundle, coords, beam_force_vabs=FF,
                                             frame="material", n_per_layer=npl)["stress"])
    V = sm_s[gi]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for k, c in enumerate(COMP):
        a = ax.flat[k]; oop = c in ("S33", "S13", "S23")
        a.plot(xs, S[:, k] / 1e6, "r--o", ms=3.5, label="RM shell (two-step)")
        a.plot(xs, V[:, k] / 1e6, "g-^", ms=4, label="VABS (.SM)")
        a.set_title(r"$\sigma_{%s}$" % c[1:], color=("darkred" if oop else "black"), fontsize=10)
        a.set_xlabel("non-dim parameter"); a.set_ylabel("%s (MPa)" % c); a.set_xlim(0, 1)
        a.grid(True, ls=":", alpha=0.6)
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)  # outside
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    ip = np.linalg.norm(S[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / np.linalg.norm(V[:, [0, 1, 5]]) * 100
    print("wrote", os.path.basename(out), "| %d pts | in-plane ||.|| = %.1f%%" % (len(coords), ip))


make("solid.lp_sparcap_right_thickness_r020.coords",
     os.path.join(FIG, "dehom_r020_capright.png"), kind="cap")
make("solid.circumferential_r020.coords",
     os.path.join(FIG, "dehom_r020_circumferential.png"), kind="circ")
