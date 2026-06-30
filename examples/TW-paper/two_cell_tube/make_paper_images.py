"""Cross-section figures for the paper. The 2-cell is a SINGLE solid-coloured
cross-section (the wall material filled in one colour, no mesh) with the 1-D
shell mid-wall reference overlaid as a dotted line; the 4-cell keeps the meshed
rendering (dormant asset). y2/y3 axes only, no title/legend. Thick geometry
(R/h=3.1) so the dotted mid-line sits clearly inside the wall.
  -> figures/xsec_2cell.png, xsec_4cell.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import yaml

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"
FIG = os.path.join(os.path.dirname(DATA), "figures")
os.makedirs(FIG, exist_ok=True)
FILL = "#9ecae1"   # single fill colour for the wall material
REF = "red"        # colour of the dotted shell mid-wall reference line


def row(r, n=2):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)][:n]


def load(solidf, shellf):
    sd = yaml.safe_load(open(os.path.join(DATA, solidf)))
    sn = np.array([row(r) for r in sd["nodes"]])
    se = np.array([[int(round(float(v))) - 1 for v in (str(r[0]).split() if len(r) == 1 else r)][:3]
                   for r in sd["elements"]])
    hd = yaml.safe_load(open(os.path.join(DATA, shellf)))
    hn = np.array([row(r) for r in hd["nodes"]])
    he = [[int(v) - 1 for v in r[0].split()] for r in hd["elements"]]
    return sn, se, hn, he


def figure(name, solidf, shellf, filled):
    sn, se, hn, he = load(solidf, shellf)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    if filled:                                                               # solid wall, one colour
        ax.add_collection(PolyCollection(sn[se], facecolors=FILL, edgecolors="none"))
    else:                                                                    # meshed
        ax.triplot(sn[:, 0], sn[:, 1], se, color="0.6", lw=0.35)
    for (a, b) in he:                                                         # shell mid-wall: DOTTED (red)
        ax.plot([hn[a, 0], hn[b, 0]], [hn[a, 1], hn[b, 1]], ls=":", color=REF, lw=1.8)
    pad = 0.06 * (sn[:, 0].max() - sn[:, 0].min())
    ax.set_xlim(sn[:, 0].min() - pad, sn[:, 0].max() + pad)
    ax.set_ylim(sn[:, 1].min() - pad, sn[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y_2$ (m)", fontsize=15)
    ax.set_ylabel(r"$y_3$ (m)", fontsize=15)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    out = os.path.join(FIG, "xsec_%s.png" % name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


figure("2cell_thin", "solid_tube2cell_thin.yaml", "tube2cell_thin.yaml", filled=True)
figure("2cell_thick", "solid_tube2cell_thick.yaml", "tube2cell_thick.yaml", filled=True)
figure("4cell", "solid_tube4cell_aniso_thick.yaml", "tube4cell_aniso_thick.yaml", filled=False)
