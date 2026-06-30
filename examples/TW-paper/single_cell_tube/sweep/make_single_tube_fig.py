"""Single-cell [-45] tube cross-section: FEniCS 2-D solid wall (single fill colour)
with the centric (mid-wall) reference drawn as a red dotted circle. y2/y3 axes.
  -> figs/xsec_tube_single.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import yaml

D = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\sweep\data"
FIG = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\sweep\figs"
CASE = "solid_rh03.yaml"   # R/h = 3 : clearly thick wall


def rows(r, n=2):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)][:n]


sd = yaml.safe_load(open(os.path.join(D, CASE)))
sn = np.array([rows(r) for r in sd["nodes"]])
se = np.array([[int(round(float(v))) - 1 for v in (str(r[0]).split() if len(r) == 1 else r)][:3]
               for r in sd["elements"]])
rad = np.sqrt(sn[:, 0]**2 + sn[:, 1]**2)
Rmid = 0.5 * (rad.min() + rad.max())     # mid-wall radius

fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.add_collection(PolyCollection(sn[se], facecolors="#9ecae1", edgecolors="none"))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(Rmid * np.cos(th), Rmid * np.sin(th), ls=":", color="red", lw=1.8)   # centric reference
pad = 0.06 * (sn[:, 0].max() - sn[:, 0].min())
ax.set_xlim(sn[:, 0].min() - pad, sn[:, 0].max() + pad)
ax.set_ylim(sn[:, 1].min() - pad, sn[:, 1].max() + pad)
ax.set_aspect("equal")
ax.set_xlabel(r"$y_2$ (m)", fontsize=15)
ax.set_ylabel(r"$y_3$ (m)", fontsize=15)
ax.tick_params(labelsize=12)
fig.tight_layout()
out = os.path.join(FIG, "xsec_tube_single.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print("Rmid=%.4f  inner=%.4f outer=%.4f" % (Rmid, rad.min(), rad.max()))
print("wrote", out)
