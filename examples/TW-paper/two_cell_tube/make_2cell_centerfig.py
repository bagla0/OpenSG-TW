"""2-cell tube figure with the red dotted CENTRIC reference drawn through the true
mid-line of each segment: the ring mid-wall (radius R) and the web CENTRE (measured
from the solid, since PreVABS offsets the web to one side of its baseline).
  -> figures/xsec_2cell_thick.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import yaml

D = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"
FIG = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\figures"
R = 0.05


def rows(r, n=2):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)][:n]


sd = yaml.safe_load(open(os.path.join(D, "solid_tube2cell_thick.yaml")))
sn = np.array([rows(r) for r in sd["nodes"]])
se = np.array([[int(round(float(v))) - 1 for v in (str(r[0]).split() if len(r) == 1 else r)][:3]
               for r in sd["elements"]])

# web centre x: centre of the solid web's x-extent at the equator
sl = sn[np.abs(sn[:, 1]) < 0.003]
wx = sl[np.abs(sl[:, 0]) < 0.02, 0]
xc = 0.5 * (wx.min() + wx.max())
yc = np.sqrt(R**2 - xc**2)

fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.add_collection(PolyCollection(sn[se], facecolors="#9ecae1", edgecolors="none"))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(th), R * np.sin(th), ls=":", color="red", lw=1.8)   # ring mid-wall
ax.plot([xc, xc], [-yc, yc], ls=":", color="red", lw=1.8)              # web centre line
pad = 0.06 * (sn[:, 0].max() - sn[:, 0].min())
ax.set_xlim(sn[:, 0].min() - pad, sn[:, 0].max() + pad)
ax.set_ylim(sn[:, 1].min() - pad, sn[:, 1].max() + pad)
ax.set_aspect("equal")
ax.set_xlabel(r"$y_2$ (m)", fontsize=15)
ax.set_ylabel(r"$y_3$ (m)", fontsize=15)
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "xsec_2cell_thick.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print("web centre xc=%.4f  yc=%.4f  (R=%.3f)" % (xc, yc, R))
