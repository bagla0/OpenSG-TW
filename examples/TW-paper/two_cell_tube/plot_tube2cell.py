"""Picture of the 2-cell curved tube: FEniCS-2D-solid mesh (annulus + diametral web)
with the JAX shell mid-wall reference line on top.  Shows the 2 closed cells."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"
FIG = os.path.join(os.path.dirname(DATA), "figures")
os.makedirs(FIG, exist_ok=True)


def row(r, n=2):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)][:n]


fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
for ax, tag, ttl in ((axs[0], "thin", "THIN  (h/R = 0.08)"), (axs[1], "thick", "THICK  (h/R = 0.32)")):
    sd = yaml.safe_load(open(os.path.join(DATA, "solid_tube2cell_%s.yaml" % tag)))
    sn = np.array([row(r) for r in sd["nodes"]])
    se = np.array([[int(round(float(v))) - 1 for v in (str(r[0]).split() if len(r) == 1 else r)][:3]
                   for r in sd["elements"]])
    hd = yaml.safe_load(open(os.path.join(DATA, "tube2cell_%s.yaml" % tag)))
    hn = np.array([row(r) for r in hd["nodes"]])
    he = [[int(v) - 1 for v in r[0].split()] for r in hd["elements"]]
    ax.triplot(sn[:, 0], sn[:, 1], se, color="0.78", lw=0.25)        # solid continuum mesh
    for (a, b) in he:
        ax.plot([hn[a, 0], hn[b, 0]], [hn[a, 1], hn[b, 1]], "-", color="tab:red", lw=1.6)
    ax.plot([], [], "-", color="tab:red", lw=1.6, label="shell mid-wall ref")
    ax.plot([], [], "-", color="0.6", lw=1.0, label="2D-solid mesh")
    ax.text(-0.028, 0, "cell 1", ha="center", va="center", fontsize=13, color="tab:blue")
    ax.text(0.028, 0, "cell 2", ha="center", va="center", fontsize=13, color="tab:blue")
    ax.set_aspect("equal"); ax.set_title(ttl, fontsize=15)
    ax.set_xlabel("$y_2$ (m)", fontsize=13); ax.set_ylabel("$y_3$ (m)", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
fig.suptitle("2-cell curved tube (isotropic): solid mesh + shell mid-wall reference", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(FIG, "tube2cell_mesh.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
