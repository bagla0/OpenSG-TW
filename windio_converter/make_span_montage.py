"""Montage of the IEA-22 cross-sections across the span: shell contour + webs + e2/e3 frames, one panel
per station (to the same metre scale), showing the taper and the thin-walled construction."""
import os, sys
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
VAL = os.path.join(CC, "windio_converter", "validation")
STATIONS = [(0.1, "r010"), (0.3, "r030"), (0.5, "r050"), (0.7, "r070"), (0.9, "r090"), (0.95, "r095")]


def load(p):
    d = yaml.safe_load(open(p))
    nodes = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in d["elementOrientations"]])
    web = set()
    for s in d["sets"]["element"]:
        if s["name"] == "layup_5":
            web = set(int(x) - 1 for x in s["labels"])
    return nodes, elems, oris, web


fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (r, tag) in zip(axes.flat, STATIONS):
    nodes, elems, oris, web = load(os.path.join(VAL, "shell_iea22_%s.yaml" % tag))
    C = nodes.mean(0)
    for k, e in enumerate(elems):
        P = nodes[e[:2]]
        col = "tab:gray" if k not in web else "tab:purple"
        ax.plot(P[:, 0], P[:, 1], color=col, lw=(0.7 if k not in web else 1.6))
    for k in range(0, len(elems), 2):
        cen = nodes[elems[k][:2]].mean(0); e3 = oris[k, 6:8]
        grn = (k not in web) and (np.dot(e3, C - cen) > 0)
        ax.quiver(cen[0], cen[1], e3[0], e3[1], color=("tab:green" if grn else "tab:red"),
                  scale=40, width=0.004, alpha=0.8)
    ax.set_aspect("equal"); ax.set_title("r=%.2f  (chord %.2f m)" % (r, nodes[:, 0].max() - nodes[:, 0].min()), fontsize=10)
    ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)")
fig.suptitle("IEA-22 blade cross-sections across the span  (shell contour + 3 shear webs;  e3 green=OML->IML, red=web e1xe2)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(VAL, "iea22_span_montage.png")
fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
