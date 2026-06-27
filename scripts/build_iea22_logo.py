# -*- coding: utf-8 -*-
"""OpenSG-TW logo: the span loft (cross-sections + beam axis + station dots), no axes/labels, + wordmark."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image, ImageChops
REPO = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, REPO)
from opensg_jax.fe_jax.blade_viz import build_layup_registry, _station, PALETTE
IB = os.path.join(REPO, "examples", "data", "iea_blade")
ST = [("r020", 0.2), ("r030", 0.3), ("r040", 0.4), ("r050", 0.5), ("r060", 0.6), ("r070", 0.7), ("r080", 0.8), ("r090", 0.9)]
SH = [os.path.join(IB, "shell_%s.yaml" % t) for t, _ in ST]; RR = [r for _, r in ST]
sig2idx, labels = build_layup_registry(SH)

fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_axes([0, 0, 1, 1], projection="3d")
ax.set_proj_type("ortho")
centers = []
for y, r in zip(SH, RR):
    nd, elems, lay, oris, emid = _station(y, sig2idx)
    segs = [[(r, nd[e[0], 0], nd[e[0], 1]), (r, nd[e[1], 0], nd[e[1], 1])] for e in elems]
    ax.add_collection3d(Line3DCollection(segs, colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=2.0))
    c = nd.mean(0); centers.append((r, c[0], c[1]))
centers = np.array(centers)
ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="black", ls=":", lw=2.2, zorder=10)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="red", s=42, zorder=11)
ax.set_xlim(min(RR) - 0.05, max(RR) + 0.05); ax.set_ylim(0, 5); ax.set_zlim(-0.9, 0.9)
ax.set_box_aspect((6.5, 5.0, 1.6))
ax.view_init(elev=18, azim=-74)
ax.set_axis_off()
fig.text(0.64, 0.84, "OpenSG-TW", ha="center", va="center", fontsize=42, weight="bold", color="#2b3a55")
fig.text(0.64, 0.69, "thin-walled composite beam homogenization", ha="center", va="center",
         fontsize=13, color="#5a6b7b")


def autocrop(path, transparent, pad=14):
    im = Image.open(path)
    if transparent:
        bbox = im.split()[-1].getbbox()                      # alpha channel non-zero region
    else:
        rgb = im.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        bbox = ImageChops.difference(rgb, bg).getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        im = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
        im.save(path)


prev = r"Y:\claude_tmp\opensg_logo.png"
repo = os.path.join(REPO, "docs", "_static", "opensg_tw_logo.png")
fig.savefig(prev, dpi=150, bbox_inches="tight"); autocrop(prev, False)
fig.savefig(repo, dpi=150, bbox_inches="tight", transparent=True); autocrop(repo, True)
print("wrote opensg_tw_logo.png (cropped)")
