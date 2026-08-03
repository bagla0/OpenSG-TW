"""compose_cutaway_row.py -- the separate cutaway-segment figure: the inboard conformal
quad-mesh band (r/R = 0.24-0.29) coloured by sigma11, sigma22, sigma12, side by side in
a SINGLE ROW, with ONE xyz triad.  Reads cut_S11/S22/S12.png (from render_cutaway_row.py).
Writes fig_blade_cutaway_row.png in this folder.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def trim(name):
    im = mpimg.imread(os.path.join(HERE, name))
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    im = im[:, :, :3]
    nz = (im.min(2) < 245)
    ys, xs = np.where(nz)
    p = 5
    return im[max(ys.min() - p, 0):ys.max() + p, max(xs.min() - p, 0):xs.max() + p]


_TRIAD = [((0.926, -0.378), "#d62728", "x"),
          ((0.875, 0.484), "#8c8c1a", "y"),
          ((0.0, 1.0), "#2ca02c", "z")]


def draw_triad(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.35, 1.7); ax.set_ylim(-0.75, 1.35)
    for (dx, dy), c, lab in _TRIAD:
        ax.add_patch(FancyArrowPatch((0, 0), (dx, dy), arrowstyle="-|>",
                     mutation_scale=12, lw=2.0, color=c, clip_on=False))
        ax.text(1.30 * dx, 1.30 * dy, lab, color=c, fontsize=12, fontweight="bold",
                ha="center", va="center")


panels = [("cut_S11.png", r"$\sigma_{11}$"), ("cut_S22.png", r"$\sigma_{22}$"),
          ("cut_S12.png", r"$\sigma_{12}$")]
imgs = [trim(f) for f, _ in panels]
fig = plt.figure(figsize=(13.2, 3.5))
for k, (im, (_, lab)) in enumerate(zip(imgs, panels)):
    ax = fig.add_axes([0.005 + k / 3.0, 0.02, 1 / 3.0 - 0.01, 0.90])
    ax.imshow(im); ax.axis("off")
    ax.set_title(lab, fontsize=17, pad=2)
draw_triad(fig, [0.005, 0.05, 0.075, 0.42])
fig.savefig(os.path.join(HERE, "fig_blade_cutaway_row.png"), dpi=175, bbox_inches="tight")
plt.close(fig)
print("wrote fig_blade_cutaway_row.png")
