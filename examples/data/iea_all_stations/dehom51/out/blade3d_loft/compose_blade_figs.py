"""compose_blade_figs.py -- Figs 19 & 20: tight page-fitting composites of the
full-blade recovered fields, with ONE xyz triad for the whole set (not per image),
and---for the stress figure---a magnified quad-mesh cutaway inset just above the blade
showing the inboard high-stress band, connected by a simple arrow (no box).

Reads the pv_png_sigma/ ParaView renders (1500x1000, parallel projection, same camera;
a baked-in orientation triad at lower-left, a colour bar at right).  Whitens the triad
(and, for the cutaway inset, the redundant colour bar), autocrops to content, and lays
the panels out in absolute figure coordinates for full control.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "pv_png_sigma")


def load_trim(name, drop_bar=False):
    im = mpimg.imread(os.path.join(PNG, name))
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    im = im[:, :, :3].copy()
    h, w = im.shape[:2]
    im[int(0.74 * h):, :int(0.14 * w)] = 255                # erase baked-in xyz triad
    if drop_bar:
        im[:, int(0.80 * w):] = 255                         # erase redundant colour bar
    nonwhite = (im.min(2) < 245)
    ys, xs = np.where(nonwhite)
    pad = 5
    return im[max(ys.min() - pad, 0):ys.max() + pad,
              max(xs.min() - pad, 0):xs.max() + pad]


# screen-projected unit directions of the world axes under the iso render camera
# (position offset (2.4,-2.8,2.0)*dd, view-up z): z is up, x down-right, y up-right.
_TRIAD = [((0.926, -0.378), "#d62728", "x"),
          ((0.875, 0.484), "#8c8c1a", "y"),
          ((0.0, 1.0), "#2ca02c", "z")]


def draw_triad(fig, rect):
    """One clean isometric xyz tripod in a dedicated equal-aspect inset (rect in figure
    coordinates), matching the render camera; labels padded so nothing crowds."""
    ax = fig.add_axes(rect)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.35, 1.7); ax.set_ylim(-0.75, 1.35)
    for (dx, dy), c, lab in _TRIAD:
        ax.add_patch(FancyArrowPatch((0, 0), (dx, dy), arrowstyle="-|>",
                     mutation_scale=13, lw=2.2, color=c, clip_on=False))
        ax.text(1.30 * dx, 1.30 * dy, lab, color=c, fontsize=13, fontweight="bold",
                ha="center", va="center")


def blade_row(fig, im, rect, lab):
    ax = fig.add_axes(rect)
    ax.imshow(im); ax.axis("off")
    ax.text(-0.015, 0.5, lab, transform=ax.transAxes, rotation=90, va="center",
            ha="right", fontsize=17)
    return ax


def stress_figure():
    rows = [("blade_conf_RM_S11.png", r"$\sigma_{11}$"),
            ("blade_conf_RM_S22.png", r"$\sigma_{22}$"),
            ("blade_conf_RM_S12.png", r"$\sigma_{12}$")]
    imgs = [load_trim(f) for f, _ in rows]
    fig = plt.figure(figsize=(7.4, 6.2))
    ys = [0.660, 0.345, 0.030]
    for y, im, (_, lab) in zip(ys, imgs, rows):
        blade_row(fig, im, [0.07, y, 0.90, 0.300], lab)
    draw_triad(fig, [0.02, 0.86, 0.13, 0.13])                       # ONE triad, top-left
    fig.savefig(os.path.join(HERE, "fig_blade_stress.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_blade_stress.png")


def disp_figure():
    rows = [("blade_conf_RM_u1.png", r"$u_1$"),
            ("blade_conf_RM_u2.png", r"$u_2$"),
            ("blade_conf_RM_u3.png", r"$u_3$")]
    imgs = [load_trim(f) for f, _ in rows]
    fig = plt.figure(figsize=(7.4, 6.2))
    ys = [0.660, 0.345, 0.030]
    for y, im, (_, lab) in zip(ys, imgs, rows):
        blade_row(fig, im, [0.07, y, 0.90, 0.300], lab)
    draw_triad(fig, [0.02, 0.86, 0.13, 0.13])
    fig.savefig(os.path.join(HERE, "fig_blade_disp.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_blade_disp.png")


stress_figure()
disp_figure()
