# -*- coding: utf-8 -*-
"""windIO -> OpenSG -> RM/KL/Solid -> Timoshenko beam workflow flowchart."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 5.0))
ax.set_xlim(0, 18); ax.set_ylim(0, 5); ax.axis("off")
BLUE, ORANGE, GREEN, PURPLE, YEL, TEAL = "#cfe3f3", "#fde0c0", "#cfead0", "#e3d4ef", "#fff2b3", "#bfe6e6"


def box(x, y, w, h, text, fc, fs=14, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec="0.35", lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight)


def proc(x, y, w, h, head, sub, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec="0.35", lw=1.3))
    ax.text(x, y + h * 0.27, head, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x, y - h * 0.16, sub, ha="center", va="center", fontsize=9.5)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.5, color="0.35"))


box(1.6, 2.5, 2.6, 1.1, "WindIO Blade\n(.yaml)", BLUE, 14, "bold")
proc(5.2, 3.8, 3.3, 1.35, "OpenSG_io", "load_blade →\nbuild_cross_section →\nemit_opensg_yaml", ORANGE)
proc(5.2, 1.2, 3.3, 1.35, "PreVABS", "emit_prevabs →\nprevabs --vabs --hm →\nconvert_sg_to_yaml", ORANGE)
box(8.9, 3.8, 2.3, 1.0, "1D shell\nSG YAML", GREEN, 13)
box(8.9, 1.2, 2.3, 1.0, "2D solid\nSG YAML", GREEN, 13)
box(11.7, 2.5, 2.0, 1.15, "OpenSG", PURPLE, 16, "bold")
box(14.2, 3.9, 1.6, 0.8, "RM", YEL, 14)
box(14.2, 2.5, 1.6, 0.8, "KL", YEL, 14)
box(14.2, 1.1, 1.6, 0.8, "Solid", YEL, 14)
box(16.9, 2.5, 2.2, 1.25, "Timoshenko\nBeam", TEAL, 14)

arrow(2.9, 2.8, 3.55, 3.5); arrow(2.9, 2.2, 3.55, 1.5)
arrow(6.85, 3.8, 7.75, 3.8); arrow(6.85, 1.2, 7.75, 1.2)
arrow(10.05, 3.8, 10.75, 2.85); arrow(10.05, 1.2, 10.75, 2.15)
arrow(12.7, 2.75, 13.4, 3.85); arrow(12.7, 2.5, 13.4, 2.5); arrow(12.7, 2.25, 13.4, 1.15)
arrow(15.0, 3.9, 15.8, 2.8); arrow(15.0, 2.5, 15.8, 2.5); arrow(15.0, 1.1, 15.8, 2.2)
fig.tight_layout()
fig.savefig(r"Y:\claude_tmp\windio_workflow.png", dpi=105, bbox_inches="tight")
fig.savefig(r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\docs\tutorials\_img\windio_workflow.png",
            dpi=135, bbox_inches="tight")
print("wrote windio_workflow.png (preview + repo)")
