"""render_ell_boundary.py -- ellipse BOUNDARY (end cross-section) mesh figure:
RM shell ring (1-D mid-surface contour + web mid-lines) vs the 3-D solid end face
(2-D through-thickness-resolved quad mesh).  Webbed ellipse, thick wall so the
annulus/web thickness is visible.

    python render_ell_boundary.py [out_dir]
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

OUTD = sys.argv[1] if len(sys.argv) > 1 else "."
A, B, T = 1.0, 0.6, 0.2                    # semi-axes at the L end; thick wall
NC, NR, NW = 48, 4, 6                      # hoop / through-thickness / per web
CSW = [0.5, 0.0, -0.5]                     # web chord fractions x = c*a


def n2d(th):
    v = np.array([B * math.cos(th), A * math.sin(th)]); return v / np.linalg.norm(v)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))

# ---------- SHELL ring: mid-surface contour + web mid-lines ----------
th = np.linspace(0, 2 * math.pi, NC, endpoint=False)
sx = A * np.cos(th); sy = B * np.sin(th)
axL.plot(np.append(sx, sx[0]), np.append(sy, sy[0]), "-o", color="#4c78a8",
         ms=3, lw=1.2, mfc="#4c78a8", mec="#26456e")
for c, col in zip(CSW, ("#c23b22", "#2e8b57", "#c9a227")):
    s = math.sqrt(1 - c * c)
    yy = np.linspace(-s * B, s * B, NW + 1)
    axL.plot(np.full_like(yy, c * A), yy, "-o", color=col, ms=3, lw=1.4)
axL.set_title("RM shell ring (mid-surface)\n%d contour nodes + 3 webs" % NC, fontsize=11)

# ---------- SOLID end face: through-thickness quad mesh ----------
quads = []
def skin_node(k, m):
    t = 2 * math.pi * k / NC
    off = (m / NR - 0.5) * T
    return np.array([A * math.cos(t), B * math.sin(t)]) + off * n2d(t)
for k in range(NC):
    k1 = (k + 1) % NC
    for m in range(NR):
        quads.append([skin_node(k, m), skin_node(k1, m), skin_node(k1, m + 1), skin_node(k, m + 1)])
# webs
for c in CSW:
    s = math.sqrt(1 - c * c)
    for j in range(NW):
        y0 = (-1 + 2 * j / NW) * s * B; y1 = (-1 + 2 * (j + 1) / NW) * s * B
        for m in range(NR):
            x0 = c * A + (m / NR - 0.5) * T; x1 = c * A + ((m + 1) / NR - 0.5) * T
            quads.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
pc = PolyCollection([np.asarray(q) for q in quads], facecolors="#e9b98a",
                    edgecolors="#7a4a1e", linewidths=0.5)
axR.add_collection(pc)
axR.set_title("3-D solid end face\n%d hoop $\\times$ %d thick + webs" % (NC, NR), fontsize=11)

for ax in (axL, axR):
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel("$x_2$"); ax.set_ylabel("$x_3$")
fig.tight_layout()
fn = os.path.join(OUTD, "ell_boundary_mesh.png")
fig.savefig(fn, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", fn)
