"""Verify the material-orientation convention is consistent between the 2D-solid and 1D-shell YAMLs:
  e1 = beam axis (z in YAML; fibre tilts off it by theta3 for off-axis plies),
  e2 = contour FLOW direction (the .dat/XML traversal tangent),
  e3 = ply through-thickness normal, pointing OML->IML (inward, toward the section interior).
The shell e3 must point the SAME way (inward) as the solid e3.  Cross-section is in (x,y), beam axis z,
so e2/e3 are in-plane and e1 is out-of-page; a 2D quiver looking down the beam axis shows them."""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SOLID = os.path.join(HERE, "yaml_solid", "solid_f100.yaml")
SHELL = os.path.join(HERE, "yaml_shell", "shell_f100.yaml")


def load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(v) for v in str(r[0]).split()] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in r] for r in d["elementOrientations"]])
    cent = np.array([nodes[e].mean(axis=0) for e in elems])
    return nodes, elems, oris, cent


def report(name, oris, cent):
    C = cent[:, :2].mean(axis=0)                       # section centroid (x,y)
    e1, e2, e3 = oris[:, 0:3], oris[:, 3:6], oris[:, 6:9]
    e1z = np.abs(e1[:, 2])                              # e1 . beam(z)
    e2z = np.abs(e2[:, 2])                              # e2 out-of-plane comp (should be ~0)
    inward = np.einsum("ij,ij->i", e3[:, :2], C - cent[:, :2])   # >0 => e3 points toward centroid (OML->IML)
    print("=== %s ===" % name)
    print("  e1.z  (beam-axis alignment): min=%.3f mean=%.3f  (1.0 = fibre along beam; <1 = off-axis tilt theta3)" % (e1z.min(), e1z.mean()))
    print("  e2 out-of-plane |z|:         max=%.3e  (should be ~0: e2 is the in-plane contour tangent)" % e2z.max())
    print("  e3 inward (OML->IML):        %d / %d elements (%.1f%%)  -- e3.(centroid-elem) > 0" % (
        (inward > 0).sum(), len(inward), 100.0 * (inward > 0).mean()))
    return C


ns, es, os_, cs = load(SOLID)
nsh, esh, osh, csh = load(SHELL)
print("Orientation convention check (f=1.0):\n")
Csol = report("2D SOLID (solid_f100.yaml)", os_, cs)
print()
Cshl = report("1D SHELL (shell_f100.yaml)", osh, csh)

# ---- visual: e3 (blue) + e2 (red) quivers, looking down beam axis ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (name, nodes, elems, oris, cent) in zip(
        axes, [("2D solid: e3 ply-normal (blue)", ns, es, os_, cs),
               ("1D shell: e2 flow (red), e3 normal (blue)", nsh, esh, osh, csh)]):
    for e in elems:
        p = nodes[e][:, :2]
        ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]), "-", color="0.8", lw=0.4)
    step = max(1, len(elems) // 220)
    idx = np.arange(0, len(elems), step)
    L = 0.04 * np.linalg.norm(nodes[:, :2].max(0) - nodes[:, :2].min(0))
    e2, e3 = oris[:, 3:6], oris[:, 6:9]
    if "shell" in name:
        ax.quiver(cent[idx, 0], cent[idx, 1], e2[idx, 0], e2[idx, 1], color="tab:red",
                  scale=1 / L, scale_units="xy", angles="xy", width=0.003, label="e2 (flow tangent)")
    ax.quiver(cent[idx, 0], cent[idx, 1], e3[idx, 0], e3[idx, 1], color="tab:blue",
              scale=1 / L, scale_units="xy", angles="xy", width=0.003, label="e3 (ply normal, OML->IML)")
    ax.set_aspect("equal"); ax.set_title(name); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.2)
fig.suptitle("mh104 material orientation: e1=beam(z, into page), e2=contour flow, e3=ply normal (inward) — solid vs shell", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "plots", "orientation_check.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140)
print("\nwrote", out)
