"""mh104 e1/e2/e3 element-orientation images for BOTH the shell (1D) and the solid (2D) meshes,
saved into oml_mh104_4way/individual/.  e1=(0,0,1) out-of-plane; e2 (tangent), e3 (ply normal) plotted
as in-plane arrows at element centroids.  Handles 2-node (shell) and 3-node (solid) elements."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
OUT = os.path.join(CC, "oml_mh104_4way", "individual"); os.makedirs(OUT, exist_ok=True)


def _row(r):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)]


def _ir(r):
    return [int(float(v)) for v in (str(r[0]).split() if len(r) == 1 else r)]


def plot_orient(yamlpath, title, fname, sub=1, al=0.04):
    d = yaml.safe_load(open(yamlpath))
    nodes = np.array([_row(r) for r in d["nodes"]])
    elems = [np.array(_ir(r)) - 1 for r in d["elements"]]
    oris = np.array([_row(r) for r in d["elementOrientations"]])
    ctr = np.array([nodes[e, :2].mean(0) for e in elems])
    e1, e2, e3 = oris[:, 0:3], oris[:, 3:6], oris[:, 6:9]
    fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    for ax, (vec, nm, col) in zip(axs, [(e2, "e2 (in-plane tangent / fiber)", "tab:blue"),
                                        (e3, "e3 (ply normal, OML->IML)", "tab:red")]):
        c = ctr[::sub]; v = vec[::sub]
        ax.quiver(c[:, 0], c[:, 1], v[:, 0], v[:, 1], color=col, angles="xy",
                  scale_units="xy", scale=1.0 / al, width=0.0035)
        ax.set_aspect("equal"); ax.set_title("%s -- %s" % (title, nm), fontsize=11)
        ax.set_xlabel("X (chord)"); ax.set_ylabel("Y")
    fig.suptitle("%s element orientation  (e1=(0,0,1) out-of-plane; e1_z in [%.3f, %.3f])"
                 % (title, e1[:, 2].min(), e1[:, 2].max()), fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)


shell = os.path.join(CC, "mh104_thickness_study", "debug", "shell_ref_f020_connect.yaml")
solid = os.path.join(CC, "mh104_thickness_study", "yaml_solid", "solid_f020.yaml")
plot_orient(shell, "mh104 SHELL (1D, f=0.2)", "orient_shell_e1e2e3.png", sub=1)
plot_orient(solid, "mh104 SOLID (2D, f=0.2)", "orient_solid_e1e2e3.png", sub=12)
