"""
Station-15 cross-section figure for the paper: the airfoil shell contour
colored by layup region, with the chosen low-thickness dehom path
(lp_fore_panel) highlighted, plus the through-thickness ply stack of that panel.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml
from fe_jax.msg_mesh import read_mesh

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
PDIR = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
        r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001\st15_path_coords")
PATH = "solid.lp_fore_panel_thickness_015.coords"
OUT = os.path.join(HERE, "..", "outputs", "tube_dehom")  # reuse the figs folder


def main():
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    xy = nodes[:, :2]
    coords = np.loadtxt(os.path.join(PDIR, PATH))[:, :2]
    pmid = coords.mean(0)

    # element midpoints + nearest element to the path -> the panel layup
    mids = np.array([0.5*(xy[c[0]]+xy[c[-1]]) for c in cells])
    inear = int(np.argmin(np.hypot(*(mids - pmid).T)))
    panel = lpe[inear]
    lp = layup_db[panel]
    htot = float(sum(lp["thick"]))
    print(f"chosen path '{panel}'  total thickness = {htot*1e3:.2f} mm, "
          f"{len(lp['thick'])} plies")
    for m, t, a in zip(lp["mat_names"], lp["thick"], lp["angles"]):
        print(f"   {m:12s} t={t*1e3:6.3f} mm  angle={a:+.0f}")

    uniq = sorted(set(lpe))
    cmap = plt.get_cmap("tab20", len(uniq))
    cidx = {ln: cmap(i) for i, ln in enumerate(uniq)}

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_axes([0.04, 0.10, 0.60, 0.82])
    for e, c in enumerate(cells):
        seg = xy[[c[0], c[-1]]]
        ax.plot(seg[:, 0], seg[:, 1], "-", lw=2.4, color=cidx[lpe[e]])
    # highlight the chosen panel elements
    for e, c in enumerate(cells):
        if lpe[e] == panel:
            seg = xy[[c[0], c[-1]]]
            ax.plot(seg[:, 0], seg[:, 1], "-", lw=5, color="k", alpha=0.25)
    ax.plot(coords[:, 0], coords[:, 1], "r.-", ms=4, lw=1.5)
    ax.annotate(f"dehom path\n({panel}, {htot*1e3:.0f} mm)",
                xy=(pmid[0], pmid[1]), xytext=(pmid[0]-0.55, pmid[1]+0.55),
                fontsize=10, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
    ax.set_aspect("equal"); ax.set_xlabel("$y_2$ (m)"); ax.set_ylabel("$y_3$ (m)")
    ax.set_title("(a) Station-15 cross-section, layup regions")
    ax.legend(handles=[Line2D([0], [0], color="r", marker=".",
                              label="dehom path (lp\\_fore\\_panel)")],
              loc="lower right", fontsize=9)

    # (b) through-thickness ply stack of the panel
    axb = fig.add_axes([0.72, 0.12, 0.24, 0.78])
    z = 0.0
    for m, t, a in zip(lp["mat_names"], lp["thick"], lp["angles"]):
        col = "tab:gray" if ("foam" in m.lower() or "balsa" in m.lower()
                             or "core" in m.lower()) else "tab:blue"
        axb.add_patch(plt.Rectangle((0, z), 1, t*1e3, facecolor=col,
                                    edgecolor="k", alpha=0.75))
        if t*1e3 > 0.8:
            axb.text(0.5, z + t*1e3/2, f"{m}\n{a:+.0f}$^\\circ$", ha="center",
                     va="center", fontsize=7)
        z += t*1e3
    axb.set_xlim(0, 1); axb.set_ylim(0, z); axb.set_xticks([])
    axb.set_ylabel("through-thickness (mm), OML$\\to$IML")
    axb.set_title("(b) panel layup")
    fig.savefig(os.path.join(OUT, "fig_st15_section.png"), dpi=150)
    plt.close(fig); print("wrote", os.path.join(OUT, "fig_st15_section.png"))


if __name__ == "__main__":
    main()
