"""1D-shell mesh refinement sweep for the IEA-22: for every station, build the contour at mesh_size
0.01 / 0.005 / 0.0025 (up to ~4x finer), run RM and KL at each, report the max diagonal change
coarse->fine (convergence), emit the finest shell, and write a solid+shell orientation PNG per station
plus an all-station montage."""
import os, sys
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff
from fe_jax.orient_plot import plot_orient

VAL = os.path.join(CC, "windio_converter", "validation")
REF = os.path.join(VAL, "refined"); os.makedirs(REF, exist_ok=True)
ORI = os.path.join(VAL, "orient_refined"); os.makedirs(ORI, exist_ok=True)
blade = WindIOBlade(os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml"))
STATIONS = [(round(0.1 * k, 2), "r%03d" % (10 * k)) for k in range(1, 10)] + [(0.95, "r095")]
LEVELS = [0.01, 0.005, 0.0025]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


print("  r    n_el(coarse->fine)   max|diag Δ| coarse->fine:  RM%      KL%")
fine_shells = {}
for r, tag in STATIONS:
    diagRM, diagKL, nel = [], [], []
    for ms in LEVELS:
        cs = build_cross_section(blade, r, mesh_size=ms)
        sh = os.path.join(REF, "shell_iea22_%s_ms%s.yaml" % (tag, str(ms).replace(".", "p")))
        emit_opensg_yaml(cs, sh, web_mesh=ms * cs["chord"])   # refine webs at the SAME density as the skin
        nel.append(len(cs["elems"]))
        diagRM.append(np.diag(sym(rm_timoshenko_6x6(sh, 0.0, orient=False))))
        diagKL.append(np.diag(sym(gradient_junction_kirchhoff(sh, frac=0.0, orient=False)[0])))
        if ms == LEVELS[-1]:
            fine_shells[tag] = sh
    dRM = 100 * np.max(np.abs((diagRM[-1] - diagRM[0]) / diagRM[0]))
    dKL = 100 * np.max(np.abs((diagKL[-1] - diagKL[0]) / diagKL[0]))
    print("  %.2f   %4d -> %4d           %+8.3f  %+8.3f" % (r, nel[0], nel[-1], dRM, dKL))
    # orientation at finest, with the 2D-solid for reference
    solid = os.path.join(VAL, "solid_iea22_%s.yaml" % tag)
    plot_orient(fine_shells[tag], solid if os.path.exists(solid) else None,
                os.path.join(ORI, "orient_iea22_%s_refined.png" % tag))


# ---- montage of the refined shells (all 10) ----
def load(p):
    d = yaml.safe_load(open(p))
    nd = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    el = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    ori = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in d["elementOrientations"]])
    web = set().union(*[set(int(x) - 1 for x in s["labels"]) for s in d["sets"]["element"] if s["name"] == "layup_5"]) \
        if any(s["name"] == "layup_5" for s in d["sets"]["element"]) else set()
    return nd, el, ori, web


fig, axes = plt.subplots(2, 5, figsize=(22, 7))
for ax, (r, tag) in zip(axes.flat, STATIONS):
    nd, el, ori, web = load(fine_shells[tag]); C = nd.mean(0)
    for k, e in enumerate(el):
        P = nd[e[:2]]; ax.plot(P[:, 0], P[:, 1], color=("tab:purple" if k in web else "0.6"),
                               lw=(1.4 if k in web else 0.5))
    for k in range(0, len(el), 3):
        cen = nd[el[k][:2]].mean(0); e3 = ori[k, 6:8]
        grn = (k not in web) and (np.dot(e3, C - cen) > 0)
        ax.quiver(cen[0], cen[1], e3[0], e3[1], color=("tab:green" if grn else "tab:red"), scale=42, width=0.005)
    ax.set_aspect("equal"); ax.set_title("r=%.2f  (%d elems)" % (r, len(el)), fontsize=9)
fig.suptitle("IEA-22 highly-refined 1D-shell cross-sections (mesh_size=0.0025) — e3 green=OML->IML, red=web/LE", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(VAL, "iea22_refined_montage.png"), dpi=130, bbox_inches="tight")
print("\nwrote refined shells -> validation/refined/, per-station orient -> validation/orient_refined/,")
print("montage -> validation/iea22_refined_montage.png")
