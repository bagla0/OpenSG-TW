"""emit_st15_dehom_paper.py -- all the station-15 dehomogenization figures/tables for the
paper, rendered/computed from the committed YAMLs + benchmarks:
  tab_rm/st15_mat.tex     material properties (from st15_shell.yaml)
  tab_rm/st15_layup.tex   the 10 wall laminates (grouped plies)
  tab_rm/st15_homo.tex    RM shell Timoshenko 6x6 vs VABS .K (%err)
  figures/st15_mesh.png   2-D solid (by material) + 1-D shell contour, from the YAMLs
  figures/st15_path.png   the section with the cap-centre dehom path drawn as an arrow
"""
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml

DATA = os.path.join(REPO, "examples", "data")
SHELL = os.path.join(DATA, "1d_yaml", "st15_shell.yaml")
SOLID = os.path.join(DATA, "2d_yaml", "st15_solid.yaml")
KF = os.path.join(DATA, "benchmark", "st15_vabs.K")
DEH = os.path.join(DATA, "dehom_st15")
TEX = os.path.join(HERE, "results", "tex_rm"); os.makedirs(TEX, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
FILL = ["#9ec7e8", "#f6b78b", "#a7d6a0", "#c8b3de", "#f0a3a3", "#d9c69a", "#bcbddc",
        "#fdae6b", "#a1d99b"]
NICE = {"Gelcoat": "Gelcoat", "Adhesive": "Adhesive", "glass_uni": "Glass uniax",
        "glass_biax": "Glass biax", "glass_triax": "Glass triax",
        "carbon_uni_industry_baseline": "Carbon uniax",
        "medium_density_foam": "Foam (MD)", "resin": "Resin", "steel": "Steel"}
ds = yaml.safe_load(open(SHELL))


def rd(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


# ---------------- material properties table ----------------
used = set()
for s in ds["sections"]:
    for p in s["layup"]:
        used.add(p[0])
lines = [r"\begin{table}[htpb]\centering\small",
         r"\caption{Constituent material properties of the IEA/BAR-URC station-15 laminates "
         r"($E$, $G$ in GPa; $\rho$ in kg/m$^3$).}\label{tab:st15_mat}",
         r"\setlength{\tabcolsep}{5pt}\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lcccccccccc}", r"\hline",
         r"Material & $E_1$ & $E_2$ & $E_3$ & $G_{12}$ & $G_{13}$ & $G_{23}$ & "
         r"$\nu_{12}$ & $\nu_{13}$ & $\nu_{23}$ & $\rho$ \\", r"\hline"]
for mm in ds["materials"]:
    if mm["name"] not in used:
        continue
    e = mm.get("elastic", mm)
    E = [v / 1e9 for v in e["E"]]; G = [v / 1e9 for v in e["G"]]; nu = e["nu"]
    rho = mm.get("density", mm.get("rho", 0.0))
    lines.append("%-13s & %.3g & %.3g & %.3g & %.4g & %.4g & %.4g & %.3g & %.3g & %.3g & %.0f \\\\"
                 % (NICE.get(mm["name"], mm["name"]), E[0], E[1], E[2], G[0], G[1], G[2],
                    nu[0], nu[1], nu[2], rho))
lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table}"]
open(os.path.join(TEX, "st15_mat.tex"), "w").write("\n".join(lines))
print("wrote st15_mat.tex")

# ---------------- layup table (group consecutive identical plies) ----------------
REG = {0: "LE glass skin", 1: "Foam panel", 2: "Carbon spar cap", 3: "Foam panel",
       4: "TE glass skin", 5: "Foam panel", 6: "Carbon spar cap", 7: "Foam panel",
       8: "Shear web", 9: "Shear web"}
lines = [r"\begin{table}[htpb]\centering\small",
         r"\caption{Wall laminates of the station-15 section (from the outer mould line "
         r"inward); each row is one stack of identical plies.}\label{tab:st15_layup}",
         r"\resizebox{0.92\textwidth}{!}{%", r"\begin{tabular}{llclcc}", r"\hline",
         r"Name & Region & Layer & Material & Ply thickness (m) & \# plies \\", r"\hline"]
for si, sec in enumerate(ds["sections"]):
    plies = [(p[0], float(p[1])) for p in sec["layup"]]
    groups = []
    for mn, t in plies:
        if groups and groups[-1][0] == mn and abs(groups[-1][1] - t) < 1e-12:
            groups[-1][2] += 1
        else:
            groups.append([mn, t, 1])
    nm = sec.get("elementSet", "layup_%d" % si).replace("_", r"\_")
    for k, (mn, t, n) in enumerate(groups):
        head = (r"\multirow{%d}{*}{%s} & \multirow{%d}{*}{%s}" % (len(groups), nm, len(groups),
                REG.get(si, "")) if k == 0 else " & ")
        lines.append("%s & %d & %s & %.4g & %d \\\\" % (head, k + 1, NICE.get(mn, mn), t, n))
    lines.append(r"\hline")
lines += [r"\end{tabular}", r"}", r"\end{table}"]
open(os.path.join(TEX, "st15_layup.tex"), "w").write("\n".join(lines))
print("wrote st15_layup.tex")

# ---------------- homogenization table RM vs VABS .K ----------------
def load_vabs_timo(path):
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i + 1:]:
        p = ln.split()
        try:
            [float(x) for x in p]; ok = (len(p) == 6)
        except ValueError:
            ok = False
        if ok:
            rows.append([float(x) for x in p])
        if len(rows) == 6:
            break
    return np.array(rows)

K = load_vabs_timo(KF)
J = {r: np.asarray(solve_tw_from_yaml(SHELL, frac=f)["Timo"])
     for r, f in (("OML", 0.0), ("cen", 0.5), ("IML", 1.0))}
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
UN = ["N", "N", "N", "N\\,m$^2$", "N\\,m$^2$", "N\\,m$^2$"]
lines = [r"\begin{table}[htpb]\centering\small",
         r"\caption{Station-15 Timoshenko $6\times6$: RM shell (three references) vs the VABS "
         r"$.\mathrm{K}$ (section axes). The thick carbon spar cap makes the chordwise "
         r"bending $EI_3$ the hardest mode for a single-director shell.}\label{tab:st15_homo}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"term & VABS $.\mathrm{K}$ & RM (OML) & \%err OML & \%err cen & \%err IML \\",
         r"\midrule"]
for i in range(6):
    v = K[i, i]
    lines.append("$%s$ & $%.3e$ & $%.3e$ & $%+.1f$ & $%+.1f$ & $%+.1f$ \\\\"
                 % (LBL[i], v, J["OML"][i, i], 100*(J["OML"][i,i]-v)/v,
                    100*(J["cen"][i,i]-v)/v, 100*(J["IML"][i,i]-v)/v))
fro = np.linalg.norm(J["OML"] - K) / np.linalg.norm(K) * 100
lines += [r"\midrule",
          r"\multicolumn{6}{l}{\small full-$6\times6$ Frobenius error at OML $=%.1f\%%$} \\" % fro,
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "st15_homo.tex"), "w").write("\n".join(lines))
print("wrote st15_homo.tex (Frobenius %.2f%%)" % fro)

# ---------------- mesh figure: 2-D solid (by material) + 1-D shell contour ----------------
dsol = yaml.safe_load(open(SOLID))
nd = np.array([rd(n)[:2] for n in dsol["nodes"]])
mat = np.zeros(len(dsol["elements"]), int)
for si, grp in enumerate(dsol.get("sets", {}).get("element", [])):
    mi = int("".join(c for c in grp["name"] if c.isdigit()) or si + 1) - 1
    for lab in grp["labels"]:
        mat[int(lab) - 1] = mi
polys, cidx = [], []
for k, e in enumerate(dsol["elements"]):
    ii = [int(round(x)) - 1 for x in rd(e)]
    ii = [i for i in ii if i >= 0][:4]
    if len(ii) >= 3:
        polys.append(nd[ii]); cidx.append(mat[k])
nsh = np.array([rd(n)[:2] for n in ds["nodes"]])
segs = [[int(round(x)) - 1 for x in rd(e)][:2] for e in ds["elements"]]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
pc = PolyCollection(polys, facecolors=[FILL[c % len(FILL)] for c in cidx], edgecolors="none")
ax[0].add_collection(pc); ax[0].set_title("2-D solid (by material)", fontsize=10)
for ii in segs:
    ax[1].plot(nsh[ii, 0], nsh[ii, 1], "-", color="k", lw=1.3)
ax[1].set_title("1-D RM shell contour", fontsize=10)
for a in ax:
    a.set_aspect("equal"); a.autoscale(); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "st15_mesh.png"), dpi=190, bbox_inches="tight")
plt.close(fig); print("wrote st15_mesh.png (solid %d elems / shell %d segs)" % (len(polys), len(segs)))

# ---------------- section + dehom path drawn as an arrow (with a cap zoom inset) ----------------
cc = np.loadtxt(os.path.join(DEH, "solid.lp_sparcap_center_thickness_015.coords"))[:, :2]
cx, cy = cc[:, 0].mean(), cc[:, 1].mean()
fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.add_collection(PolyCollection(polys, facecolors="#e2e8f0", edgecolors="none"))
for ii in segs:
    ax.plot(nsh[ii, 0], nsh[ii, 1], "-", color="0.35", lw=0.8)
# marker box at the cap + leader to the zoom inset
ax.add_patch(plt.Rectangle((cx - 0.13, cc[:, 1].min() - 0.02), 0.26,
             np.ptp(cc[:, 1]) + 0.04, fill=False, ec="#d62728", lw=1.4))
ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
ax.set_title("Station-15 section", fontsize=10, loc="left")
# zoom inset of the cap with the OML->IML arrow through the thickness
axin = fig.add_axes([0.66, 0.20, 0.30, 0.62])
axin.add_collection(PolyCollection(polys, facecolors=[FILL[c % len(FILL)] for c in cidx],
                                   edgecolors="none"))
axin.annotate("", xy=cc[-1], xytext=cc[0],
              arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.4, mutation_scale=16))
axin.plot(cc[:, 0], cc[:, 1], ".", color="#d62728", ms=3)
axin.text(cc[0, 0] + 0.02, cc[0, 1], "OML", color="#d62728", fontsize=8, va="center")
axin.text(cc[-1, 0] + 0.02, cc[-1, 1], "IML", color="#d62728", fontsize=8, va="center")
pad = max(np.ptp(cc, 0)) * 0.7 + 0.02
axin.set_xlim(cx - pad, cx + pad); axin.set_ylim(cc[:, 1].min() - 0.02, cc[:, 1].max() + 0.02)
axin.set_aspect("equal"); axin.set_xticks([]); axin.set_yticks([])
for sp in axin.spines.values():
    sp.set_edgecolor("#d62728")
axin.set_title("cap zoom", fontsize=8, color="#d62728")
fig.savefig(os.path.join(FIG, "st15_path.png"), dpi=190, bbox_inches="tight")
plt.close(fig); print("wrote st15_path.png")
