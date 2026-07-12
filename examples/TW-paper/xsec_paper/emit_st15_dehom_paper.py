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

K = 0.5 * (load_vabs_timo(KF) + load_vabs_timo(KF).T)
# RM ring (C0 Lagrange 6-DOF, MITC-tied g23) -- the SAME model as the paper's other tables and
# the dehom (dehom_rm), NOT the KL Hermite shell.  Reported at the OML reference (the airfoil
# convention, center_ref=False), which is where the dehom recovers and what VABS .K compares to.
sys.path.insert(0, HERE)
from oml_ring import load_ring_ref, c6
J = {"OML": c6(load_ring_ref(SHELL, "oml"))}
# all nonzero terms of the VABS .K: 6 diagonals + couplings > 1e-3 of the leading stiffness
gmax = max(abs(K[k, k]) for k in range(6))
NZ = [(i, j) for i in range(6) for j in range(i, 6)
      if i == j or abs(K[i, j]) >= 1e-3 * gmax]
DIAG = {0: "EA", 1: "GA_2", 2: "GA_3", 3: "GJ", 4: "EI_2", 5: "EI_3"}
lines = [r"\begin{table}[htpb]\centering\small",
         r"\caption{Station-15 Timoshenko $6\times6$: every nonzero $C_{ij}$ of the RM shell "
         r"(the $C^0$ drilling-constrained ring with MITC-tied $\gamma_{23}$, at the OML "
         r"reference---the same model the dehomogenization inverts) vs.\ the VABS $.\mathrm{K}$ "
         r"(section axes). VABS order $[1\!=\!\mathrm{ext},2,3\!=\!\mathrm{shear},"
         r"4\!=\!\mathrm{twist},5,6\!=\!\mathrm{bend}]$; the thick carbon spar cap makes the "
         r"chordwise bending $C_{66}$ the hardest mode for a single-director shell.}"
         r"\label{tab:st15_homo}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"$C_{ij}$ & VABS $.\mathrm{K}$ & RM shell & \%\,err \\",
         r"\midrule"]
for (i, j) in NZ:
    nm = "C_{%d%d}" % (i + 1, j + 1)
    if i == j:
        nm += "\\,(%s)" % DIAG[i]
    v = K[i, j]
    lines.append("$%s$ & $%.3e$ & $%.3e$ & $%+.1f$ \\\\"
                 % (nm, v, J["OML"][i, j], 100*(J["OML"][i, j] - v) / v))
fro = np.linalg.norm(J["OML"] - K) / np.linalg.norm(K) * 100
lines += [r"\midrule",
          r"\multicolumn{4}{l}{\small full-$6\times6$ Frobenius error $=%.1f\%%$} \\" % fro,
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "st15_homo.tex"), "w").write("\n".join(lines))
print("wrote st15_homo.tex (%d nonzero terms, Frobenius %.2f%%)" % (len(NZ), fro))

# ---------------- mesh figure: 2-D solid (by material, with legend) + 1-D shell contour ------
from matplotlib.patches import Patch
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

# name each solid material index by matching its E1 to the (named) shell materials
sh_E1 = {m["name"]: m.get("elastic", m)["E"][0] for m in ds["materials"]}
def solid_name(mi):
    mm = dsol["materials"][mi]
    E1 = mm.get("elastic", mm)["E"][0]
    best = min(sh_E1, key=lambda k: abs(sh_E1[k] - E1))
    return NICE.get(best, best)
MNAME = {mi: solid_name(mi) for mi in sorted(set(cidx))}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
pc = PolyCollection(polys, facecolors=[FILL[c % len(FILL)] for c in cidx], edgecolors="none")
ax[0].add_collection(pc); ax[0].set_title("2-D solid (by material)", fontsize=10)
for ii in segs:
    ax[1].plot(nsh[ii, 0], nsh[ii, 1], "-", color="k", lw=1.3)
ax[1].set_title("1-D RM shell contour", fontsize=10)
for a in ax:
    a.set_aspect("equal"); a.autoscale(); a.axis("off")
handles = [Patch(facecolor=FILL[mi % len(FILL)], edgecolor="0.4", label=MNAME[mi])
           for mi in sorted(MNAME)]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(FIG, "st15_mesh.png"), dpi=190, bbox_inches="tight")
plt.close(fig); print("wrote st15_mesh.png (solid %d elems / shell %d segs)" % (len(polys), len(segs)))

# ---------------- path figure: (a) section w/ line arrows across both paths, (b) cap zoom ----
cc = np.loadtxt(os.path.join(DEH, "solid.lp_sparcap_center_thickness_015.coords"))[:, :2]
cx, cy = cc[:, 0].mean(), cc[:, 1].mean()
circ = np.loadtxt(os.path.join(DEH, "solid.circumferential_015.coords"))[:, :2]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9),
                       gridspec_kw={"width_ratios": [2.0, 1.0]})
# (a) full section with both dehom paths drawn as line arrows
ax[0].add_collection(PolyCollection(polys, facecolors="#e2e8f0", edgecolors="none"))
for ii in segs:
    ax[0].plot(nsh[ii, 0], nsh[ii, 1], "-", color="0.45", lw=0.7)
# circumferential path = blue polyline around the section, with a direction arrow
ax[0].plot(circ[:, 0], circ[:, 1], "-", color="#1f77b4", lw=2.0, zorder=5)
nq = int(len(circ) * 0.32)  # place the direction arrow toward the LE, clear of the cap
ax[0].annotate("", xy=circ[nq + 1], xytext=circ[nq - 1],
               arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.0, mutation_scale=18),
               zorder=6)
# label dropped below the skin into open interior, with a thin leader to the path
ax[0].annotate("circumferential path", xy=circ[nq], xytext=(circ[nq, 0], circ[nq, 1] - 0.42),
               color="#1f77b4", fontsize=9, ha="center",
               arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.7), zorder=6)
# cap-centre through-thickness path = short red line arrow at the LP spar cap
d = cc[-1] - cc[0]; n = d / (np.hypot(*d) + 1e-12)
p0, p1 = cc[0] - 0.25 * d, cc[-1] + 0.25 * d
ax[0].annotate("", xy=p1, xytext=p0,
               arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2, mutation_scale=16),
               zorder=6)
ax[0].annotate("spar-cap path\n(see (b))", xy=(cx, cy), xytext=(cx - 0.05, cy - 0.55),
               color="#d62728", fontsize=9, ha="center",
               arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
ax[0].set_aspect("equal"); ax[0].autoscale(); ax[0].axis("off")
ax[0].text(0.01, 0.02, "(a) station-15 section", transform=ax[0].transAxes,
           fontsize=10, va="bottom")
# (b) separate cap-laminate zoom with the OML -> IML through-thickness arrow
ax[1].add_collection(PolyCollection(polys, facecolors=[FILL[c % len(FILL)] for c in cidx],
                                    edgecolors="none"))
ax[1].annotate("", xy=cc[-1], xytext=cc[0],
               arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.6, mutation_scale=18))
ax[1].plot(cc[:, 0], cc[:, 1], ".", color="#d62728", ms=3)
ax[1].text(cc[0, 0] + 0.015, cc[0, 1], "OML", color="#d62728", fontsize=9, va="center")
ax[1].text(cc[-1, 0] + 0.015, cc[-1, 1], "IML", color="#d62728", fontsize=9, va="center")
pad = max(np.ptp(cc, 0)) * 0.8 + 0.02
ax[1].set_xlim(cx - pad, cx + pad)
ax[1].set_ylim(cc[:, 1].min() - 0.03, cc[:, 1].max() + 0.03)
ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
for sp in ax[1].spines.values():
    sp.set_edgecolor("#d62728")
ax[1].set_title("(b) spar-cap zoom", fontsize=10, loc="left", color="#d62728")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "st15_path.png"), dpi=190, bbox_inches="tight")
plt.close(fig); print("wrote st15_path.png (with circumferential + cap paths)")
