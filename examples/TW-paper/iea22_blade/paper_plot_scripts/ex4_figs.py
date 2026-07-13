"""ex4_figs.py -- spanwise figures + table from results/ex4_spanwise.npz (real windIO stations).
  figures/full_blade_rm_span_real.png   RM 6-DOF ring vs VABS 2-D solid, diagonal %err vs r
  figures/span_dehom.png                spanwise local stress + displacement (VABS vs RM) at the
                                        section crown, across the span
  results/tex_rm/fullblade_real.tex     per-station diagonal %err + Frobenius (RM vs VABS)
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); TEX = os.path.join(OUT, "tex_rm"); os.makedirs(TEX, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
z = np.load(os.path.join(OUT, "ex4_spanwise.npz"), allow_pickle=True)
r = z["r"]; af = [str(x) for x in z["af"]]; tags = [str(x) for x in z["tags"]]
E = z["diag_err"]; frob = z["frob"]
uv = z["uv"] * 1e3; ur = z["ur"] * 1e3                  # mm
sv = z["sv"] / 1e6; sr = z["sr"] / 1e6                  # MPa  [S11,S22,S33,S23,S13,S12]
DIAG = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]

# ================= (1) spanwise stiffness %err (Example 4) =================
MARK = ["o", "s", "^", "D", "v", "P"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3), sharex=True)
groups = [([1, 2, 3], "transverse shear & torsion"), ([0, 4, 5], "extension & bending")]
handles, labels = [], []
for a, (idx, title) in zip(ax, groups):
    a.axhline(0, color="0.6", lw=0.8)
    for k in idx:
        ln, = a.plot(r, E[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=6, label="$%s$" % DIAG[k])
        handles.append(ln); labels.append(ln.get_label())
    a.set_xlabel("span station  $r$"); a.set_title(title); a.grid(alpha=0.3)
ax[0].set_ylabel("diagonal % error vs 2-D solid (VABS)")
order = [3, 0, 1, 2, 4, 5]
fig.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=9,
           frameon=False, ncol=1, loc="center left", bbox_to_anchor=(0.91, 0.5))
fig.tight_layout(rect=(0, 0, 0.90, 1))
fig.savefig(os.path.join(FIG, "full_blade_rm_span_real.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote full_blade_rm_span_real.png")

# ============ (2) spanwise local stress + displacement (VABS vs RM), INDIVIDUAL figures ============
xr = (r - r.min()) / (r.max() - r.min())          # non-dim spanwise position, 0 (root sta.) .. 1 (tip sta.)
XL = r"normalized spanwise position $\bar r$"

# --- (2a) local in-plane stress at the crown ---
fig, a = plt.subplots(figsize=(7, 4.2))
S_ID = [(0, "\\sigma_{11}", "#d62728", "o"), (1, "\\sigma_{22}", "#1f77b4", "s"),
        (5, "\\sigma_{12}", "#2ca02c", "^")]
for ci, lab, col, mk in S_ID:
    a.plot(xr, sv[:, ci], color=col, marker=mk, ls="-", lw=1.7, ms=6, label="$%s$ VABS" % lab)
    a.plot(xr, sr[:, ci], color=col, marker=mk, ls="--", lw=1.7, ms=6, mfc="none", label="$%s$ RM" % lab)
a.axhline(0, color="0.6", lw=0.8); a.set_xlim(0, 1)
a.set_xlabel(XL); a.set_ylabel("local in-plane stress at crown (MPa)"); a.grid(alpha=0.3)
a.legend(fontsize=8, ncol=1, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "span_stress.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
print("wrote span_stress.png")

# --- (2b) local displacement at the crown ---
magV = np.linalg.norm(uv, axis=1); magR = np.linalg.norm(ur, axis=1)
fig, a = plt.subplots(figsize=(7, 4.2))
D_ID = [(None, "|u|", "#000000", "o", magV, magR),
        (0, "u_1", "#d62728", "s", uv[:, 0], ur[:, 0]),
        (1, "u_2", "#1f77b4", "^", uv[:, 1], ur[:, 1]),
        (2, "u_3", "#2ca02c", "v", uv[:, 2], ur[:, 2])]
for _, lab, col, mk, V, R in D_ID:
    a.plot(xr, V, color=col, marker=mk, ls="-", lw=1.7, ms=6, label="$%s$ VABS" % lab)
    a.plot(xr, R, color=col, marker=mk, ls="--", lw=1.7, ms=6, mfc="none", label="$%s$ RM" % lab)
a.axhline(0, color="0.6", lw=0.8); a.set_xlim(0, 1)
a.set_xlabel(XL); a.set_ylabel("local displacement at crown (mm)"); a.grid(alpha=0.3)
a.legend(fontsize=8, ncol=2, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "span_disp.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
print("wrote span_disp.png")

# ================= (3) table =================
lines = [r"\begin{table}[t]\centering\small",
         r"\caption{Full IEA-22 blade at the \emph{actual} windIO airfoil stations: RM 6-DOF shell "
         r"cross-section vs.\ the VABS 2-D solid ($.\mathrm{K}$) --- diagonal \%\,error and full "
         r"$6\times6$ Frobenius error. Stations are the windIO \texttt{spanwise\_position} entries "
         r"(airfoil in parentheses); $r=0.2$ is the root example section.}\label{tab:fullblade_real}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"$r$ & airfoil & $EA$ & $GA_2$ & $GA_3$ & $GJ$ & $EI_2$ & $EI_3$ & Frob.\\", r"\midrule"]
for i in range(len(r)):
    nm = af[i].replace("FFA-W3-", "FFA-").replace("360/330", "root")
    lines.append("$%.3f$ & %s & $%+.1f$ & $%+.1f$ & $%+.1f$ & $%+.1f$ & $%+.1f$ & $%+.1f$ & $%.1f\\%%$\\\\"
                 % (r[i], nm, E[i, 0], E[i, 1], E[i, 2], E[i, 3], E[i, 4], E[i, 5], frob[i]))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "fullblade_real.tex"), "w").write("\n".join(lines))
print("wrote fullblade_real.tex")
