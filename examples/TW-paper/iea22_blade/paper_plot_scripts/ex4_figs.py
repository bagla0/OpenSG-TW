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

# ================= (2) spanwise local stress + displacement (VABS vs RM) =================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
# stress: S11, S22, S12  (indices 0,1,5)
S_ID = [(0, "\\sigma_{11}", "#d62728", "o"), (1, "\\sigma_{22}", "#1f77b4", "s"),
        (5, "\\sigma_{12}", "#2ca02c", "^")]
for ci, lab, col, mk in S_ID:
    ax[0].plot(r, sv[:, ci], color=col, marker=mk, ls="-", lw=1.7, ms=6, label="$%s$ VABS" % lab)
    ax[0].plot(r, sr[:, ci], color=col, marker=mk, ls="--", lw=1.7, ms=6, mfc="none",
               label="$%s$ RM" % lab)
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].set_xlabel("span station  $r$"); ax[0].set_ylabel("local stress at crown (MPa)")
ax[0].set_title("local in-plane stress (section top)"); ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8, ncol=1, loc="best")
# displacement: |u| and components
magV = np.linalg.norm(uv, axis=1); magR = np.linalg.norm(ur, axis=1)
D_ID = [(None, "|u|", "#000000", "o", magV, magR),
        (0, "u_1", "#d62728", "s", uv[:, 0], ur[:, 0]),
        (1, "u_2", "#1f77b4", "^", uv[:, 1], ur[:, 1]),
        (2, "u_3", "#2ca02c", "v", uv[:, 2], ur[:, 2])]
for _, lab, col, mk, V, R in D_ID:
    ax[1].plot(r, V, color=col, marker=mk, ls="-", lw=1.7, ms=6, label="$%s$ VABS" % lab)
    ax[1].plot(r, R, color=col, marker=mk, ls="--", lw=1.7, ms=6, mfc="none", label="$%s$ RM" % lab)
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set_xlabel("span station  $r$"); ax[1].set_ylabel("local displacement at crown (mm)")
ax[1].set_title("local displacement (section top)"); ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=8, ncol=2, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "span_dehom.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
print("wrote span_dehom.png")

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
