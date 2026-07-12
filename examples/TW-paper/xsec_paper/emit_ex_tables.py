"""emit_ex_tables.py -- tables + figures for the restructured results section:
  Ex1  two-cell tube, iso + m45 (merged 2-block table) + RM convergence figure
  Ex2  elliptic 4-cell, iso & m45 x thin & thick (merged 2-block table per material)
  Ex3  IEA r/R=0.2 & 0.3 per-term tables (OML) + r0.2 convergence figure
  Ex3.1 full-blade per-station table (diag %err + time)
  timing4 -> 4-case x 3-method cost table
All C^b_ij tables in the RM-taper style: one row per nonzero constant.
  -> results/tex_rm/*.tex, figures/*.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TEX = os.path.join(RES, "tex_rm"); os.makedirs(TEX, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
MARK = ["o", "s", "^", "D", "v", "P"]


def sym(M):
    return 0.5 * (np.asarray(M) + np.asarray(M).T)


def rows_of(*Ss):
    """union of physically nonzero (i,j): diagonals always; couplings if >2% of sqrt(Cii*Cjj)
    in ANY of the given matrices."""
    ij = []
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                ij.append((i, j)); continue
            for S in Ss:
                rel = abs(S[i, j]) / (np.sqrt(abs(S[i, i] * S[j, j])) + 1e-30)
                if rel > 0.02:
                    ij.append((i, j)); break
    return ij


def fmt(x, scale):
    v = x / scale
    if abs(v) < 1e-4:
        return "$\\sim\\!0$"
    return "$%.4g$" % v


def block_table(fname, caption, label, blocks, scale, exp):
    """blocks = [(header, solid6x6, shell6x6), ...] -> one table, per-block (solid|RM|%err)."""
    Ss = [sym(s) for _h, s, _r in blocks]
    Rs = [sym(r) for _h, _s, r in blocks]
    ij = rows_of(*Ss)
    ncol = len(blocks)
    lines = [r"\begin{table}[htpb]", r"\centering", r"\small",
             r"\caption{%s ($\times10^{%d}$).}" % (caption, exp), r"\label{%s}" % label,
             r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{l%s}" % ("|rrr" * ncol), r"\hline",
             " & " + " & ".join(r"\multicolumn{3}{c%s}{%s}" % ("|" if k < ncol - 1 else "", h)
                                for k, (h, _s, _r) in enumerate(blocks)) + r" \\",
             r"$C^{b}_{ij}$" + " & solid & shell & \\%err" * ncol + r" \\", r"\hline"]
    for (i, j) in ij:
        cells = []
        for S, C in zip(Ss, Rs):
            if abs(S[i, j]) / (np.sqrt(abs(S[i, i] * S[j, j])) + 1e-30) <= 0.02 and i != j:
                cells += [r"$\sim\!0$", r"$\sim\!0$", "---"]
            else:
                e = 100.0 * (C[i, j] - S[i, j]) / S[i, j]
                cells += [fmt(S[i, j], scale), fmt(C[i, j], scale), "$%+.1f$" % e]
        lines.append(r"$C^{b}_{%d%d}$ & " % (i + 1, j + 1) + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, fname), "w").write("\n".join(lines))
    print("wrote", fname)


def single_table(fname, caption, label, solid, shell, scale, exp):
    block_table(fname, caption, label, [("", solid, shell)], scale, exp)


def conv_fig(x, E, xlabel, fname, logx=True):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0, color="0.6", lw=0.8)
    for k in range(6):
        ax.plot(x, E[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=5, label="$%s$" % LBL[k])
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("diagonal % error vs 2-D solid")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", fname)


# ================= Ex1: two-cell =================
d = np.load(os.path.join(RES, "ex1_twocell.npz"))
block_table("ex1_two.tex",
            r"Webbed two-cell tube ($R=5$~cm, $t=4$~mm): RM 6-DOF shell vs.\ the 2-D solid, "
            r"isotropic and $[-45^\circ]$ laminate", "tab:two",
            [("isotropic", d["iso_solid"], d["iso_c6"]),
             (r"$[-45^\circ]$", d["m45_solid"], d["m45_c6"])], 1e7, 7)
c = np.load(os.path.join(RES, "ex1_conv.npz"))
conv_fig(c["N"], c["diag_err"], "circumferential elements $N$", "conv_twocell.png")

# ================= Ex2: ellipse =================
d = np.load(os.path.join(RES, "ex2_ellipse.npz"))
for mk, mname in (("iso", "isotropic"), ("m45", r"$[-45^\circ]$")):
    block_table("ex2_ell_%s.tex" % mk,
                r"Elliptic four-cell tube (%s): RM 6-DOF shell vs.\ the 2-D solid, thin "
                r"($t/a=0.02$) and thick ($t/a=0.12$) wall" % mname, "tab:ell%s" % mk,
                [("thin", d["%s_thin_solid" % mk], d["%s_thin_ring" % mk]),
                 ("thick", d["%s_thick_solid" % mk], d["%s_thick_ring" % mk])], 1e9, 9)

# ================= Ex3: IEA r0.2 / r0.3 (OML) =================
d = np.load(os.path.join(RES, "ex31_full_blade.npz"))
ir = {round(float(r), 2): k for k, r in enumerate(d["r"])}
for rr, tag in ((0.2, "iea020"), (0.3, "iea030")):
    k = ir[rr]
    single_table("%s.tex" % tag,
                 r"IEA-22 blade cross-section at $r/R=%.1f$ (OML reference): RM 6-DOF shell "
                 r"vs.\ the 2-D solid" % rr, "tab:%s" % tag,
                 d["solids"][k], d["rings"][k], 1e9, 9)
cv = np.load(os.path.join(RES, "ex3_iea_conv.npz"))
conv_fig(cv["nnode"], cv["diag_err"], "contour nodes", "conv_iea_r020.png")

# ================= Ex3.1: full blade table =================
E = d["diag_err"]; R = d["r"]; T = d["times"]
lines = [r"\begin{table}[t]\centering\small",
         r"\caption{Full IEA-22 blade (OML reference): RM 6-DOF shell cross-section vs.\ "
         r"2-D solid at every span station --- diagonal \%\,error and ring solve time.}\label{tab:fullblade}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"$r$ & $EA$ & $GA_2$ & $GA_3$ & $GJ$ & $EI_2$ & $EI_3$ & time (s)\\", r"\midrule"]
for k in range(len(R)):
    lines.append("$%.1f$ & " % R[k] + " & ".join("$%+.1f$" % E[k, i] for i in range(6))
                 + " & $%.2f$" % T[k] + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "fullblade.tex"), "w").write("\n".join(lines))
print("wrote fullblade.tex")

# ================= timing table =================
t = np.load(os.path.join(RES, "timing4.npz"))
names = {"two_cell": "two-cell tube", "ellipse": "elliptic four-cell",
         "iea_r020": r"IEA-22 $r/R=0.2$", "full_blade": "full IEA-22 blade (8 stations)"}
lines = [r"\begin{table}[t]\centering\small",
         r"\caption{Wall-clock cost of the three homogenizers on the four examples: the RM "
         r"6-DOF shell ring (1-D contour), the OpenSG-JAX 2-D solid, and the OpenSG-FEniCS "
         r"2-D solid (same mesh). JAX time is the compiled-kernel solve; its one-off JIT "
         r"compilation is excluded. The blade row sums all eight stations.}\label{tab:timing4}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l|rr|rr|rr}", r"\toprule",
         r"& \multicolumn{2}{c|}{RM shell (1-D)} & \multicolumn{2}{c|}{JAX 2-D solid} & "
         r"\multicolumn{2}{c}{FEniCS 2-D solid}\\",
         r"example & DOF & $t$ (s) & DOF & $t$ (s) & DOF & $t$ (s)\\", r"\midrule"]
for name, row in zip(t["names"], t["rows"]):
    ds, ts, dj, _tj1, tj2, df, tf = row
    def dof(v):
        return "%d" % v if v > 0 else "---"
    lines.append("%s & %s & $%.2f$ & %s & $%.2f$ & %s & $%.2f$\\\\"
                 % (names.get(str(name), str(name)), dof(ds), ts, dof(dj), tj2, dof(df), tf))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "timing4.tex"), "w").write("\n".join(lines))
print("wrote timing4.tex")

# ================= mesh figures (real computed meshes) =================
import yaml as _y
def rd(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str)
                               else v)]


def mesh_fig(solid_yaml, shell_yaml, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ds = _y.safe_load(open(solid_yaml))
    nd = np.array([rd(n) for n in ds["nodes"]])
    for q in ds["elements"]:
        ii = [int(x) - 1 for x in (q[0].split() if isinstance(q[0], str) else q)]
        p = nd[ii + [ii[0]]]
        ax[0].fill(p[:, 0], p[:, 1], color="#88aacc", lw=0)
    ax[0].set_title("2-D solid"); ax[0].set_aspect("equal"); ax[0].axis("off")
    dsh = _y.safe_load(open(shell_yaml))
    nds = np.array([rd(n) for n in dsh["nodes"]])
    for e in dsh["elements"]:
        ii = [int(x) - 1 for x in (e if not isinstance(e[0], str) else e[0].split())]
        p = nds[ii]
        ax[1].plot(p[:, 0], p[:, 1], color="k", lw=1.2)
    ax[1].set_title("1-D RM shell contour"); ax[1].set_aspect("equal"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", fname)


mdir = os.path.join(HERE, "ellipse", "meshes")
mesh_fig(os.path.join(mdir, "solid_ell4cell_iso.yaml"),
         os.path.join(mdir, "shell_ell4cell_iso.yaml"), "ell4cell_mesh.png")
TCD = os.path.join(HERE, "..", "two_cell_tube", "data")
mesh_fig(os.path.join(TCD, "solid_tube2cell_thin.yaml"),
         os.path.join(TCD, "tube2cell_thin.yaml"), "twocell_mesh.png")
print("ALL EMITTED")
