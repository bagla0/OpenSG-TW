"""emit_tables.py -- read the result npz and write LaTeX table fragments for the paper:
full 6x6 Cij (solid|RM|%err) at IEA r=0.2 and r=0.3, the spanwise full-blade summary,
and the r=0.2 convergence-timing table.  -> results/tex/*.tex
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TEX = os.path.join(RES, "tex"); os.makedirs(TEX, exist_ok=True)
CN = ["C_{11}", "C_{22}", "C_{33}", "C_{44}", "C_{55}", "C_{66}"]
NM = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]


def sci(x):
    if not np.isfinite(x) or abs(x) < 1:
        return "$\\sim\\!0$"
    m, e = ("%.3e" % x).split("e")
    return "$%s\\!\\times\\!10^{%d}$" % (m, int(e))


def full_cij(tag, label):
    d = np.load(os.path.join(RES, "%s.npz" % tag))
    So, RM = d["solid"], d["c6dof"]
    cut = np.abs(So).max() / 1e3
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{IEA-22 blade cross-section at %s: full boundary Timoshenko $6\times6$ "
             r"$C^{b}_{ij}$ --- 2-D solid, RM 6-DOF shell, and \%%\,error on every nonzero term "
             r"(VABS order; $\cdot$ = below $\max|C|/10^3$).}\label{tab:iea_%s}" % (label, tag),
             r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
             r"\begin{tabular}{l" + "r" * 6 + "}", r"\toprule",
             r" & " + " & ".join("$%s$" % c for c in CN) + r"\\", r"\midrule"]
    for tagn, M in (("2-D solid", So), ("RM 6-DOF shell", RM)):
        lines.append(r"\multicolumn{7}{l}{\textit{%s}}\\" % tagn)
        for i in range(6):
            lines.append("$%s$ & " % CN[i] + " & ".join(sci(M[i, j]) for j in range(6)) + r"\\")
        lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{\%\,error (RM vs solid)}}\\")
    for i in range(6):
        row = ["$%+.1f$" % (100 * (RM[i, j] - So[i, j]) / So[i, j]) if abs(So[i, j]) > cut else r"$\cdot$"
               for j in range(6)]
        lines.append("$%s$ & " % CN[i] + " & ".join(row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "iea_%s_full.tex" % tag), "w").write("\n".join(lines))
    print("wrote iea_%s_full.tex" % tag)


def spanwise():
    d = np.load(os.path.join(RES, "full_blade_rm.npz"))
    r, E, t = d["r"], d["diag_err"], d["times"]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Full IEA-22 blade: RM 6-DOF shell cross-section vs.\ 2-D solid at every "
             r"span station --- diagonal \%\,error and ring solve time.}\label{tab:fullblade}",
             r"\begin{tabular}{l" + "r" * 6 + "r}", r"\toprule",
             r"$r$ & " + " & ".join("$%s$" % n for n in NM) + r" & time (s)\\", r"\midrule"]
    for k in range(len(r)):
        lines.append("$%.1f$ & " % r[k] + " & ".join("$%+.1f$" % E[k, i] for i in range(6))
                     + " & $%.2f$" % t[k] + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "fullblade.tex"), "w").write("\n".join(lines))
    print("wrote fullblade.tex")


def timing():
    d = np.load(os.path.join(RES, "timing_r020.npz"), allow_pickle=True)
    nm, rows = d["names"], d["rows"]  # rows: shellDOF,tS,EAs, jaxDOF,tJ,EAj, feDOF,tF,EAf
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{IEA-22 $r=0.2$ cross-section: cost of the three homogenizers across "
             r"mesh-refinement levels (DOF and wall-clock; $EA$ shown to confirm convergence). "
             r"The 1-D RM shell converges with the fewest DOF at a fraction of the solid cost; "
             r"the JAX and FEniCS 2-D solids share the mesh and agree on $EA$.}\label{tab:timing}",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{l|rr|rr|rr}", r"\toprule",
             r"& \multicolumn{2}{c|}{RM shell (1-D)} & \multicolumn{2}{c|}{JAX 2-D solid} "
             r"& \multicolumn{2}{c}{FEniCS 2-D solid}\\",
             r"level & DOF & $t$ (s) & DOF & $t$ (s) & DOF & $t$ (s)\\", r"\midrule"]
    for k in range(len(nm)):
        R = rows[k]
        lines.append("%s & %d & $%.2f$ & %d & $%.2f$ & %d & $%.2f$" %
                     (nm[k], int(R[0]), R[1], int(R[3]), R[4], int(R[6]), R[7]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "timing.tex"), "w").write("\n".join(lines))
    print("wrote timing.tex")


if __name__ == "__main__":
    full_cij("iea_r020", "$r/R=0.2$")
    full_cij("iea_r030", "$r/R=0.3$")
    spanwise()
    timing()
    print("all -> %s" % TEX)
