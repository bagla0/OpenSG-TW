"""Two LaTeX tables for the single-cell [-45] tube: a thick (R/h=2) and a thin
(R/h=10) wall, each in the FE-2D-solid (absolute) | JAX-KL (%) | JAX-RM (%) format
(matching the thesis-style table).  Built from the C6 stiffness files.
  -> tab_aniso_tube_thick.tex, tab_aniso_tube_thin.tex"""
import os
import numpy as np

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\sweep\data"
OUTDIR = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\sweep"
TERMS = [(r"$EA$", 0, 0), (r"$GA_2$", 1, 1), (r"$GA_3$", 2, 2), (r"$GJ$", 3, 3),
         (r"$EI_2$", 4, 4), (r"$EI_3$", 5, 5), (r"$C_{14}$", 0, 3),
         (r"$C_{25}$", 1, 4), (r"$C_{36}$", 2, 5)]


def L(n):
    M = np.loadtxt(os.path.join(DATA, n))
    return 0.5 * (M + M.T)


def pe(M, S, i, j):
    return 100.0 * (M[i, j] - S[i, j]) / S[i, j]


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"$%.4f{\times}10^{%d}$" % (x / 10.0**e, e)


def table(tag, rh, label, word):
    S = L("C6_solid_rh%s.txt" % rh)
    KL = L("C6_jax_kirch_rh%s.txt" % rh)
    RM = L("C6_jax_rm_rh%s.txt" % rh)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Single-cell $[-45^\circ]$ tube, %s wall ($R/h = %d$): Timoshenko "
             r"$6\times6$ stiffness from the FEniCS 2-D solid (absolute) and the percentage error "
             r"of the JAX Kirchhoff (KL) and Reissner--Mindlin (RM) shells. Centric reference, exact "
             r"hoop curvature $k_{22}=-1/R$. Units: $EA,GA$ in N, $GJ,EI$ in N\,m$^2$, couplings in N\,m.}"
             % (word, int(rh)),
             r"\label{%s}" % label,
             r"\renewcommand{\arraystretch}{1.4}\setlength{\tabcolsep}{10pt}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Term & FE-solid & JAX-KL & JAX-RM \\", r"\midrule"]
    for (lab, i, j) in TERMS:
        lines.append(r"%s & %s & $%+.2f\%%$ & $%+.2f\%%$ \\"
                     % (lab, sci(S[i, j]), pe(KL, S, i, j), pe(RM, S, i, j)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUTDIR, "tab_aniso_tube_%s.tex" % tag), "w").write("\n".join(lines) + "\n")
    print("wrote tab_aniso_tube_%s.tex" % tag)


table("thick", "02", "tab:aniso_tube_thick", "thick")
table("thin", "10", "tab:aniso_tube_thin", "thin")
