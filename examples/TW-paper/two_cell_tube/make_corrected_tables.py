"""Corrected 2-cell tables: UNCHANGED shell (KL gradient-junction dshift=t/2, RM
curved dshift=t/2) vs the WEB-CENTERED solid. Prints old->new and writes LaTeX
in the MSG-Solid(VABS) | KL% | RM% format."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
from gradient_kirchhoff import gradient_junction_kirchhoff
from strip_RM import rm_timoshenko_6x6

D = os.path.join(CC, "multicell_tube", "data")
TEX = os.path.join(CC, "multicell_tube", "tex")


def L(n):
    M = np.loadtxt(os.path.join(D, n))
    return 0.5 * (M + M.T)


def pe(M, S, i, j):
    return 100.0 * (M[i, j] - S[i, j]) / S[i, j]


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"$%.4f \times 10^{%d}$" % (x / 10.0**e, e)


DIAG = [(r"C$_{11}$ (EA)", 0, 0), (r"C$_{22}$ (GA$_2$)", 1, 1), (r"C$_{33}$ (GA$_3$)", 2, 2),
        (r"C$_{44}$ (GJ)", 3, 3), (r"C$_{55}$ (EI$_2$)", 4, 4), (r"C$_{66}$ (EI$_3$)", 5, 5)]
C14 = (r"C$_{14}$ (ext-twist)", 0, 3)
WORD = {"iso": "Isotropic", "aniso": "Anisotropic"}
CASES = [("iso", "thin", 0.004, "12.5", "tube2cell_thin.yaml", "C6_solid_tube2cell_thin.txt", "C6_solid_tube2cell_thin_wc.txt"),
         ("iso", "thick", 0.016, "3.1", "tube2cell_thick.yaml", "C6_solid_tube2cell_thick.txt", "C6_solid_tube2cell_thick_wc.txt"),
         ("aniso", "thin", 0.004, "12.5", "tube2cell_aniso_thin.yaml", "C6_solid_tube2cell_aniso_thin.txt", "C6_solid_tube2cell_aniso_thin_wc.txt"),
         ("aniso", "thick", 0.016, "3.1", "tube2cell_aniso_thick.yaml", "C6_solid_tube2cell_aniso_thick.txt", "C6_solid_tube2cell_aniso_thick_wc.txt")]

for mat, thk, t, Rh, mesh, oc6, nc6 in CASES:
    Sold, Snew = L(oc6), L(nc6)
    KL = np.asarray(gradient_junction_kirchhoff(os.path.join(D, mesh), frac=0.0, dshift=t / 2)[0])
    KL = 0.5 * (KL + KL.T)
    RM = np.asarray(rm_timoshenko_6x6(os.path.join(D, mesh), 0.0, dshift=t / 2, curved=True))
    RM = 0.5 * (RM + RM.T)
    terms = DIAG + ([C14] if mat == "aniso" else [])
    print("\n=== %s %s (R/h=%s) :  KL%%  old->new   |   RM%%  old->new ===" % (mat, thk, Rh))
    for lab, i, j in terms:
        print("  %-18s  KL %+6.2f -> %+6.2f   RM %+6.2f -> %+6.2f"
              % (lab, pe(KL, Sold, i, j), pe(KL, Snew, i, j), pe(RM, Sold, i, j), pe(RM, Snew, i, j)))
    # corrected LaTeX table (user's format)
    lines = [r"\begin{table}[htpb]", r"\centering",
             r"\caption{%s wall (%s) --- $R/h = %s$ (web-centered solid)}" % (thk.capitalize(), WORD[mat], Rh),
             r"\label{tab:multicell2_%s_%s}" % (mat, thk),
             r"\begin{tabular}{lccc}", r"\hline",
             r"\multirow{2}{*}{Stiffness} & MSG-Solid & $\%$ Error & $\%$ Error\\",
             r" & (VABS) & (Kirchhoff--Love) & (Reissner--Mindlin) \\", r"\hline"]
    for lab, i, j in terms:
        lines.append(r"%s & %s & $%+.1f$ & $%+.1f$ \\" % (lab, sci(Snew[i, j]), pe(KL, Snew, i, j), pe(RM, Snew, i, j)))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "tab_mc2corr_%s_%s.tex" % (mat, thk)), "w").write("\n".join(lines) + "\n")
print("\nwrote 4 corrected tables to tex/tab_mc2corr_*.tex")
