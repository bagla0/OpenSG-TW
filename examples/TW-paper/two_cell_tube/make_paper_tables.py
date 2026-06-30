"""Enlarged LaTeX %-error tables (KL & RM vs FEniCS-2D-solid, all Timoshenko terms)
for the 2-cell (iso + [-45]) and 4-cell ([-45]) curved tubes.  -> tex/."""
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

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"
TEX = os.path.join(os.path.dirname(DATA), "tex")
os.makedirs(TEX, exist_ok=True)
# (latex term, i, j)
TERMS = [(r"$C_{11}$ (EA)", 0, 0), (r"$C_{22}$ (GA$_2$)", 1, 1), (r"$C_{33}$ (GA$_3$)", 2, 2),
         (r"$C_{44}$ (GJ)", 3, 3), (r"$C_{55}$ (EI$_2$)", 4, 4), (r"$C_{66}$ (EI$_3$)", 5, 5),
         (r"$C_{14}$ (ext-twist)", 0, 3)]


def L(n):
    M = np.loadtxt(os.path.join(DATA, n))
    return 0.5 * (M + M.T)


def errs(solidf, shellf, t):
    S = L(solidf)
    mesh = os.path.join(DATA, shellf)
    KF, _, _ = gradient_junction_kirchhoff(mesh, frac=0.0, dshift=t / 2.0)
    KF = 0.5 * (np.asarray(KF) + np.asarray(KF).T)
    RM = np.asarray(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True))
    RM = 0.5 * (RM + RM.T)
    return S, KF, RM


def pe(M, S, i, j):
    return 100.0 * (M[i, j] - S[i, j]) / S[i, j]


def table(cases, caption, label, terms=TERMS):
    # cases: list of (grouplabel, solidf, shellf, t)
    cols = "l" + "rr" * len(cases)
    head1 = " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % g for g, *_ in cases) + r" \\"
    cmid = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * k, 3 + 2 * k) for k in range(len(cases)))
    head2 = "Term & " + " & ".join(["KL & RM"] * len(cases)) + r" \\"
    data = [errs(s, sh, t) for (_g, s, sh, t) in cases]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label,
             r"\renewcommand{\arraystretch}{1.45}\setlength{\tabcolsep}{8pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{%s}" % cols, r"\toprule", head1, cmid, head2, r"\midrule"]
    for (lab, i, j) in terms:
        row = lab
        for (S, KF, RM) in data:
            # only the OFF-DIAGONAL coupling (C14) can be physically zero (iso -> no ext-twist);
            # diagonal terms (GJ, EI) are always real even when small relative to EA, so never suppress them
            coup = (abs(S[i, i] * S[j, j])) ** 0.5
            if i != j and abs(S[i, j]) < 1e-3 * coup:           # negligible coupling (iso C14 = 0)
                row += r" & \multicolumn{2}{c}{$-$}"
            else:
                row += r" & $%+.1f$ & $\mathbf{%+.1f}$" % (pe(KF, S, i, j), pe(RM, S, i, j))
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


THIN = r"thin ($R/h{=}12.5$)"
THICK = r"thick ($R/h{=}3.1$)"
two_iso = [(THIN, "C6_solid_tube2cell_thin.txt", "tube2cell_thin.yaml", 0.004),
           (THICK, "C6_solid_tube2cell_thick.txt", "tube2cell_thick.yaml", 0.016)]
two_aniso = [(THIN, "C6_solid_tube2cell_aniso_thin.txt", "tube2cell_aniso_thin.yaml", 0.004),
             (THICK, "C6_solid_tube2cell_aniso_thick.txt", "tube2cell_aniso_thick.yaml", 0.016)]
four = [(THIN, "C6_solid_tube4cell_aniso_thin.txt", "tube4cell_aniso_thin.yaml", 0.004),
        (THICK, "C6_solid_tube4cell_aniso_thick.txt", "tube4cell_aniso_thick.yaml", 0.016)]

# isotropic section has no extension-twist coupling -> drop the (all-zero) C14 row
open(os.path.join(TEX, "tab_multicell2_iso.tex"), "w").write(
    table(two_iso, r"Two-cell curved tube, isotropic wall: JAX Kirchhoff (KL) and Reissner--Mindlin (RM) percentage error vs.\ the 2D-solid, for the thin ($R/h{=}12.5$) and thick ($R/h{=}3.1$) wall.", "tab:multicell2_iso", terms=TERMS[:-1]))
open(os.path.join(TEX, "tab_multicell2_aniso.tex"), "w").write(
    table(two_aniso, r"Two-cell curved tube, $[-45^\circ]$ laminate: JAX Kirchhoff (KL) and Reissner--Mindlin (RM) percentage error vs.\ the 2D-solid, for the thin ($R/h{=}12.5$) and thick ($R/h{=}3.1$) wall.", "tab:multicell2_aniso"))
open(os.path.join(TEX, "tab_multicell4.tex"), "w").write(
    table(four, r"Four-cell curved tube, $[-45^\circ]$ laminate: JAX Kirchhoff (KL) and Reissner--Mindlin (RM) percentage error vs.\ the 2D-solid, for the thin ($R/h{=}12.5$) and thick ($R/h{=}3.1$) wall.", "tab:multicell4"))
print("wrote tab_multicell2_iso.tex, tab_multicell2_aniso.tex, tab_multicell4.tex")
