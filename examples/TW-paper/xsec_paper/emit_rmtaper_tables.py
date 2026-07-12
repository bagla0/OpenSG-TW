"""emit_rmtaper_tables.py -- cross-section result tables in the RM-TAPER paper style:
one row per nonzero Timoshenko constant C^b_ij, columns = solid | shell | %err.
Reads the saved npz (xsec_5v6_master results).  -> results/tex_rm/*.tex

Row rule (physically meaningful "nonzero"): always keep the six diagonal terms
C^b_11..C^b_66; keep an off-diagonal coupling (i,j) only when
|C_ij| > 0.02 * sqrt(|C_ii C_jj|).  This avoids the max/1e3 cutoff dropping the
small-but-real torsion/bending terms of thin sections.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TEX = os.path.join(RES, "tex_rm"); os.makedirs(TEX, exist_ok=True)


def _sym(M):
    return 0.5 * (np.asarray(M) + np.asarray(M).T)


def _rows(So):
    """row-major upper-triangle order, diagonals always, real couplings only."""
    ij = []
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                ij.append((i, j))
            else:
                rel = abs(So[i, j]) / (np.sqrt(abs(So[i, i] * So[j, j])) + 1e-30)
                if rel > 0.02:
                    ij.append((i, j))
    return ij


def _fmt(x, scale):
    v = x / scale
    if abs(v) < 1e-4:
        return "$\\sim\\!0$"
    return "$%.4g$" % v


def one_case(solid, shell, scale, exp, caption, label):
    So, Sh = _sym(solid), _sym(shell)
    lines = [r"\begin{table}[htpb]", r"\centering",
             r"\caption{%s ($\times10^{%d}$).}" % (caption, exp), r"\label{%s}" % label,
             r"\begin{tabular}{l rrr}", r"\hline",
             r"$C^{b}_{ij}$ & solid & shell & \%err \\", r"\hline"]
    for (i, j) in _rows(So):
        e = 100.0 * (Sh[i, j] - So[i, j]) / So[i, j]
        lines.append(r"$C^{b}_{%d%d}$ & %s & %s & $%+.1f$ \\"
                     % (i + 1, j + 1, _fmt(So[i, j], scale), _fmt(Sh[i, j], scale), e))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, label.split(":")[-1] + ".tex"), "w").write("\n".join(lines))
    print("wrote", label)


def drill_inset(solid, c6, c5, scale, exp, label):
    """compact torsion comparison: solid | 6-DOF (Lagrange) | 5-DOF (eliminated)."""
    So, S6, S5 = _sym(solid), _sym(c6), _sym(c5)
    t0, t6, t5 = So[3, 3], S6[3, 3], S5[3, 3]
    e6 = 100.0 * (t6 - t0) / t0
    e5 = 100.0 * (t5 - t0) / t0
    lines = [r"\begin{table}[htpb]", r"\centering",
             r"\caption{Two-cell tube torsion $C^{b}_{44}$: the six-parameter "
             r"drilling-constrained (Lagrange) element against the five-parameter "
             r"drilling-eliminated element ($\times10^{%d}$). The Lagrange constraint "
             r"recovers the drilling-carried torsional shear that the flat-wall "
             r"elimination destroys.}" % exp, r"\label{%s}" % label,
             r"\begin{tabular}{l rr}", r"\hline",
             r"model & $C^{b}_{44}$ & \%err \\", r"\hline",
             r"solid (2-D reference) & %s & --- \\" % _fmt(t0, scale),
             r"RM 6-DOF (Lagrange) & %s & $%+.1f$ \\" % (_fmt(t6, scale), e6),
             r"RM 5-DOF (eliminated) & %s & $%+.1f$ \\" % (_fmt(t5, scale), e5),
             r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, label.split(":")[-1] + ".tex"), "w").write("\n".join(lines))
    print("wrote", label)


def from_npz(tag, scale, exp, caption, label, skey="solid", rkey="c6dof"):
    d = np.load(os.path.join(RES, "%s.npz" % tag))
    one_case(d[skey], d[rkey], scale, exp, caption, label)


if __name__ == "__main__":
    from_npz("single_tube", 1e7, 7,
             r"Single-cell $[-45^\circ]$ tube ($R=7.15$~cm, $t=8.68$~mm): "
             r"RM 6-DOF shell cross-section vs.\ the 2-D solid", "tab:single")
    from_npz("two_cell_tube", 1e7, 7,
             r"Webbed two-cell tube (isotropic): RM 6-DOF shell vs.\ the 2-D solid", "tab:two")
    d = np.load(os.path.join(RES, "two_cell_tube.npz"))
    drill_inset(d["solid"], d["c6dof"], d["c5dof"], 1e4, 4, "tab:twodrill")
    from_npz("iea_r020", 1e9, 9,
             r"IEA-22 blade cross-section at $r/R=0.2$: RM 6-DOF shell vs.\ the 2-D solid", "tab:iea020")
    from_npz("iea_r030", 1e9, 9,
             r"IEA-22 blade cross-section at $r/R=0.3$: RM 6-DOF shell vs.\ the 2-D solid", "tab:iea030")
    print("all ->", TEX)
