"""Benchmark: RM baseline (shear='mitc') vs tie-both eps13+eps23 (shear='mitc_both', Eq.12 form).
Full diagonal %-error vs the FEniCS 2D-solid / VABS reference on every validated TW case.
Run:  python scripts/rm_research/bench_tie_both.py
"""
import os, sys
import numpy as np
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("opensg_jax", ""):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
from opensg_jax.fe_jax.timo_report import sym, full_pcterr, nonzero_terms

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def pe(m, s):
    return 100.0 * (m - s) / s if s != 0 else float("nan")


def wall_t(mp):
    return sum(float(p[1]) for p in yaml.safe_load(open(mp))["sections"][0]["layup"])


def parse_vabs(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "Timoshenko Stiffness Matrix" in l)
    rows = []
    for l in lines[i + 1:]:
        q = l.split()
        if len(q) == 6:
            try: rows.append([float(x) for x in q])
            except ValueError: continue
        if len(rows) == 6: break
    return np.array(rows)


# (label, mesh_abs, solid_abs, curved, dshift_mode, frac, ref_kind, cutoff)
G = lambda f, m: os.path.join(CC, "tests", "research", f, "data", m)
E = lambda *a: os.path.join(CC, "examples", "data", *a)
CASES = [
    ("SINGLE box 1-cell", G("multicell_box", "box_shell_1cell.yaml"), G("multicell_box", "C6_solid_box1.txt"), False, "half", 0.0, "txt", 1000.0),
    ("SINGLE tube [-45]", G("tube_thesis_314", "shell_center.yaml"), G("tube_thesis_314", "C6_solid_314.txt"), True, "none", 0.0, "txt", 1000.0),
    ("MULTI  box 2-cell", G("multicell_box", "box_shell_2cell.yaml"), G("multicell_box", "C6_solid_box2.txt"), False, "half", 0.0, "txt", 1000.0),
    ("MULTI  box 2-cell THICK", G("multicell_box", "box_shell_2cell_thick.yaml"), G("multicell_box", "C6_solid_box2_thick.txt"), False, "half", 0.0, "txt", 1000.0),
    ("MULTI  tube 2-cell thin", G("multicell_tube", "tube2cell_thin.yaml"), G("multicell_tube", "C6_solid_tube2cell_thin.txt"), True, "half", 0.0, "txt", 1000.0),
    ("MULTI  tube 2-cell THICK", G("multicell_tube", "tube2cell_thick.yaml"), G("multicell_tube", "C6_solid_tube2cell_thick.txt"), True, "half", 0.0, "txt", 1000.0),
    ("mh104 soft-core f0.5", G("mh104_center_ref", "shell_mh104_mid_f020.yaml"), G("mh104_center_ref", "C6_solid_mh104_center_f020.txt"), False, "none", 0.5, "txt", 1000.0),
    ("st15 (vs VABS)", E("1d_yaml", "st15_shell.yaml"), E("benchmark", "st15_vabs.K"), False, "none", 0.0, "vabs", 200.0),
    ("two-cell m45 (vs solid)", E("1d_yaml", "tube2cell_m45_shell.yaml"), E("benchmark", "tube2cell_m45_solid_ref.txt"), True, "half", 0.0, "txt", 1000.0),
]
IB = lambda m: os.path.join(CC, "examples", "data", "iea_blade", m)
for _r in ("020", "030", "040", "050", "060", "070", "080", "090"):
    CASES.append(("IEA-22 r" + _r, IB("shell_r" + _r + ".yaml"), IB("C6_solid_r" + _r + ".txt"),
                  False, "half", 0.0, "txt", 1000.0))


def run(mesh, solid, curved, dsm, frac, kind, shear):
    T = wall_t(mesh)
    ds = T / 2.0 if dsm == "half" else None
    return sym(rm_timoshenko_6x6(mesh, frac, dshift=ds, curved=curved, shear=shear,
                                 v1shear=shear, orient=False))


print("=" * 96)
print("RM tie-both benchmark:  baseline shear='mitc'  vs  tie-both shear='mitc_both' (Eq.12 eps13)")
print("=" * 96)
summary = []
for label, mesh, solid, curved, dsm, frac, kind, cutoff in CASES:
    if not (os.path.exists(mesh) and os.path.exists(solid)):
        print("\n%-26s  SKIP (missing data)" % label); continue
    S = sym(parse_vabs(solid) if kind == "vabs" else np.loadtxt(solid))
    try:
        KL = sym(gradient_junction_kirchhoff(mesh, frac=frac,
                 dshift=(wall_t(mesh)/2.0 if dsm == "half" else None))[0])
    except Exception:
        KL = None
    Rm = run(mesh, solid, curved, dsm, frac, kind, "mitc")
    Rb = run(mesh, solid, curved, dsm, frac, kind, "mitc_both")
    keep = nonzero_terms(S, cutoff)
    em, eb = full_pcterr(Rm, S), full_pcterr(Rb, S)
    mxm = max(abs(em[i, j]) for i, j, _ in keep)
    mxb = max(abs(eb[i, j]) for i, j, _ in keep)
    print("\n%-26s  (curved=%s, dshift=%s)" % (label, curved, dsm))
    print("   term      solid        KL%        RM_mitc%    RM_both%   d(both-mitc)")
    for i in range(6):
        kl = pe(KL[i, i], S[i, i]) if KL is not None else float("nan")
        print("   %-5s  %11.4e  %+8.2f   %+9.2f   %+9.2f    %+7.2f"
              % (LBL[i], S[i, i], kl, pe(Rm[i, i], S[i, i]), pe(Rb[i, i], S[i, i]),
                 pe(Rb[i, i], S[i, i]) - pe(Rm[i, i], S[i, i])))
    print("   max%% over all kept terms:   RM_mitc=%.2f   RM_both=%.2f   -> %s"
          % (mxm, mxb, "BOTH better" if mxb < mxm - 0.05 else ("baseline better" if mxb > mxm + 0.05 else "tie")))
    summary.append((label, mxm, mxb))

print("\n" + "=" * 96)
print("SUMMARY  (max %% error over kept terms)")
print("%-28s  %10s  %10s  %s" % ("case", "RM_mitc", "RM_both", "winner"))
for label, mxm, mxb in summary:
    win = "both" if mxb < mxm - 0.05 else ("baseline" if mxb > mxm + 0.05 else "tie")
    print("%-28s  %10.2f  %10.2f  %s" % (label, mxm, mxb, win))
print("=" * 96)
